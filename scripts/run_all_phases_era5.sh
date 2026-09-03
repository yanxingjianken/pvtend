#!/usr/bin/env bash
# ----------------------------------------------------------------------
# pvtend Phase 6 → 13 supervisor (v2.10.9)
#   v2.10.9: export PVTEND_COMPOSITE_WORKERS=16 by default so the
#            pass-2 composite accumulator (build_composites) runs in
#            parallel for both p8_composite_* and p10_website_*.
#
# Runs every remaining pipeline stage in a single nohup-able loop.
# - Each stage is wrapped in `setsid` so its process group is isolated.
# - Idle-CPU child reaper kills python workers that have been at 0% CPU
#   for > $ORPHAN_TIMEOUT seconds (multiprocessing leftovers).
# - On stage failure (exit != 0) or abrupt termination the script
#   sleeps RELAUNCH_DELAY then re-runs that exact stage; --skip-existing
#   on `compute` ensures NPZs that did finish are reused.
# - Stage progress is checkpointed in $STATE so a crash of the
#   supervisor itself can be resumed by re-running this script.
#
# Usage:
#   nohup scripts/run_all_phases_era5.sh > /net/flood/data2/users/x_yan/tmp/pvtend_phases.log 2>&1 &
# ----------------------------------------------------------------------
set -u
shopt -s lastpipe

# ---- Config ----------------------------------------------------------
ROOT="${ROOT:-/net/flood/data2/users/x_yan/pvtend}"
ERA5_DIR="${ERA5_DIR:-/net/flood/data2/users/x_yan/era}"
CLIM_DIR="${CLIM_DIR:-/net/flood/data2/users/x_yan/era/clim}"
# pvtend climatology loader treats --clim-path as a stem: files matching
# {stem}_{mon}_{var}.nc in stem.parent. So we pass a sentinel .nc path.
CLIM_STEM="${CLIM_STEM:-$CLIM_DIR/era5_hourly_clim_1990-2020.nc}"
TMP="${TMP:-/net/flood/data2/users/x_yan/tmp}"
STATE="${STATE:-$TMP/pvtend_phases.state}"
LOG_DIR="${LOG_DIR:-$TMP/pvtend_phases}"

WORKERS="${WORKERS:-48}"
# Parallel workers for pass-2 composite accumulation (build_composites).
# Used by p8_composite_* (via pvtend-pipeline composite) and
# p10_website_* (via website/build_and_export_clusters.py). Exported so
# every setsid child inherits it.
export PVTEND_COMPOSITE_WORKERS="${PVTEND_COMPOSITE_WORKERS:-16}"
ORPHAN_TIMEOUT="${ORPHAN_TIMEOUT:-1800}"   # 30 min idle → kill
CHECK_INTERVAL="${CHECK_INTERVAL:-120}"
RELAUNCH_DELAY="${RELAUNCH_DELAY:-30}"
MAX_RETRIES="${MAX_RETRIES:-5}"

ENV_RUN="micromamba run -n blocking"
PIPELINE="$ENV_RUN pvtend-pipeline"
PY="$ENV_RUN python"

BLK_CSV="$ROOT/docs/_static/ERA5_TempestExtremes_z500_anticyclone_blocking.csv"
PRP_CSV="$ROOT/docs/_static/ERA5_TempestExtremes_z500_exclusif_anticyclone_propagating.csv"
BLK_OUT="$ROOT/outputs/era5_blocking"
PRP_OUT="$ROOT/outputs/era5_prp"
EXC_DIR="$ROOT/outputs/blowup_scan"

mkdir -p "$LOG_DIR" "$EXC_DIR"

# ---- State / logging -------------------------------------------------
log()    { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/supervisor.log"; }
mark()   { grep -v "^$1=" "$STATE" 2>/dev/null > "$STATE.tmp" || true
           echo "$1=$2" >> "$STATE.tmp"
           mv "$STATE.tmp" "$STATE"; }
done_p() { [[ -f "$STATE" ]] && grep -q "^$1=done$" "$STATE"; }

# ---- Reparented orphan sweeper ---------------------------------------
# Catches multiprocessing-fork python workers that have been reparented
# to init (PPID=1) after their parent pvtend-pipeline was killed.
# These leaks consume ~30 GB RSS each and can exhaust swap; they cannot
# belong to any healthy run, so we kill them unconditionally.
sweep_reparented_orphans() {
    local pids
    pids=$(ps -u "$USER" -o pid,ppid,cmd --no-headers 2>/dev/null \
           | awk '$2==1 && /multiprocessing-fork/ {print $1}')
    if [[ -n "$pids" ]]; then
        local n; n=$(echo "$pids" | wc -l)
        log "ORPHAN SWEEP: terminating $n reparented multiprocessing workers"
        echo "$pids" | xargs -r kill -TERM 2>/dev/null
        sleep 5
        pids=$(ps -u "$USER" -o pid,ppid,cmd --no-headers 2>/dev/null \
               | awk '$2==1 && /multiprocessing-fork/ {print $1}')
        if [[ -n "$pids" ]]; then
            echo "$pids" | xargs -r kill -KILL 2>/dev/null
            log "ORPHAN SWEEP: SIGKILL fallback applied"
        fi
    fi
}

# ---- Orphan reaper (background) --------------------------------------
reap_orphans() {
    local parent="$1"
    while kill -0 "$parent" 2>/dev/null; do
        # Find python descendants of $parent that have been in S state
        # with 0% CPU for >$ORPHAN_TIMEOUT seconds.
        ps -eo pid,ppid,etimes,pcpu,stat,comm --no-headers 2>/dev/null \
        | awk -v par="$parent" -v to="$ORPHAN_TIMEOUT" '
            $6 ~ /python/ && $3 > to && $4+0 == 0 && $5 ~ /S/ {print $1}
        ' | while read -r pid; do
            # Walk up to verify ancestry to $parent
            cur=$pid
            for _ in 1 2 3 4 5 6; do
                [[ $cur -eq 1 ]] && break
                pp=$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d ' ')
                [[ -z $pp ]] && break
                if [[ $pp -eq $parent ]]; then
                    echo "[$(date '+%F %T')] reap idle orphan pid=$pid" >> "$LOG_DIR/supervisor.log"
                    kill -TERM "$pid" 2>/dev/null
                    break
                fi
                cur=$pp
            done
        done
        sleep "$CHECK_INTERVAL"
    done
}

# ---- Run-with-retry helper -------------------------------------------
run_stage() {
    local key="$1"; shift
    local desc="$1"; shift
    if done_p "$key"; then
        log "SKIP  $key — already done"
        return 0
    fi
    local stagelog="$LOG_DIR/${key}.log"
    local attempt=0
    while (( attempt < MAX_RETRIES )); do
        attempt=$((attempt+1))
        log "STAGE $key  attempt=$attempt   $desc"
        log "      cmd: $*"
        # setsid → new session/process group
        setsid bash -c "$*" >>"$stagelog" 2>&1 &
        local pgid=$!
        reap_orphans "$pgid" &
        local reaper=$!
        wait "$pgid"
        local rc=$?
        kill "$reaper" 2>/dev/null; wait "$reaper" 2>/dev/null
        if [[ $rc -eq 0 ]]; then
            log "DONE  $key (rc=0)"
            mark "$key" done
            sweep_reparented_orphans
            return 0
        fi
        log "FAIL  $key (rc=$rc) — sleeping $RELAUNCH_DELAY s before retry"
        sweep_reparented_orphans
        sleep "$RELAUNCH_DELAY"
    done
    log "GIVE UP on $key after $MAX_RETRIES attempts"
    return 1
}

cd "$ROOT"

# Initial sweep: kill any reparented multiprocessing workers left over
# from previously-killed pipelines before we start consuming RAM again.
sweep_reparented_orphans

# ────────────────────────────────────────────────────────────────────
#  ORDERING (2026-05-10): blocking-first.
#  Phases 6→8→10 are completed for **blocking** before any prp work
#  begins, so downstream analysis on blocking can start while we
#  generate prp.  Stage keys preserve their original names so the
#  $STATE checkpoint remains compatible with prior runs.
# ────────────────────────────────────────────────────────────────────

run_p6_compute() {
    # $1=evt  $2=csv  $3=out  $4=stage
    local evt="$1" csv="$2" out="$3" stage="$4"
    run_stage "p6_${evt}_${stage}" \
        "compute $evt/$stage NPZs (skip-existing)" \
        "$PIPELINE compute \
            --event-type $evt \
            --events-csv '$csv' \
            --era5-dir '$ERA5_DIR' \
            --clim-path '$CLIM_STEM' \
            --clim-helmholtz-dir '$CLIM_DIR' \
            --out-dir '$out' \
            --stages $stage \
            --dh-range='-12:13:1' \
            --qg-method log20 \
            --n-workers $WORKERS \
            --skip-existing && \
         { n=\$(find '$out/$stage' -name 'track_*.npz' 2>/dev/null | wc -l); \
           echo \"[verify] $evt/$stage NPZ count=\$n\"; \
           [ \"\$n\" -gt 0 ]; }"
}

run_variant() {
    # Runs the full p6→p7→p8→p10 stack for one event variant.
    # $1=evt   $2=csv   $3=out
    local evt="$1" csv="$2" out="$3"
    for stage in onset peak decay; do
        run_p6_compute "$evt" "$csv" "$out" "$stage"
    done

    run_stage "p7_excl_${evt}" "aggregate QG blowup → exclude_tracks_${evt}" \
        "$PY $ROOT/scripts/aggregate_qg_blowup.py \
             --npz-dir '$out' --threshold 25.0 \
             --out '$EXC_DIR/exclude_tracks_${evt}.csv' \
             --report '$EXC_DIR/qg_blowup_report_${evt}.csv'"

    run_stage "p8_classify_${evt}" "classify RWB wavg ${evt}" \
        "$PIPELINE classify \
             --npz-dir '$out' \
             --output '$out/rwb_variant_tracksets_wavg.pkl' \
             --levels wavg \
             --threshold 1 \
             --contours circumpolar \
             --source era5 \
             --archive-dir '$ERA5_DIR' \
             --exclude-file '$EXC_DIR/exclude_tracks_${evt}.csv'"

    run_stage "p8_composite_${evt}" "composite ${evt}" \
        "$PIPELINE composite \
             --npz-dir '$out' \
             --rwb-pkl '$out/rwb_variant_tracksets_wavg.pkl' \
             --pkl-out '$out/composite.pkl' \
             --exclude-file '$EXC_DIR/exclude_tracks_${evt}.csv'"

    run_stage "p10_website_${evt}" "build_and_export_clusters ${evt}" \
        "rm -rf '$ROOT/website/${evt}_export' && \
         $PY $ROOT/website/build_and_export_clusters.py --event-type $evt \
           --rwb-pkl '$out/rwb_variant_tracksets_wavg.pkl'"

    # HF upload for this variant only.  upload_to_hf.py already deletes any
    # stale files in the remote subfolder that aren't in the local set.
    # We upload blocking first (so the website dropdown picks it up while
    # prp NPZs are still being generated), then prp at the end.
    if [[ "$evt" == "blocking" ]]; then
        run_stage "p10_hf_${evt}" "HF upload ${evt} (skip prp)" \
            "$PY $ROOT/website/upload_to_hf.py --skip-prp --skip-cleanup"
    else
        # Final prp upload + repo-root cleanup of old top-level files.
        run_stage "p10_hf_${evt}" "HF upload ${evt} + root cleanup" \
            "$PY $ROOT/website/upload_to_hf.py --skip-blocking"
    fi
}

# ── Variant A: blocking (p6→p8→p10_website→p10_hf_blocking) ──────────
run_variant "blocking" "$BLK_CSV" "$BLK_OUT"

# ── Variant B: prp     (p6→p8→p10_website→p10_hf_prp + root cleanup) ─
run_variant "prp"      "$PRP_CSV" "$PRP_OUT"

log "ALL PHASES COMPLETE"



