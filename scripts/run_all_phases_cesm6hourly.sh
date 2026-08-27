#!/usr/bin/env bash
# ----------------------------------------------------------------------
# pvtend supervisor — CESM2-LENS2 smbb 6-hourly branch (members 91-100)
#
# Drives the full dh=0 regeneration for both event types against the
# Wu-9-level NH archive:
#   p0  preflight        archive/clim/catalogue contract checks (fail fast)
#   p6  compute          per member x {blocking,prp} x {onset,peak,decay},
#                        --source cesm, dh=0 only, PPVI inline
#   p7  blowup exclude   aggregate_qg_blowup.py --fields all
#   p8  classify         RWB wavg -> rwb_variant_tracksets_wavg.pkl
#   p8b composite        variant-aware composite.pkl
# No website/HF phases on this branch.
#
# Same supervision model as run_all_phases_era5.sh: every stage runs in
# its own setsid process group, is checkpointed in $STATE, and retried
# up to $MAX_RETRIES on failure; `--skip-existing` makes compute retries
# cheap. The idle reaper measures REAL idleness (delta-CPU over the
# whole timeout window), not lifetime-average CPU, so workers blocked on
# slow NFS reads are not killed.
#
# Usage:
#   nohup scripts/run_all_phases_cesm6hourly.sh \
#       > /net/flood/data2/users/x_yan/tmp/pvtend_cesm6h.log 2>&1 &
# ----------------------------------------------------------------------
set -u
shopt -s lastpipe

# ---- Config ----------------------------------------------------------
ROOT="${ROOT:-/net/flood/data2/users/x_yan/pvtend}"
ARCHIVE="${ARCHIVE:-$ROOT/archive/cesm2_lens2_wu9_nh}"
CLIM="${CLIM:-$ROOT/clim/LENS2_smbb91_100_wu9_clim_6hourly_1985_2014.nc}"
CAT="${CAT:-$ROOT/catalogues/events}"
OUT_BLK="${OUT_BLK:-$ROOT/outputs/cesm6hourly_blocking}"
OUT_PRP="${OUT_PRP:-$ROOT/outputs/cesm6hourly_prp}"
EXC_DIR="${EXC_DIR:-$ROOT/outputs/blowup_scan}"
TMP="${TMP:-/net/flood/data2/users/x_yan/tmp}"
STATE="${STATE:-$TMP/pvtend_cesm6h.state}"
LOG_DIR="${LOG_DIR:-$TMP/pvtend_cesm6h}"

MEMBERS="${MEMBERS:-91 92 93 94 95 96 97 98 99 100}"
WORKERS="${WORKERS:-64}"
export PVTEND_COMPOSITE_WORKERS="${PVTEND_COMPOSITE_WORKERS:-16}"
CLASSIFY_WORKERS="${CLASSIFY_WORKERS:-16}"
ORPHAN_TIMEOUT="${ORPHAN_TIMEOUT:-1800}"   # 30 min truly idle → kill
CHECK_INTERVAL="${CHECK_INTERVAL:-120}"
RELAUNCH_DELAY="${RELAUNCH_DELAY:-30}"
MAX_RETRIES="${MAX_RETRIES:-5}"

ENV_RUN="micromamba run -n blocking"
PIPELINE="$ENV_RUN pvtend-pipeline"
PY="$ENV_RUN python"

mkdir -p "$LOG_DIR" "$EXC_DIR" "$OUT_BLK" "$OUT_PRP"

# ---- State / logging -------------------------------------------------
log()    { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/supervisor.log"; }
mark()   { grep -v "^$1=" "$STATE" 2>/dev/null > "$STATE.tmp" || true
           echo "$1=$2" >> "$STATE.tmp"
           mv "$STATE.tmp" "$STATE"; }
done_p() { [[ -f "$STATE" ]] && grep -q "^$1=done$" "$STATE"; }

# ---- Reparented orphan sweeper ---------------------------------------
# multiprocessing workers reparented to init (PPID=1) cannot belong to
# any healthy run; kill unconditionally.
sweep_reparented_orphans() {
    local pids
    pids=$(ps -u "$USER" -o pid,ppid,cmd --no-headers 2>/dev/null \
           | awk '$2==1 && /multiprocessing-(fork|spawn)/ {print $1}')
    if [[ -n "$pids" ]]; then
        local n; n=$(echo "$pids" | wc -l)
        log "ORPHAN SWEEP: terminating $n reparented multiprocessing workers"
        echo "$pids" | xargs -r kill -TERM 2>/dev/null
        sleep 5
        pids=$(ps -u "$USER" -o pid,ppid,cmd --no-headers 2>/dev/null \
               | awk '$2==1 && /multiprocessing-(fork|spawn)/ {print $1}')
        if [[ -n "$pids" ]]; then
            echo "$pids" | xargs -r kill -KILL 2>/dev/null
            log "ORPHAN SWEEP: SIGKILL fallback applied"
        fi
    fi
}

# ---- Idle reaper (background, delta-CPU based) -----------------------
# Kills python descendants of $parent whose cumulative CPU time has not
# advanced for > $ORPHAN_TIMEOUT s. Unlike an etimes/pcpu test, a worker
# blocked minutes on an NFS read still advances CPU between windows and
# survives; only a truly wedged process shows zero delta over 30 min.
reap_orphans() {
    local parent="$1"
    declare -A prev_cpu idle_since
    while kill -0 "$parent" 2>/dev/null; do
        local now; now=$(date +%s)
        ps -eo pid,ppid,cputimes,comm --no-headers 2>/dev/null \
        | awk -v par="$parent" '$4 ~ /python/ {print $1, $2, $3}' \
        | while read -r pid ppid cpu; do
            # verify ancestry to $parent (up to 6 hops)
            cur=$pid; anc=0
            for _ in 1 2 3 4 5 6; do
                [[ $cur -eq 1 ]] && break
                pp=$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d ' ')
                [[ -z $pp ]] && break
                [[ $pp -eq $parent ]] && { anc=1; break; }
                cur=$pp
            done
            [[ $anc -eq 1 ]] && echo "$pid $cpu"
        done | {
            while read -r pid cpu; do
                if [[ "${prev_cpu[$pid]:-}" == "$cpu" ]]; then
                    local since="${idle_since[$pid]:-$now}"
                    idle_since[$pid]=$since
                    if (( now - since > ORPHAN_TIMEOUT )); then
                        echo "[$(date '+%F %T')] reap idle (no CPU for $((now-since))s) pid=$pid" \
                            >> "$LOG_DIR/supervisor.log"
                        kill -TERM "$pid" 2>/dev/null
                        unset "idle_since[$pid]" "prev_cpu[$pid]"
                        continue
                    fi
                else
                    idle_since[$pid]=$now
                fi
                prev_cpu[$pid]=$cpu
            done
        }
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
sweep_reparented_orphans

# ── p0: preflight — fail loudly before burning days ───────────────────
run_stage "p0_preflight" "archive/clim/catalogue contract checks" \
    "$PY - <<'PYEOF'
import sys
import xarray as xr

CLIM = '$CLIM'
ARCHIVE = '$ARCHIVE'
CAT = '$CAT'
MEMBERS = '$MEMBERS'.split()
HELM_BARS = ('u_rot_bar', 'u_div_bar', 'v_rot_bar', 'v_div_bar')
WU_PLEVS = [1000, 850, 700, 500, 400, 300, 250, 200, 100]
errors = []

clim = xr.open_dataset(CLIM, chunks={'slot': 1})
sizes = dict(clim.sizes)
if sizes.get('slot') != 1460:
    errors.append(f'clim slot dim != 1460: {sizes}')
if 96 not in sizes.values() or 288 not in sizes.values():
    errors.append(f'clim lacks 96x288 lat/lon dims: {sizes}')
missing_bars = [b for b in HELM_BARS if b not in clim.variables]
if missing_bars:
    errors.append(f'clim lacks folded Helmholtz bars {missing_bars} — '
                  'the CESM compute path requires them inside the slot clim')
plev_name = next((c for c in ('plev', 'pressure_level', 'lev')
                  if c in clim.coords), None)
if plev_name is None:
    errors.append(f'clim has no pressure coordinate: {list(clim.coords)}')
else:
    plevs = [int(round(float(v))) for v in clim[plev_name].values]
    if plevs != WU_PLEVS and plevs != WU_PLEVS[::-1]:
        errors.append(f'clim plev != Wu levels: {plevs}')
for v in ('pv', 'PV'):
    if v in clim.variables:
        print(f'clim {v} units attr: {clim[v].attrs.get(\"units\", \"<none>\")}')
clim.close()

import os
for m in MEMBERS:
    fp = os.path.join(ARCHIVE, f'lens2_smbb_m{int(m)}_1985_plev.nc')
    if not os.path.exists(fp):
        errors.append(f'archive missing {fp}')
    for evt in ('blocking', 'prp'):
        csv = os.path.join(CAT, f'events_{evt}_m{int(m):03d}.csv')
        if not os.path.exists(csv):
            errors.append(f'catalogue missing {csv}')

import csv as _csv
probe = os.path.join(CAT, f'events_blocking_m{int(MEMBERS[0]):03d}.csv')
if os.path.exists(probe):
    with open(probe) as fh:
        header = next(_csv.reader(fh))
    need_any = [('evt_name', 'type', 'stage'), ('track_id',),
                ('lat0', 'lat'), ('lon0', 'lon180'),
                ('base_ts', 'timestamp')]
    for alts in need_any:
        if not any(a in header for a in alts):
            errors.append(f'{probe}: none of {alts} in header {header}')

first = os.path.join(ARCHIVE, f'lens2_smbb_m{int(MEMBERS[0])}_1985_plev.nc')
if os.path.exists(first):
    ds = xr.open_dataset(first)
    print(f'archive vars: {sorted(ds.data_vars)}')
    print(f'archive dims: {dict(ds.sizes)}')
    for want in ('U', 'V', 'OMEGA', 'PV', 'Z3', 'T', 'Q'):
        if want not in ds.variables:
            errors.append(f'archive lacks CAM variable {want} '
                          f'(has {sorted(ds.data_vars)})')
    if 'PV' in ds.variables:
        print(f'archive PV units attr: {ds.PV.attrs.get(\"units\", \"<none>\")}')
    ds.close()

if errors:
    print('PREFLIGHT FAILED:')
    for e in errors:
        print(' -', e)
    sys.exit(1)
print('PREFLIGHT OK')
PYEOF"

# ── p6: compute, sequential members for resumability ──────────────────
for m in $MEMBERS; do
    m03=$(printf "m%03d" "$m")
    for evt in blocking prp; do
        out="$OUT_BLK"; [[ $evt == prp ]] && out="$OUT_PRP"
        csv="$CAT/events_${evt}_${m03}.csv"
        for stage in onset peak decay; do
            run_stage "p6_cesm_${evt}_${m03}_${stage}" \
                "compute $evt/$stage member $m (dh=0, PPVI inline)" \
                "$PIPELINE compute \
                    --event-type $evt \
                    --source cesm \
                    --member $m \
                    --events-csv '$csv' \
                    --era5-dir '$ARCHIVE' \
                    --clim-path '$CLIM' \
                    --out-dir '$out' \
                    --stages $stage \
                    --dh-range='0:1' \
                    --qg-method log20 \
                    --n-workers $WORKERS \
                    --skip-existing && \
                 { n=\$(find '$out/$stage/dh=+0' -name 'track_${m03}_*.npz' 2>/dev/null | wc -l); \
                   echo \"[verify] $evt/$stage $m03 NPZ count=\$n\"; \
                   [ \"\$n\" -gt 0 ]; }"
        done
    done
done

# ── p7: blowup exclusion (omega + divergent wind, all levels) ─────────
for evt in blocking prp; do
    out="$OUT_BLK"; [[ $evt == prp ]] && out="$OUT_PRP"
    run_stage "p7_excl_cesm_${evt}" "blowup scan → exclude_tracks_cesm6hourly_${evt}" \
        "$PY $ROOT/scripts/aggregate_qg_blowup.py \
             --npz-dir '$out' --fields all \
             --out '$EXC_DIR/exclude_tracks_cesm6hourly_${evt}.csv' \
             --report '$EXC_DIR/blowup_report_cesm6hourly_${evt}.csv'"
done

# ── p8: RWB classify + composite ──────────────────────────────────────
for evt in blocking prp; do
    out="$OUT_BLK"; [[ $evt == prp ]] && out="$OUT_PRP"
    run_stage "p8_classify_cesm_${evt}" "classify RWB wavg cesm6hourly ${evt}" \
        "$PIPELINE classify \
             --npz-dir '$out' \
             --output '$out/rwb_variant_tracksets_wavg.pkl' \
             --levels wavg \
             --threshold 1 \
             --n-workers $CLASSIFY_WORKERS \
             --exclude-file '$EXC_DIR/exclude_tracks_cesm6hourly_${evt}.csv'"

    run_stage "p8_composite_cesm_${evt}" "composite cesm6hourly ${evt}" \
        "$PIPELINE composite \
             --npz-dir '$out' \
             --rwb-pkl '$out/rwb_variant_tracksets_wavg.pkl' \
             --pkl-out '$out/composite.pkl' \
             --n-workers $PVTEND_COMPOSITE_WORKERS \
             --exclude-file '$EXC_DIR/exclude_tracks_cesm6hourly_${evt}.csv'"
done

log "ALL CESM 6-HOURLY PHASES COMPLETE"
