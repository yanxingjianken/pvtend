#!/usr/bin/env bash
# CESM2-LENS2 smbb 6-hourly (members 91-100) tendency + piecewise PV inversion
# at the event time only (dh = 0).
#
# Sibling of run_era5_dh0.sh: same state file, same per-block markers, same
# --skip-existing resume. The one structural difference is the member. The
# CESM archive is one file per member and year, and `compute --source cesm`
# takes the member as an argument because it cannot be inferred from an event
# row, so a block here is (event type, stage, member) and the ten members run
# sequentially inside a block: only one member-year file is open at a time and
# a resume re-reads at most one member's worth of catalogue.
#
# Worker count. The inversion is the global spherical solver on its own T84
# Gaussian grid, identical for both archives, and its cost is dominated by
# spectral transforms streaming a Legendre table through the CPU cache. That
# makes memory bandwidth, not storage, the ceiling: on this node the ERA5 run
# measured 24.66 events/min at 48 workers, 32.36 at 96 and 31.95 at 144, with
# the D-state count back to one or two within a sampling interval of the
# workers' first read. Ninety-six is the knee there, and the inversion that
# sets it is the same calculation here, so ninety-six is the default. It is
# not a proven knee for this archive: the CESM data grid carries 96x288 points
# against ERA5's 61x240, which is nearly twice the traffic through the parts
# that are not the solver -- the regrid onto the solver grid, the QG-omega
# solves, the Helmholtz decomposition -- so a run of this data could saturate
# earlier. The first block says which: it logs its own events/min, and below
# about twenty-five with the node otherwise quiet, stop and re-measure with
# RAMP. Set RAMP to measure the knee outright.
#
# Usage:
#   scripts/run_cesm6h_dh0.sh                     # everything at 96 workers
#   WORKERS=64 scripts/run_cesm6h_dh0.sh          # different fixed count
#   RAMP="48 96 144" scripts/run_cesm6h_dh0.sh    # re-measure the knee first
#   MEMBERS="91 92" scripts/run_cesm6h_dh0.sh     # a subset of the ensemble
#
# Resume: rerun it. Every block is skipped once its marker is in the state file,
# and every event is skipped once its NPZ exists.
set -uo pipefail

ROOT="${ROOT:-/net/flood/data2/users/x_yan/pvtend}"
# Inputs live with the other CESM data; pvtend holds only the package,
# outputs/ and paper/.
C6H="${C6H:-/net/flood/data2/users/x_yan/cesm-blocking/cesm_6hourly}"
ARCHIVE="${ARCHIVE:-$C6H/cesm2_lens2_wu9_nh}"
# The folded Helmholtz bars (u/v_rot_bar, u/v_div_bar) live inside this slot
# climatology, so the Helmholtz directory is its own and needs no flag.
CLIM="${CLIM:-$C6H/clim/LENS2_smbb91_100_wu9_clim_6hourly_1985_2014.nc}"
CAT="${CAT:-$C6H/catalogues/events}"
BLK_OUT="${BLK_OUT:-$ROOT/outputs/cesm6hourly_blocking}"
PRP_OUT="${PRP_OUT:-$ROOT/outputs/cesm6hourly_prp}"
TMP="${TMP:-/net/flood/data2/users/x_yan/tmp}"
LOG_DIR="${LOG_DIR:-$TMP/pvtend_cesm6h_dh0}"
STATE="${STATE:-$LOG_DIR/state}"
MEMBERS="${MEMBERS:-91 92 93 94 95 96 97 98 99 100}"
WORKERS="${WORKERS:-96}"                      # the measured knee; see header
RAMP="${RAMP:-}"                              # set to re-measure it
RAMP_SIZES="${RAMP_SIZES:-360 720 1080 1620}" # catalogue ROWS per rung; a third of them
                                              # are the stage being computed
PIECES="${PIECES:-scale}"
# Worker recycling. The pool wedges -- every worker gone, the parent asleep with
# no processor time accruing and nothing timing out -- and on the other archive
# it did so three times out of three, always in a block long enough to reach the
# recycling point and never in one that was not: three blocks stopped at 1533,
# 1534 and 1536 events with 96 workers, which is 96 x 16, the default. Recycling
# bounds per-worker memory, so it is not removed but pushed past the longest
# block here (about 1,900 events, twenty per worker); a block boundary still
# gives every worker a fresh process. Setting this to 0 would also silently
# switch the pool from spawn to fork, which is worse.
RECYCLE="${RECYCLE:-64}"

ENV_RUN="micromamba run -n blocking"
PIPELINE="$ENV_RUN pvtend-pipeline"

mkdir -p "$LOG_DIR" "$BLK_OUT" "$PRP_OUT"
touch "$STATE"

log()  { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/supervisor.log" >&2; }
done_p() { grep -qx "$1" "$STATE"; }
mark()   { echo "$1" >> "$STATE"; }

csv_for() {  # event type, member -> catalogue path
    printf '%s/events_%s_m%03d.csv' "$CAT" "$1" "$2"
}

# ---- Health: workers blocked on storage rather than working ---------------
# A run that saturates the filesystem shows as processes in uninterruptible
# sleep. Compute-bound workers stay runnable, so a block with a D-state count
# comparable to its worker count is not going faster by being larger.
sample_health() {
    local tag=$1 seconds=$2 out="$LOG_DIR/health_$tag.log"
    # The sampler writes to its own file and its inherited standard output is
    # closed below: a background job that keeps the pipe of this function's
    # command substitution open would make the caller wait for the whole run.
    (
        local end=$((SECONDS + seconds))
        while (( SECONDS < end )); do
            local d r
            d=$(ps -u "$USER" -o state= | grep -c '^D')
            r=$(ps -u "$USER" -o state= | grep -c '^R')
            local rss
            rss=$(ps -u "$USER" -o rss=,args= | awk '/spawn_main/ {s += $1} END {printf "%.0f", s / 1048576}')
            echo "$(date '+%F %T') D=$d R=$r load=$(cut -d' ' -f1 /proc/loadavg) workerRSS=${rss}G" >> "$out"
            sleep 30
        done
    ) > /dev/null 2>&1 &
    echo $!
}

# ---- A pool that has stopped producing --------------------------------------
# The worker pool can wedge: every worker gone, the parent alive with its
# management threads asleep and no processor time accruing. Nothing in it times
# out, so the block would sit there until someone looked. This samples the
# block's own output and kills a compute that has produced nothing for
# STALL_MIN minutes, which fails the block: the supervisor logs it, leaves it
# unmarked, and moves on. A rerun picks it up where it stopped.
STALL_MIN="${STALL_MIN:-20}"

watch_stall() {
    local pid=$1 dir=$2 tag=$3 last=-1 same=0 need=$((STALL_MIN / 2))
    while kill -0 "$pid" 2>/dev/null; do
        sleep 120
        local n
        n=$(find "$dir" -name 'track_*.npz' 2>/dev/null | wc -l)
        if [ "$n" = "$last" ]; then same=$((same + 1)); else same=0; last=$n; fi
        if [ "$same" -ge "$need" ]; then
            log "$tag: no output for $STALL_MIN min at $n files; the pool is wedged"
            pkill -TERM -P "$pid" 2>/dev/null
            kill -TERM "$pid" 2>/dev/null
            sleep 15
            kill -KILL "$pid" 2>/dev/null
            return
        fi
    done
}

# ---- One compute block ----------------------------------------------------
# Prints the throughput in events per minute so the ramp can compare rungs.
run_block() {
    local tag=$1 evt=$2 stage=$3 member=$4 csv=$5 out=$6 workers=$7
    if done_p "$tag"; then log "skip $tag (done)"; return 0; fi
    local m03 n_before n_after start elapsed made rate sampler
    m03=$(printf 'm%03d' "$member")
    n_before=$(find "$out/$stage/dh=+0" -name "track_${m03}_*.npz" 2>/dev/null | wc -l)
    log "$tag: $evt/$stage $m03, $workers workers, $(wc -l < "$csv") catalogue rows"
    sampler=$(sample_health "$tag" 86400)
    start=$SECONDS
    $PIPELINE compute \
        --event-type "$evt" \
        --source cesm \
        --member "$member" \
        --events-csv "$csv" \
        --era5-dir "$ARCHIVE" \
        --clim-path "$CLIM" \
        --out-dir "$out" \
        --stages "$stage" \
        --dh-range='0:1' \
        --qg-method log20 \
        --ppvi-pieces "$PIECES" \
        --n-workers "$workers" \
        --max-tasks-per-child "$RECYCLE" \
        --skip-existing \
        >> "$LOG_DIR/$tag.log" 2>&1 &
    local pid=$!
    watch_stall "$pid" "$out/$stage/dh=+0" "$tag" &
    local watcher=$!
    wait "$pid"
    local rc=$?
    kill "$watcher" 2>/dev/null
    elapsed=$((SECONDS - start))
    kill "$sampler" 2>/dev/null
    n_after=$(find "$out/$stage/dh=+0" -name "track_${m03}_*.npz" 2>/dev/null | wc -l)
    made=$((n_after - n_before))
    rate=$(awk -v m="$made" -v s="$elapsed" 'BEGIN{printf "%.2f", (s > 0 ? m * 60 / s : 0)}')
    log "$tag: rc=$rc, $made NPZ in $((elapsed / 60)) min = $rate events/min ($workers workers)"
    echo "$rate" > "$LOG_DIR/rate_$tag"
    (( rc == 0 )) && mark "$tag"
    return $rc
}

# ---- Optional ramp on the first member's blocking peak catalogue -----------
# Each rung takes the next slice of that catalogue, so the ramp finishes part
# of the stage rather than sampling it; the block that follows picks up the
# rest under --skip-existing.
choose_workers() {
    local first_member header line=1 best=0 best_rate=0 cat_csv
    first_member=$(echo "$MEMBERS" | cut -d' ' -f1)
    cat_csv=$(csv_for blocking "$first_member")
    header=$(head -1 "$cat_csv")
    local i=0
    for workers in $RAMP; do
        i=$((i + 1))
        local size
        size=$(echo "$RAMP_SIZES" | cut -d' ' -f$i)
        [ -z "$size" ] && size=240
        local slice="$LOG_DIR/ramp_${workers}.csv"
        { echo "$header"; awk -v s="$line" -v n="$size" 'NR>1 && NR>s && NR<=s+n' "$cat_csv"; } > "$slice"
        line=$((line + size))
        if [ "$(wc -l < "$slice")" -le 1 ]; then log "ramp: catalogue exhausted at $workers"; break; fi
        run_block "ramp_${workers}" blocking peak "$first_member" "$slice" "$BLK_OUT" "$workers"
        local rate per
        rate=$(cat "$LOG_DIR/rate_ramp_${workers}" 2>/dev/null || echo 0)
        per=$(awk -v r="$rate" -v w="$workers" 'BEGIN{printf "%.4f", (w > 0 ? r / w : 0)}')
        log "ramp: $workers workers -> $rate events/min ($per per worker)"
        # Stop where the total throughput no longer rises by a tenth: past that
        # the extra processes are contending for cache, not doing work.
        if awk -v r="$rate" -v b="$best_rate" 'BEGIN{exit !(r > b * 1.10)}'; then
            best_rate=$rate; best=$workers
        else
            log "ramp: $workers is not better than $best; keeping $best"
            break
        fi
    done
    [ "$best" = 0 ] && best=$(echo "$RAMP" | cut -d' ' -f1)
    echo "$best"
}

log "=== CESM 6-hourly dh=0, pieces=$PIECES, members $MEMBERS, out=$BLK_OUT and $PRP_OUT"
if [ -n "$RAMP" ]; then
    WORKERS=$(choose_workers)
    log "=== ramp chose $WORKERS workers"
else
    log "=== $WORKERS workers (measured knee; set RAMP to re-measure)"
fi

# ---- The blocks -----------------------------------------------------------
# Peak first for both event types, so the analyses that need it can start while
# onset and decay are still running.
#
# The first block is also the check on the worker count. The knee it defaults to
# was measured on the other archive, whose data grid carries about half the
# points through everything that is not the solver, so this one could saturate
# earlier. If the first block comes in below FIRST_BLOCK_MIN_RATE events per
# minute the run stops rather than spending a day at a count that is not the
# knee; rerun with RAMP set to measure it, and the finished block is skipped.
FIRST_BLOCK_MIN_RATE="${FIRST_BLOCK_MIN_RATE:-25}"
first_block=1

for block in blocking:peak prp:peak blocking:onset prp:onset blocking:decay prp:decay; do
    evt="${block%%:*}"; stage="${block##*:}"
    out="$BLK_OUT"; [ "$evt" = prp ] && out="$PRP_OUT"
    for m in $MEMBERS; do
        m03=$(printf 'm%03d' "$m")
        tag="${evt}_${stage}_${m03}"
        run_block "$tag" "$evt" "$stage" "$m" "$(csv_for "$evt" "$m")" "$out" "$WORKERS"
        if [ "$first_block" = 1 ] && [ -z "$RAMP" ]; then
            first_block=0
            rate=$(cat "$LOG_DIR/rate_$tag" 2>/dev/null || echo 0)
            if awk -v r="$rate" -v m="$FIRST_BLOCK_MIN_RATE" 'BEGIN{exit !(r < m)}'; then
                log "=== $tag ran at $rate events/min, below $FIRST_BLOCK_MIN_RATE:"
                log "===   $WORKERS is not the knee for this archive. Stopping."
                log "===   Rerun with RAMP=\"48 96 144\" to measure it; this block is done."
                exit 3
            fi
            log "=== first block $rate events/min: $WORKERS workers confirmed"
        fi
    done
done

log "=== done"
for d in "$BLK_OUT" "$PRP_OUT"; do
    for s in onset peak decay; do
        n=$(find "$d/$s/dh=+0" -name 'track_*.npz' 2>/dev/null | wc -l)
        log "  $(basename "$d")/$s: $n NPZ"
    done
done
