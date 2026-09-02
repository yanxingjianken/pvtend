#!/usr/bin/env bash
# ERA5 tendency + piecewise PV inversion at the event time only (dh = 0).
#
# The inversion is the global spherical solver, whose cost is dominated by
# spectral transforms streaming a Legendre table through the CPU cache, so the
# worker count is not a free parameter: eight of these processes sharing one
# core complex run five times slower each than one alone, while a hundred and
# twenty-eight processes of the older relaxation solver ran at ninety-six per
# cent efficiency. The run therefore ramps, and the ramp is production: each
# rung processes its own slice of the blocking peak catalogue with
# --skip-existing, so nothing is computed twice and nothing is thrown away.
# The rung whose throughput per worker stops improving sets the count for the
# remaining blocks.
#
# Usage:
#   scripts/run_era5_dh0.sh                 # ramp, then the rest at the chosen count
#   WORKERS=96 scripts/run_era5_dh0.sh      # skip the ramp, run everything at 96
#   RAMP="48 96 144" scripts/run_era5_dh0.sh
#
# Resume: rerun it. Every block is skipped once its marker is in the state file,
# and every event is skipped once its NPZ exists.
set -uo pipefail

ROOT="${ROOT:-/net/flood/data2/users/x_yan/pvtend}"
ERA5_DIR="${ERA5_DIR:-/net/flood/data2/users/x_yan/era}"
CLIM_DIR="${CLIM_DIR:-$ERA5_DIR/clim}"
CLIM_STEM="${CLIM_STEM:-$CLIM_DIR/era5_hourly_clim_1990-2020.nc}"
BLK_CSV="${BLK_CSV:-$ROOT/docs/_static/ERA5_TempestExtremes_z500_anticyclone_blocking.csv}"
PRP_CSV="${PRP_CSV:-$ROOT/docs/_static/ERA5_TempestExtremes_z500_exclusif_anticyclone_propagating.csv}"
BLK_OUT="${BLK_OUT:-$ROOT/outputs/era5_blocking}"
PRP_OUT="${PRP_OUT:-$ROOT/outputs/era5_prp}"
TMP="${TMP:-/net/flood/data2/users/x_yan/tmp}"
LOG_DIR="${LOG_DIR:-$TMP/pvtend_era5_dh0}"
STATE="${STATE:-$LOG_DIR/state}"
RAMP="${RAMP:-48 96 144 192}"
RAMP_SIZES="${RAMP_SIZES:-360 720 1080 1620}"  # catalogue ROWS per rung; a third of them
                                              # are the stage being computed
WORKERS="${WORKERS:-}"                        # set to skip the ramp
PIECES="${PIECES:-scale}"

ENV_RUN="micromamba run -n blocking"
PIPELINE="$ENV_RUN pvtend-pipeline"

mkdir -p "$LOG_DIR" "$BLK_OUT" "$PRP_OUT"
touch "$STATE"

log()  { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/supervisor.log"; }
done_p() { grep -qx "$1" "$STATE"; }
mark()   { echo "$1" >> "$STATE"; }

# ---- Health: workers blocked on storage rather than working ---------------
# A run that saturates the filesystem shows as processes in uninterruptible
# sleep. Compute-bound workers stay runnable, so a rung with a D-state count
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
            echo "$(date '+%F %T') D=$d R=$r load=$(cut -d' ' -f1 /proc/loadavg)" >> "$out"
            sleep 30
        done
    ) > /dev/null 2>&1 &
    echo $!
}

# ---- One compute block ----------------------------------------------------
# Prints the throughput in events per minute so the ramp can compare rungs.
run_block() {
    local tag=$1 evt=$2 stage=$3 csv=$4 out=$5 workers=$6
    if done_p "$tag"; then log "skip $tag (done)"; return 0; fi
    local n_before n_after start elapsed made rate sampler
    n_before=$(find "$out/$stage/dh=+0" -name 'track_*.npz' 2>/dev/null | wc -l)
    log "$tag: $evt/$stage, $workers workers, $(wc -l < "$csv") catalogue rows"
    sampler=$(sample_health "$tag" 86400)
    start=$SECONDS
    $PIPELINE compute \
        --event-type "$evt" \
        --events-csv "$csv" \
        --era5-dir "$ERA5_DIR" \
        --clim-path "$CLIM_STEM" \
        --clim-helmholtz-dir "$CLIM_DIR" \
        --out-dir "$out" \
        --stages "$stage" \
        --dh-range='0:1' \
        --qg-method log20 \
        --ppvi-pieces "$PIECES" \
        --n-workers "$workers" \
        --skip-existing \
        >> "$LOG_DIR/$tag.log" 2>&1
    local rc=$?
    elapsed=$((SECONDS - start))
    kill "$sampler" 2>/dev/null
    n_after=$(find "$out/$stage/dh=+0" -name 'track_*.npz' 2>/dev/null | wc -l)
    made=$((n_after - n_before))
    rate=$(awk -v m="$made" -v s="$elapsed" 'BEGIN{printf "%.2f", s>0 ? m*60/s : 0}')
    log "$tag: rc=$rc, $made NPZ in $((elapsed / 60)) min = $rate events/min ($workers workers)"
    echo "$rate" > "$LOG_DIR/rate_$tag"
    (( rc == 0 )) && mark "$tag"
    return $rc
}

# ---- Ramp on the blocking peak catalogue ----------------------------------
# Each rung takes the next slice of the same catalogue, so the ramp finishes
# the stage rather than sampling it.
choose_workers() {
    local header line=1 best=0 best_rate=0 prev_per_worker=0
    header=$(head -1 "$BLK_CSV")
    local i=0
    for workers in $RAMP; do
        i=$((i + 1))
        local size
        size=$(echo "$RAMP_SIZES" | cut -d' ' -f$i)
        [ -z "$size" ] && size=240
        local slice="$LOG_DIR/ramp_${workers}.csv"
        { echo "$header"; awk -v s="$line" -v n="$size" 'NR>1 && NR>s && NR<=s+n' "$BLK_CSV"; } > "$slice"
        line=$((line + size))
        if [ "$(wc -l < "$slice")" -le 1 ]; then log "ramp: catalogue exhausted at $workers"; break; fi
        run_block "ramp_${workers}" blocking peak "$slice" "$BLK_OUT" "$workers"
        local rate per
        rate=$(cat "$LOG_DIR/rate_ramp_${workers}" 2>/dev/null || echo 0)
        per=$(awk -v r="$rate" -v w="$workers" 'BEGIN{printf "%.4f", w>0 ? r/w : 0}')
        log "ramp: $workers workers -> $rate events/min ($per per worker)"
        # Stop where the total throughput no longer rises by a tenth: past that
        # the extra processes are contending for cache, not doing work.
        if awk -v r="$rate" -v b="$best_rate" 'BEGIN{exit !(r > b * 1.10)}'; then
            best_rate=$rate; best=$workers
        else
            log "ramp: $workers is not better than $best; keeping $best"
            break
        fi
        prev_per_worker=$per
    done
    [ "$best" = 0 ] && best=$(echo "$RAMP" | cut -d' ' -f1)
    echo "$best"
}

log "=== ERA5 dh=0, pieces=$PIECES, out=$BLK_OUT and $PRP_OUT"
if [ -z "$WORKERS" ]; then
    WORKERS=$(choose_workers)
    log "=== ramp chose $WORKERS workers"
else
    log "=== worker count given: $WORKERS (no ramp)"
fi

# ---- The remaining blocks -------------------------------------------------
# Peak first for both event types, so the analyses that need it can start while
# onset and decay are still running.
for block in blocking:peak prp:peak blocking:onset prp:onset blocking:decay prp:decay; do
    evt="${block%%:*}"; stage="${block##*:}"
    csv="$BLK_CSV"; out="$BLK_OUT"
    if [ "$evt" = prp ]; then csv="$PRP_CSV"; out="$PRP_OUT"; fi
    run_block "${evt}_${stage}" "$evt" "$stage" "$csv" "$out" "$WORKERS"
done

log "=== done"
for d in "$BLK_OUT" "$PRP_OUT"; do
    for s in onset peak decay; do
        n=$(find "$d/$s/dh=+0" -name 'track_*.npz' 2>/dev/null | wc -l)
        log "  $(basename "$d")/$s: $n NPZ"
    done
done
