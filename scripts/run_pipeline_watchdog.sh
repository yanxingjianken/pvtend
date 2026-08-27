#!/usr/bin/env bash
# Pipeline watchdog: launches `pvtend-pipeline compute`, then every
# CHECK_INTERVAL seconds reaps Python child processes that have been
# inactive (no CPU / no I/O) for longer than ORPHAN_TIMEOUT seconds.
# Designed for the long event-NPZ recompute (Phase 6) that has been
# observed to leave orphaned worker processes when a single ERA5
# slice fails inside the multiprocessing pool.
#
# Usage:
#   scripts/run_pipeline_watchdog.sh \
#       --workers 96 \
#       --output-root /net/flood/data2/users/x_yan/pvtend/outputs/era5_blocking \
#       --era5-root  /net/flood/data2/users/x_yan/era \
#       --clim-dir   /net/flood/data2/users/x_yan/era/clim \
#       --csv        /path/to/tempest_blocking.csv \
#       --stage      onset
#
# Defaults match the dolma layout in /memories/session/plan.md (Phase 4).

set -euo pipefail

WORKERS="${WORKERS:-96}"
ORPHAN_TIMEOUT="${ORPHAN_TIMEOUT:-1800}"   # 30 min
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
LOG_DIR="${LOG_DIR:-/net/flood/data2/users/x_yan/tmp/pvtend_watchdog}"

# Parse a few common pvtend-pipeline compute flags, forward the rest.
PIPELINE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)        WORKERS="$2";       shift 2 ;;
        --orphan-timeout) ORPHAN_TIMEOUT="$2"; shift 2 ;;
        --check-interval) CHECK_INTERVAL="$2"; shift 2 ;;
        --log-dir)        LOG_DIR="$2";       shift 2 ;;
        *)                PIPELINE_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="$LOG_DIR/pipeline_${TS}.log"
WATCHDOG_LOG="$LOG_DIR/watchdog_${TS}.log"

echo "[watchdog] starting pvtend-pipeline compute --n-workers $WORKERS"
echo "[watchdog]   pipeline log  : $PIPELINE_LOG"
echo "[watchdog]   watchdog log  : $WATCHDOG_LOG"
echo "[watchdog]   orphan timeout: $ORPHAN_TIMEOUT s"

# Launch pipeline in background; capture group PID for clean shutdown.
pvtend-pipeline compute --n-workers "$WORKERS" "${PIPELINE_ARGS[@]}" \
    > "$PIPELINE_LOG" 2>&1 &
PIPELINE_PID=$!
echo "[watchdog] pipeline PID = $PIPELINE_PID" | tee -a "$WATCHDOG_LOG"

cleanup() {
    echo "[watchdog] cleaning up children of $PIPELINE_PID" | tee -a "$WATCHDOG_LOG"
    pkill -P "$PIPELINE_PID" 2>/dev/null || true
    kill -TERM "$PIPELINE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Watchdog loop: every CHECK_INTERVAL, scan worker python processes that
# share the pipeline's session, find ones whose ETIME exceeds the
# ORPHAN_TIMEOUT but whose %CPU has been 0.0 (a proxy for "stuck on a
# shared lock" which is the failure mode we observed).
while kill -0 "$PIPELINE_PID" 2>/dev/null; do
    sleep "$CHECK_INTERVAL"
    now="$(date +%s)"
    # ps fields: pid, etime (seconds), pcpu, command
    ps -eo pid,etimes,pcpu,sess,comm --no-headers \
      | awk -v sess="$(ps -o sess= -p "$PIPELINE_PID")" \
            -v limit="$ORPHAN_TIMEOUT" \
            '$4 == sess && $2 > limit && $3+0.0 == 0.0 && $5 ~ /python/ {print $1}' \
      | while read -r orphan; do
            [[ -z "$orphan" || "$orphan" == "$PIPELINE_PID" ]] && continue
            echo "[$now] killing orphan PID $orphan (idle > $ORPHAN_TIMEOUT s)" \
                | tee -a "$WATCHDOG_LOG"
            kill -TERM "$orphan" 2>/dev/null || true
        done
done

wait "$PIPELINE_PID"
RC=$?
echo "[watchdog] pipeline exited with code $RC" | tee -a "$WATCHDOG_LOG"
exit "$RC"
