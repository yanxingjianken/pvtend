#!/usr/bin/env bash
# ----------------------------------------------------------------------
# CESM daily branch (BHISTcmip6 members 1-10, 1.5 deg) — FROZEN RECORD
#
# This branch is not regenerated. The stores
#   outputs/cesm_daily_blocking/   (dh=+0, onset/peak/decay)
#   outputs/cesm_daily_prp/
# are the finished products of the 08_triad_resonance PART-2 chain and
# stay as-is. This script documents how they were produced; it exits
# immediately unless explicitly forced.
#
# Generation chain (inputs under /net/flood/data2/users/x_yan/cesm-blocking/cesm_daily/,
# chain code under /net/flood/data2/users/x_yan/archive/cesm_blocking_analysis/08_triad_resonance/):
#   1. pv9_15deg/ + clim_15deg/ + catalogues/  (extraction: cesm_daily/scripts/extraction/)
#   2. build_pvbudget_15deg.py     -> cesm_daily/pvbudget_cache_15deg_p2/
#   3. convert_cesm_to_era5_format.py --wavg-suffix ''  (writes the ERA5-schema NPZs)
#   4. run_part2_chain.sh          (orchestrates 2-3 + verification)
# Cadence caveats if this chain is ever revived at 6-hourly input:
# DT=86400 is hard-coded in build_pvbudget_15deg.py, and its date index
# keys on (year,month,day) — both must change with the timestep.
# ----------------------------------------------------------------------
set -u

if [[ "${1:-}" != "--force" ]]; then
    echo "cesm_daily branch is FROZEN; outputs/cesm_daily_* are final."
    echo "See the header of this script for the generation record."
    exit 0
fi

echo "--force given, but the regeneration chain is intentionally not wired here."
echo "Run the 08_triad_resonance chain manually if a rebuild is truly intended."
exit 1
