#!/usr/bin/env python
"""Convert ω-blowup timestamp CSV → track-ID exclude CSV.

For each event in *events_csv* (columns ``track_id, base_ts``), check
whether any blowup timestamp falls within ±dh hours of *any* of that
track's onset/peak/decay base_ts. If so, the track is added to the
exclude list — composites built from any of its stages would otherwise
be contaminated.

Usage:
    python blowup_to_excludes.py \
        --blowup outputs/blowup_scan/omega_300hPa_10pa.csv \
        --events docs/_static/ERA5_TempestExtremes_z500_anticyclone_blocking.csv \
        --dh 12 \
        --out  outputs/blowup_scan/exclude_tracks_blocking.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blowup", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--dh", type=int, default=12,
                    help="Half-window hours around each event base_ts (default 12).")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    blow = pd.read_csv(args.blowup, parse_dates=["timestamp"])
    evt = pd.read_csv(args.events, parse_dates=["timestamp"])

    if blow.empty:
        print("[blowup_to_excludes] no blowups → empty exclude file")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"track_id": []}).to_csv(args.out, index=False)
        return

    bt = blow["timestamp"].values.astype("datetime64[h]").astype("int64")
    et = evt["timestamp"].values.astype("datetime64[h]").astype("int64")
    bt_sorted = np.sort(bt)

    # For each event base_ts, find whether any blowup is within ±dh hours.
    lo = np.searchsorted(bt_sorted, et - args.dh, side="left")
    hi = np.searchsorted(bt_sorted, et + args.dh, side="right")
    contaminated = hi > lo  # True where at least one blowup in window

    bad_tracks = sorted(set(evt.loc[contaminated, "track_id"].astype(int).tolist()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"track_id": bad_tracks}).to_csv(args.out, index=False)

    n_total = evt["track_id"].nunique()
    print(f"[blowup_to_excludes] {len(bad_tracks)} / {n_total} tracks "
          f"contaminated (±{args.dh}h window) → {args.out}")


if __name__ == "__main__":
    main()
