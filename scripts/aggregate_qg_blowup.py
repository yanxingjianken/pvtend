#!/usr/bin/env python
"""Aggregate solver blowup metrics from per-event NPZs → exclude CSV.

Scans the scalar blowup metrics embedded in every NPZ under
*npz_dir*/{onset,peak,decay}/dh=*/ and emits a CSV of tracks whose
per-track maximum exceeds the per-field threshold in any scanned field.

Two field groups (select with --fields):

  omega    max_abs_w_{adiabatic,diabatic,qg_diabatic,lhr_moist}
           All-level solver-omega maxima (falls back to the legacy
           ``*_300`` 300-hPa keys on NPZs written before v2.18).
           Raw |omega| is *not* checked — only solver output.  Default
           threshold 25 Pa/s, calibrated against the empirical raw-ERA5
           envelope at 300 hPa over 1990-2020 hourly (max=22.4,
           99.9th=19.9 Pa/s).

  divwind  max_abs_{u,v}_div_{anom,adiabatic,diabatic,qg_diabatic,lhr_moist}
           All-level divergent-wind maxima (v2.18+ NPZs only; older
           NPZs are counted and reported as lacking these keys).
           Hard cutoffs follow the ERA5 distribution scan that produced
           outputs/blowup_scan/source_blowups/: 50 m/s for the
           adiabatic/diabatic branches and the Helmholtz anomaly,
           30 m/s for qg_diabatic, 20 m/s for lhr_moist.

Track ids are ERA5 ints (``track_123_...``) or CESM strings
(``track_m091_t00002_...``); both are preserved as written.  The output
CSV carries columns ``track_id,field,max_val,threshold,reason`` — the
downstream ``--exclude-file`` readers use only ``track_id``.

Usage:
    python aggregate_qg_blowup.py \
        --npz-dir outputs/era5_blocking \
        --fields omega \
        --out outputs/blowup_scan/exclude_tracks_blocking.csv \
        --report outputs/blowup_scan/blowup_report_blocking.csv
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
import sys
import zipfile
import numpy as np
import pandas as pd

_OMEGA_THRESH = 25.0
_DIV_THRESH = {
    "anom": 50.0, "adiabatic": 50.0, "diabatic": 50.0,
    "qg_diabatic": 30.0, "lhr_moist": 20.0,
}

_OMEGA_FIELDS = {
    f"max_abs_w_{b}": _OMEGA_THRESH
    for b in ("adiabatic", "diabatic", "qg_diabatic", "lhr_moist")
}
_DIV_FIELDS = {
    f"max_abs_{c}_div_{b}": t
    for c in ("u", "v") for b, t in _DIV_THRESH.items()
}
_GROUPS = {
    "omega": _OMEGA_FIELDS,
    "divwind": _DIV_FIELDS,
    "all": {**_OMEGA_FIELDS, **_DIV_FIELDS},
}

# ERA5 ids are bare ints; CESM ids are m<member>_t<track> strings.
_TRK_RE = re.compile(r"track_((?:m\d+_t)?\d+)_")


def _parse_track_id(name: str) -> int | str | None:
    m = _TRK_RE.search(name)
    if not m:
        return None
    tok = m.group(1)
    try:
        return int(tok)
    except ValueError:
        return tok


def _read_one(fp: Path, stage: str, npz_dir: Path,
              fields: dict[str, float]) -> dict | None:
    """Extract scalar blowup metrics from a single NPZ. I/O-bound; np.load
    releases the GIL so a thread pool yields near-linear speedup on a
    parallel filesystem."""
    tid = _parse_track_id(fp.name)
    if tid is None:
        return None
    try:
        with np.load(fp, allow_pickle=False) as z:
            rec: dict = {"track_id": tid, "stage": stage,
                         "file": str(fp.relative_to(npz_dir))}
            missing = 0
            for key in fields:
                if key in z.files:
                    rec[key] = float(z[key])
                elif key + "_300" in z.files:
                    # pre-v2.18 NPZ: only the 300-hPa omega scalars exist
                    rec[key] = float(z[key + "_300"])
                else:
                    rec[key] = np.nan
                    missing += 1
            rec["n_missing_keys"] = missing
            return rec
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as e:
        # Corrupt or in-flight NPZ (e.g. crashed worker, concurrent
        # writer).  Skip rather than aborting the scan; the offending
        # track simply won't appear in the exclude list this pass.
        print(f"  [warn] cannot read {fp}: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None


def _scan(npz_dir: Path, stages: list[str], n_workers: int,
          fields: dict[str, float]) -> pd.DataFrame:
    tasks: list[tuple[Path, str]] = []
    for stage in stages:
        stage_dir = npz_dir / stage
        if not stage_dir.is_dir():
            continue
        for dh_dir in sorted(stage_dir.glob("dh=*")):
            for fp in sorted(dh_dir.glob("track_*.npz")):
                tasks.append((fp, stage))

    if not tasks:
        return pd.DataFrame()

    print(f"[aggregate_qg_blowup] scanning {len(tasks)} NPZs with "
          f"{n_workers} threads...", flush=True)

    rows: list[dict] = []
    # ThreadPool because np.load on a small NPZ is dominated by file-I/O
    # and ZIP-header parsing, both of which release the GIL.  Avoids the
    # multiprocessing-fork orphan risk seen earlier.
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_read_one, fp, stage, npz_dir, fields)
                for fp, stage in tasks]
        done = 0
        report_every = max(1, len(tasks) // 20)
        for fut in as_completed(futs):
            rec = fut.result()
            if rec is not None:
                rows.append(rec)
            done += 1
            if done % report_every == 0:
                print(f"  [{done:>6d}/{len(tasks)}] "
                      f"({100.0*done/len(tasks):5.1f}%)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True, type=Path)
    ap.add_argument("--fields", default="omega",
                    help="Field group to scan: omega | divwind | all, "
                         "or a comma-separated list of explicit "
                         "max_abs_* key names (default: omega).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override the omega threshold [Pa/s] "
                         f"(default {_OMEGA_THRESH}).")
    ap.add_argument("--field-threshold", action="append", default=[],
                    metavar="KEY=VAL",
                    help="Override one field's threshold, e.g. "
                         "max_abs_u_div_anom=40. Repeatable.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output exclude-tracks CSV "
                         "(track_id,field,max_val,threshold,reason).")
    ap.add_argument("--stages", nargs="+",
                    default=["onset", "peak", "decay"])
    ap.add_argument("--workers", type=int,
                    default=min(48, (os.cpu_count() or 8) * 2),
                    help="Thread workers for parallel NPZ scan "
                         "(I/O-bound; default min(48, 2*ncpu)).")
    ap.add_argument("--report", type=Path, default=None,
                    help="Per-track report CSV with every field max + "
                         "per-field p99/p99.9/max summary rows.")
    args = ap.parse_args()

    if args.fields in _GROUPS:
        fields = dict(_GROUPS[args.fields])
    else:
        names = [f.strip() for f in args.fields.split(",") if f.strip()]
        unknown = [f for f in names if f not in _GROUPS["all"]]
        if unknown:
            ap.error(f"unknown field(s): {unknown}; "
                     f"choose from {sorted(_GROUPS['all'])}")
        fields = {f: _GROUPS["all"][f] for f in names}
    if args.threshold is not None:
        for k in fields:
            if k in _OMEGA_FIELDS:
                fields[k] = args.threshold
    for ov in args.field_threshold:
        k, _, v = ov.partition("=")
        if k not in fields:
            ap.error(f"--field-threshold key {k!r} not in scanned fields")
        fields[k] = float(v)

    df = _scan(args.npz_dir, args.stages, args.workers, fields)
    if df.empty:
        print(f"[aggregate_qg_blowup] no NPZ blowup metrics found under "
              f"{args.npz_dir}/{{onset,peak,decay}}", file=sys.stderr)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["track_id", "field", "max_val",
                              "threshold", "reason"]).to_csv(
            args.out, index=False)
        return

    n_lacking = int((df["n_missing_keys"] > 0).sum())
    if n_lacking:
        print(f"[aggregate_qg_blowup] {n_lacking}/{len(df)} NPZs lack some "
              f"scanned keys (pre-v2.18 files; missing fields not flagged)")

    cols = list(fields)
    per_track = df.groupby("track_id")[cols].max()

    # a track is excluded if ANY field exceeds its threshold; the CSV row
    # names the worst offender by threshold ratio
    ratio = per_track / pd.Series(fields)
    worst_field = ratio.idxmax(axis=1)
    worst_ratio = ratio.max(axis=1)
    bad_ids = worst_ratio.index[worst_ratio > 1.0]

    out_rows = []
    for tid in bad_ids:
        f = worst_field.loc[tid]
        out_rows.append({
            "track_id": tid,
            "field": f,
            "max_val": per_track.loc[tid, f],
            "threshold": fields[f],
            "reason": "omega" if f in _OMEGA_FIELDS else "divwind",
        })
    bad = pd.DataFrame(out_rows,
                       columns=["track_id", "field", "max_val",
                                "threshold", "reason"])
    if not bad.empty:
        bad = bad.sort_values("max_val", ascending=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bad.to_csv(args.out, index=False)

    print(f"[aggregate_qg_blowup] {len(bad)} / {len(per_track)} tracks "
          f"excluded → {args.out}")
    for _, r in bad.head(10).iterrows():
        print(f"    track {r['track_id']!s:>14}  {r['field']}="
              f"{r['max_val']:.3g} (thr {r['threshold']:g})")

    # per-field calibration summary — the basis for recalibrating the
    # divergent-wind cutoffs after the first full scan of a new source
    print(f"{'field':<28} {'p99':>10} {'p99.9':>10} {'max':>10} {'thr':>7}")
    summary = {}
    for f in cols:
        v = per_track[f].dropna()
        if v.empty:
            continue
        summary[f] = (v.quantile(0.99), v.quantile(0.999), v.max())
        print(f"{f:<28} {summary[f][0]:>10.3g} {summary[f][1]:>10.3g} "
              f"{summary[f][2]:>10.3g} {fields[f]:>7g}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        rep = per_track.copy()
        rep["worst_field"] = worst_field
        rep["worst_ratio"] = worst_ratio
        rep.sort_values("worst_ratio", ascending=False) \
           .to_csv(args.report, index=True)
        print(f"[aggregate_qg_blowup] full per-track report → {args.report}")


if __name__ == "__main__":
    main()
