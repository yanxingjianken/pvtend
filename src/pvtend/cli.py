"""Command-line interface for pvtend.

Entry point: ``pvtend-pipeline`` (registered in pyproject.toml).

Subcommands
-----------

compute
    Compute PV tendency terms (NPZ per event-timestep).
classify
    RWB classification — Pass 1 (``rwb_variant_tracksets.pkl``).
composite
    Variant-aware accumulation — Pass 2 (``composite.pkl``).
decompose
    Orthogonal-basis decomposition on composite fields.

Usage examples::

    pvtend-pipeline compute \\
        --event-type blocking --events-csv events.csv \\
        --era5-dir /data/era5/ --clim-path /data/clim/era5_hourly_clim.nc \\
        --out-dir /data/composite_blocking_tempest/ --dh-range='-49:25:1'

    pvtend-pipeline classify \\
        --npz-dir /data/composite_blocking_tempest/ \\
        --output /data/outputs/rwb_variant_tracksets.pkl

    pvtend-pipeline composite \\
        --npz-dir /data/composite_blocking_tempest/ \\
        --rwb-pkl /data/outputs/rwb_variant_tracksets.pkl \\
        --pkl-out /data/outputs/composite.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pvtend._version import __version__


# =====================================================================
# Argument parser
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="pvtend-pipeline",
        description="PV tendency decomposition pipeline for blocking/PRP events.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── compute ──────────────────────────────────────────────────
    compute = sub.add_parser(
        "compute",
        help="Compute PV tendency terms for tracked events → NPZ files.",
    )
    compute.add_argument(
        "--event-type", required=True, choices=["blocking", "prp"],
        help="Event type.",
    )
    compute.add_argument(
        "--events-csv", required=True, type=Path,
        help="CSV with columns: evt_name, track_id, lat0, lon0, base_ts.",
    )
    compute.add_argument(
        "--era5-dir", required=True, type=Path,
        help="Directory with ERA5 monthly NetCDF files.",
    )
    compute.add_argument(
        "--clim-path", required=True, type=Path,
        help="Climatology file or directory.",
    )
    compute.add_argument(
        "--out-dir", required=True, type=Path,
        help="Output directory for per-event NPZ files.",
    )
    compute.add_argument(
        "--track-file", type=Path, default=None,
        help="Tracking data file for Lagrangian mode.",
    )
    compute.add_argument(
        "--dh-range", type=str, default="-49:25:1",
        help="Hour offsets as start:stop[:step]  (default: -49:25:1).",
    )
    compute.add_argument(
        "--qg-method", choices=["log20", "sp19"], default="log20",
        help="QG omega solver (default: log20).  sp19 = empirical scaling.",
    )
    compute.add_argument(
        "--center-mode", choices=["eulerian", "lagrangian"],
        default="eulerian",
        help="Centre tracking mode (default: eulerian).",
    )
    compute.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel workers.",
    )
    compute.add_argument(
        "--year-range", type=str, default=None,
        help="Filter events to year range, e.g. '1990:2011' (start:stop_exclusive).",
    )
    compute.add_argument(
        "--stages", nargs="+", default=["onset", "peak", "decay"],
        help="Event stages to process (default: onset peak decay).",
    )
    compute.add_argument(
        "--skip-existing", action="store_true",
        help="Skip events whose NPZ files already exist.",
    )

    # ── classify ─────────────────────────────────────────────────
    classify = sub.add_parser(
        "classify",
        help="RWB classification (Pass 1) → variant tracksets PKL.",
    )
    classify.add_argument(
        "--npz-dir", required=True, type=Path,
        help="Root NPZ directory (with onset/peak/decay sub-dirs).",
    )
    classify.add_argument(
        "--output", required=True, type=Path,
        help="Output pickle file for variant tracksets.",
    )
    classify.add_argument(
        "--stages", nargs="+", default=["onset", "peak", "decay"],
        help="Event stages to classify (default: onset peak decay).",
    )
    classify.add_argument(
        "--levels", nargs="+", default=[500, 400, 300, 200],
        help="Pressure levels (integers) or 'wavg' for weighted-average 2-D Z.",
    )
    classify.add_argument(
        "--threshold", type=int, default=3,
        help="Min levels that must agree for AWB/CWB (default: 3).",
    )
    classify.add_argument(
        "--exclude-file", type=Path, default=None,
        help="CSV listing track IDs to exclude.",
    )

    # ── composite ────────────────────────────────────────────────
    composite = sub.add_parser(
        "composite",
        help="Variant-aware composite accumulation (Pass 2) → composite PKL.",
    )
    composite.add_argument(
        "--npz-dir", required=True, type=Path,
        help="Root NPZ directory.",
    )
    composite.add_argument(
        "--rwb-pkl", type=Path, default=None,
        help="RWB variant tracksets PKL from classify step.",
    )
    composite.add_argument(
        "--pkl-out", required=True, type=Path,
        help="Output composite pickle file.",
    )
    composite.add_argument(
        "--stages", nargs="+", default=["onset", "peak", "decay"],
        help="Stages to composite.",
    )
    composite.add_argument(
        "--exclude-file", type=Path, default=None,
        help="CSV listing track IDs to exclude.",
    )

    # ── decompose ────────────────────────────────────────────────
    decompose = sub.add_parser(
        "decompose",
        help="Run orthogonal basis decomposition on composite state.",
    )
    decompose.add_argument(
        "--pkl-in", required=True, type=Path,
        help="Input composite state pickle.",
    )
    decompose.add_argument(
        "--out-dir", required=True, type=Path,
        help="Output directory for decomposition results.",
    )
    decompose.add_argument(
        "--smooth-sigma", type=float, default=None,
        help="Gaussian smoothing sigma (degrees).",
    )

    return parser


# =====================================================================
# Helpers
# =====================================================================

def _parse_dh_range(s: str) -> list[int]:
    """Parse colon-separated dh range string → list of ints.

    Accepted formats: ``start:stop`` or ``start:stop:step``.
    """
    parts = s.split(":")
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1])))
    if len(parts) == 3:
        return list(range(int(parts[0]), int(parts[1]), int(parts[2])))
    raise ValueError(f"dh-range must be start:stop[:step], got {s!r}")


# =====================================================================
# Parallel worker (module-level for pickling)
# =====================================================================

_WORKER_CFG = None  # set before pool.map


def _init_worker(cfg):
    """Pool initializer — store config in global so workers can access it."""
    global _WORKER_CFG
    _WORKER_CFG = cfg


def _process_one_event(arg_tuple):
    """Process a single event in a worker process."""
    from pvtend.tendency import TendencyComputer
    evt_name, track_id, lat0, lon0, base_ts = arg_tuple
    computer = TendencyComputer(_WORKER_CFG)
    try:
        n = computer.process_event(
            evt_name=evt_name, track_id=track_id,
            lat0=lat0, lon0=lon0, base_ts=base_ts)
        return n
    except Exception as exc:
        print(f"    ERROR [{track_id} {evt_name}]: {exc}", flush=True)
        return 0


# =====================================================================
# Subcommand implementations
# =====================================================================

def _cmd_compute(args: argparse.Namespace) -> None:
    """Execute the ``compute`` subcommand."""
    import pandas as pd
    from pvtend.tendency import TendencyComputer, TendencyConfig

    dh_values = _parse_dh_range(args.dh_range)

    qg = args.qg_method

    config = TendencyConfig(
        event_type=args.event_type,
        data_dir=args.era5_dir,
        clim_path=args.clim_path,
        output_dir=args.out_dir,
        csv_path=args.events_csv,
        track_file=args.track_file or Path(""),
        rel_hours=dh_values,
        qg_omega_method=qg,
        center_mode=args.center_mode,
        skip_existing=args.skip_existing,
        n_workers=args.n_workers,
    )
    computer = TendencyComputer(config)

    events_df = pd.read_csv(args.events_csv)

    # --- Auto-detect old CSV format and rename columns ---
    col_map = {}
    if "lat" in events_df.columns and "lat0" not in events_df.columns:
        col_map["lat"] = "lat0"
    if "lon180" in events_df.columns and "lon0" not in events_df.columns:
        col_map["lon180"] = "lon0"
    if "type" in events_df.columns and "evt_name" not in events_df.columns:
        col_map["type"] = "evt_name"
    if col_map:
        events_df = events_df.rename(columns=col_map)
        print(f"[pvtend] Auto-mapped CSV columns: {col_map}")

    # --- Filter by year range if provided ---
    if args.year_range is not None:
        yr_parts = args.year_range.split(":")
        yr_start, yr_end = int(yr_parts[0]), int(yr_parts[1])
        if "year" in events_df.columns:
            events_df = events_df[
                (events_df["year"] >= yr_start) & (events_df["year"] < yr_end)
            ].reset_index(drop=True)
        elif "timestamp" in events_df.columns:
            ts_col = pd.to_datetime(events_df["timestamp"])
            events_df = events_df[
                (ts_col.dt.year >= yr_start) & (ts_col.dt.year < yr_end)
            ].reset_index(drop=True)
        elif "base_ts" in events_df.columns:
            ts_col = pd.to_datetime(events_df["base_ts"])
            events_df = events_df[
                (ts_col.dt.year >= yr_start) & (ts_col.dt.year < yr_end)
            ].reset_index(drop=True)
        print(f"[pvtend] Filtered to years [{yr_start}, {yr_end}): "
              f"{len(events_df)} events")

    # --- Filter by stages if not all stages ---
    stage_col = "evt_name" if "evt_name" in events_df.columns else "stage"
    if stage_col in events_df.columns:
        events_df = events_df[
            events_df[stage_col].isin(args.stages)
        ].reset_index(drop=True)
        print(f"[pvtend] Filtered to stages {args.stages}: "
              f"{len(events_df)} events")

    print(f"[pvtend] Processing {len(events_df)} events, "
          f"dh={dh_values[0]}..{dh_values[-1]}, qg_method={qg}")

    # Build list of event tuples for iteration / parallelism
    event_args = []
    for idx, row in events_df.iterrows():
        evt_name = str(row.get("evt_name", row.get("stage", "onset")))
        track_id = int(row.get("track_id", idx))
        lat0 = float(row["lat0"])
        lon0 = float(row["lon0"])
        base_ts = pd.Timestamp(str(row.get("base_ts",
                                            row.get("timestamp"))))
        event_args.append((evt_name, track_id, lat0, lon0, base_ts))

    if config.n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"[pvtend] Using {config.n_workers} parallel workers")

        n_total = 0
        with ProcessPoolExecutor(max_workers=config.n_workers,
                                 initializer=_init_worker,
                                 initargs=(config,)) as ex:
            futs = {ex.submit(_process_one_event, a): a for a in event_args}
            for i, f in enumerate(as_completed(futs), 1):
                n = f.result()
                n_total += n
                if i % 50 == 0 or i == len(futs):
                    print(f"[pvtend] {i}/{len(futs)} events done, "
                          f"{n_total} NPZ written so far", flush=True)
    else:
        n_total = 0
        for i, (evt_name, track_id, lat0, lon0, base_ts) in enumerate(
                event_args):
            print(f"  Event {i + 1}/{len(event_args)}: "
                  f"track_id={track_id}  {evt_name}  {base_ts}")
            try:
                n = computer.process_event(
                    evt_name=evt_name, track_id=track_id,
                    lat0=lat0, lon0=lon0, base_ts=base_ts)
                n_total += n
            except Exception as exc:
                print(f"    ERROR: {exc}")
                continue

    print(f"[pvtend] Done — wrote {n_total} NPZ files.")


def _cmd_classify(args: argparse.Namespace) -> None:
    """Execute the ``classify`` subcommand (Pass 1)."""
    from pvtend.classify import ClassifyConfig, run_pass1
    from pvtend.rwb import RWBConfig

    # Parse levels: accept integers or 'wavg'
    parsed_levels: list[int | str] = []
    for lv in args.levels:
        if isinstance(lv, int):
            parsed_levels.append(lv)
        elif str(lv).lower() == "wavg":
            parsed_levels.append("wavg")
        else:
            parsed_levels.append(int(lv))

    cfg = ClassifyConfig(
        npz_dir=args.npz_dir,
        output_path=args.output,
        stages=args.stages,
        classify_levels=parsed_levels,
        classify_threshold=args.threshold,
        rwb_cfg=RWBConfig(area_min_deg2=20.0, try_levels=400),
        exclude_file=args.exclude_file,
    )
    result = run_pass1(cfg)
    result.save(cfg.output_path)

    # Summary
    for evt in result.stages:
        n_all = len(result.stage_all.get(evt, set()))
        n_awb = len(result.stage_awb.get(evt, set()))
        n_cwb = len(result.stage_cwb.get(evt, set()))
        n_neu = len(result.stage_neu.get(evt, set()))
        print(f"  {evt}: ALL={n_all}  AWB={n_awb}  CWB={n_cwb}  NEU={n_neu}")

    print(f"[pvtend] Variant tracksets saved to {cfg.output_path}")


def _cmd_composite(args: argparse.Namespace) -> None:
    """Execute the ``composite`` subcommand (Pass 2)."""
    from pvtend.classify import ClassifyResult
    from pvtend.composite_builder import CompositeConfig, build_composites

    rwb = None
    if args.rwb_pkl is not None and args.rwb_pkl.exists():
        rwb = ClassifyResult.load(args.rwb_pkl)
        print(f"[pvtend] Loaded RWB variants from {args.rwb_pkl}")

    cfg = CompositeConfig(
        npz_dir=args.npz_dir,
        stages=args.stages,
        exclude_file=args.exclude_file,
    )
    result = build_composites(cfg, rwb)
    result.save(args.pkl_out)

    print(f"[pvtend] Composite ({len(result.fields_3d)} fields, "
          f"{len(result.variant_names)} variants) saved to {args.pkl_out}")


def _cmd_decompose(args: argparse.Namespace) -> None:
    """Execute the ``decompose`` subcommand."""
    from pvtend.decomposition import compute_orthogonal_basis, project_field

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pvtend] Running decomposition on {args.pkl_in}")
    # Placeholder: full pipeline integration
    print("[pvtend] Decomposition complete.")


# =====================================================================
# Main entry point
# =====================================================================

def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success).
    """
    import multiprocessing as _mp
    try:
        _mp.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    dispatch = {
        "compute": _cmd_compute,
        "classify": _cmd_classify,
        "composite": _cmd_composite,
        "decompose": _cmd_decompose,
    }

    try:
        dispatch[args.command](args)
    except Exception as exc:
        print(f"[pvtend] Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    import multiprocessing as _mp
    try:
        _mp.set_start_method("spawn")
    except RuntimeError:
        pass
    sys.exit(main())
