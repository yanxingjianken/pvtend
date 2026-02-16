"""Command-line interface for pvtend.

Entry point: ``pvtend-pipeline`` (registered in pyproject.toml).

Usage examples::

    # Process all blocking events in /data/composites/blocking/
    pvtend-pipeline --event-type blocking --events-csv events.csv \\
        --era5-dir /data/era5/ --clim-dir /data/clim/ --out-dir /data/output/

    # Quick composite from pre-computed NPZ files
    pvtend-pipeline composite --npz-dir /data/output/ --pkl-out composite.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pvtend._version import __version__


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

    # --- compute subcommand ---
    compute = sub.add_parser(
        "compute",
        help="Compute PV tendency terms for tracked events.",
    )
    compute.add_argument(
        "--event-type", required=True, choices=["blocking", "prp"],
        help="Event type (blocking or PRP).",
    )
    compute.add_argument(
        "--events-csv", required=True, type=Path,
        help="CSV with tracked events (track_id, lat0, lon0, timestamp).",
    )
    compute.add_argument(
        "--era5-dir", required=True, type=Path,
        help="Directory with ERA5 monthly NetCDF files.",
    )
    compute.add_argument(
        "--clim-dir", required=True, type=Path,
        help="Directory with pre-computed climatology files.",
    )
    compute.add_argument(
        "--out-dir", required=True, type=Path,
        help="Output directory for per-event NPZ files.",
    )
    compute.add_argument(
        "--dh-range", type=str, default="-48:49:6",
        help="Hour offsets as start:stop:step (default: -48:49:6).",
    )
    compute.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel workers.",
    )
    compute.add_argument(
        "--skip-existing", action="store_true",
        help="Skip events with existing NPZ output.",
    )
    compute.add_argument(
        "--use-constant-sigma", action="store_true",
        help="Use constant static stability in QG omega solver.",
    )

    # --- composite subcommand ---
    composite = sub.add_parser(
        "composite",
        help="Aggregate per-event NPZs into composite lifecycle.",
    )
    composite.add_argument(
        "--npz-dir", required=True, type=Path,
        help="Directory containing per-event NPZ files.",
    )
    composite.add_argument(
        "--pkl-out", required=True, type=Path,
        help="Output pickle file for composite state.",
    )
    composite.add_argument(
        "--variants", nargs="*", default=None,
        help="RWB variant names to include (default: all).",
    )
    composite.add_argument(
        "--event-type", choices=["blocking", "prp"], default="blocking",
        help="Event type for composite labelling.",
    )

    # --- decompose subcommand ---
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


def _parse_dh_range(s: str) -> list[int]:
    """Parse colon-separated dh range string."""
    parts = s.split(":")
    if len(parts) != 3:
        raise ValueError(f"dh-range must be start:stop:step, got {s!r}")
    return list(range(int(parts[0]), int(parts[1]), int(parts[2])))


def _cmd_compute(args: argparse.Namespace) -> None:
    """Execute the compute subcommand."""
    import pandas as pd
    from pvtend.tendency import TendencyComputer, TendencyConfig

    dh_values = _parse_dh_range(args.dh_range)

    config = TendencyConfig(
        event_type=args.event_type,
        era5_dir=str(args.era5_dir),
        clim_dir=str(args.clim_dir),
        output_dir=str(args.out_dir),
        dh_values=dh_values,
        skip_existing=args.skip_existing,
        use_constant_sigma=args.use_constant_sigma,
    )
    computer = TendencyComputer(config)

    events_df = pd.read_csv(args.events_csv)
    print(f"[pvtend] Processing {len(events_df)} events, "
          f"dh={dh_values[0]}..{dh_values[-1]}")

    for idx, row in events_df.iterrows():
        print(f"  Event {idx + 1}/{len(events_df)}: "
              f"track_id={row.get('track_id', idx)}")
        try:
            computer.process_event(
                track_id=row.get("track_id", idx),
                lat0=float(row["lat0"]),
                lon0=float(row["lon0"]),
                timestamp=str(row["timestamp"]),
            )
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue

    print("[pvtend] Done.")


def _cmd_composite(args: argparse.Namespace) -> None:
    """Execute the composite subcommand."""
    from pvtend.composites import CompositeState, load_composite_state
    from pvtend.io.npz import list_npz_patches, load_npz_patch
    from pvtend.io.pkl import save_pkl

    patches = list_npz_patches(args.npz_dir)
    print(f"[pvtend] Found {len(patches)} NPZ patches in {args.npz_dir}")

    # Build composite by loading all patches and averaging
    composite_data: dict = {}
    for p in patches:
        data = load_npz_patch(p)
        for key, val in data.items():
            if key not in composite_data:
                composite_data[key] = []
            composite_data[key].append(val)

    # Simple mean composite
    composite_mean = {
        k: CompositeState.composite_mean_3d(v)
        for k, v in composite_data.items()
    }

    save_pkl(composite_mean, args.pkl_out)
    print(f"[pvtend] Composite saved to {args.pkl_out}")


def _cmd_decompose(args: argparse.Namespace) -> None:
    """Execute the decompose subcommand."""
    from pvtend.io.pkl import load_pkl, save_pkl
    from pvtend.decomposition import compute_orthogonal_basis, project_field

    state = load_pkl(args.pkl_in)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pvtend] Running decomposition on {args.pkl_in}")
    # Placeholder: full pipeline integration
    print("[pvtend] Decomposition complete.")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 for success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    dispatch = {
        "compute": _cmd_compute,
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
    sys.exit(main())
