"""Batch driver.

The inversion is parallel over events, not inside one: a single event is a few
seconds of dense linear algebra that BLAS already threads, and a catalogue is
tens of thousands of them.  Each worker is pinned to one thread so the pool does
not oversubscribe the node -- with BLAS left to its own devices, sixty-four
workers each spawning sixty-four threads is slower than running serially.

Failures are recorded per event rather than raised: one event with a pathological
mean state should not end a run of thirty thousand.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np


@dataclass
class EventSpec:
    """One row of the catalogue."""

    event_id: str
    state_path: str
    state_index: int
    clim_path: str
    clim_index: int
    lat: float
    lon: float


def read_catalogue(path: str) -> list[EventSpec]:
    """Read a catalogue CSV, checking the columns before the run starts."""
    required = {
        "event_id",
        "state_path",
        "state_index",
        "clim_path",
        "clim_index",
        "lat",
        "lon",
    }
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing columns {sorted(missing)}; found "
                f"{reader.fieldnames}"
            )
        return [
            EventSpec(
                event_id=row["event_id"],
                state_path=row["state_path"],
                state_index=int(row["state_index"]),
                clim_path=row["clim_path"],
                clim_index=int(row["clim_index"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
            )
            for row in reader
        ]


def run_one(spec: EventSpec, options: dict) -> dict:
    """Invert one event and write its file; never raises.

    The linear algebra is held to one thread for the duration.  It has to be done
    here, at run time: the library sizes its thread pool when it is first loaded,
    which happens while the package is being imported, so an environment variable
    set anywhere in this module is already too late.  Measured on one event, with
    the pool left at its default of 128 threads: 3.2 minutes of wall clock and
    226 minutes of processor time, against 3.2 and 3.2 with one thread.  The
    arrays here are small enough that threading them buys nothing and the spin
    waits cost everything -- and a pool of workers each doing that would bring the
    node down.
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _run_one_inner(spec, options)


def _run_one_inner(spec: EventSpec, options: dict) -> dict:
    from .config import InversionConfig, MirrorConfig, NewtonConfig
    from .io import load_cesm_state, save_npz
    from .pipeline import invert_event

    out_path = os.path.join(options["out_dir"], f"{spec.event_id}.npz")
    if options.get("skip_existing") and os.path.exists(out_path):
        return {"event_id": spec.event_id, "status": "skipped", "seconds": 0.0}

    started = time.time()
    try:
        state = load_cesm_state(spec.state_path, spec.state_index)
        clim = load_cesm_state(spec.clim_path, spec.clim_index, state.p_hpa)
        if not (
            np.array_equal(state.lat, clim.lat) and np.array_equal(state.lon, clim.lon)
        ):
            raise ValueError(
                "the climatology is on a different grid from the state; they are "
                "combined by position, so this would silently mix latitudes"
            )
        cfg = InversionConfig(
            mirror=MirrorConfig(
                blend=options["blend"],
                blend_south=options["blend_south"],
                blend_north=options["blend_north"],
                f_floor_deg=options["f_floor_deg"],
            ),
            newton=NewtonConfig(max_steps=options["newton_max_steps"]),
        )
        output = invert_event(
            clim.as_tuple(),
            state.as_tuple(),
            state.lat,
            state.lon,
            (spec.lat, spec.lon),
            cfg=cfg,
            lat_half=options["lat_half"],
            lon_half=options["lon_half"],
            solver_nlat=options["solver_nlat"],
            solver_nlon=options["solver_nlon"],
            rotated_track=options["rotated_track"],
            pieces_mode=options["pieces_mode"],
        )
        save_npz(out_path, output.arrays, output.meta)
        return {
            "event_id": spec.event_id,
            "status": "ok"
            if output.meta["all_pieces_converged"]
            and output.meta["newton_converged"]
            else "unconverged",
            "seconds": time.time() - started,
            "newton_steps": output.meta["newton_steps"],
        }
    except Exception as error:  # one bad event must not end the batch
        return {
            "event_id": spec.event_id,
            "status": "failed",
            "seconds": time.time() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pvinv-sph",
        description=(
            "Global spherical-harmonic piecewise potential-vorticity inversion "
            "over a catalogue of events."
        ),
    )
    parser.add_argument("catalogue", help="CSV of events; see read_catalogue")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--lat-half", type=float, default=30.0)
    parser.add_argument("--lon-half", type=float, default=60.0)
    parser.add_argument("--solver-nlat", type=int, default=128)
    parser.add_argument("--solver-nlon", type=int, default=256)
    parser.add_argument("--rotated-track", action="store_true")
    parser.add_argument(
        "--pieces",
        choices=["per_level", "scale"],
        default="per_level",
        help=(
            "one piece per source, or the four-way surface / lower / "
            "planetary-upper / eddy-upper split; the two write disjoint key sets "
            "and must not be mixed in one output directory"
        ),
    )
    parser.add_argument(
        "--blend-south",
        type=float,
        default=5.0,
        help="equatorward edge of the coefficient taper, in degrees",
    )
    parser.add_argument(
        "--blend-north",
        type=float,
        default=20.0,
        help=(
            "poleward edge of the coefficient taper; with the default of 20 a "
            "patch centred at 40N reaches into it, so a solution quoted there "
            "needs this narrowed and the run repeated"
        ),
    )
    parser.add_argument(
        "--f-floor-deg",
        type=float,
        default=12.0,
        help=(
            "latitude at which the Coriolis floor is set; the ratio to |f| is "
            "1.53 at 10.5N and 1.16 at 20N for the default, so it too reaches "
            "well into the subtropics"
        ),
    )
    parser.add_argument(
        "--no-blend",
        action="store_true",
        help=(
            "skip the equatorial taper of the coefficients; appropriate only "
            "when the state is already smooth across the equator"
        ),
    )
    parser.add_argument(
        "--newton-max-steps",
        type=int,
        default=20,
        help=(
            "cap on the nonlinear iteration of the total inversion; the increment "
            "test decays slowly on some events, so a run that hits this cap may "
            "already satisfy the equations -- check meta_newton_final_increment_m "
            "before treating it as a failure"
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    specs = read_catalogue(args.catalogue)
    if args.limit:
        specs = specs[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    options = {
        "out_dir": args.out_dir,
        "lat_half": args.lat_half,
        "lon_half": args.lon_half,
        "solver_nlat": args.solver_nlat,
        "solver_nlon": args.solver_nlon,
        "rotated_track": args.rotated_track,
        "pieces_mode": args.pieces,
        "blend": not args.no_blend,
        "blend_south": args.blend_south,
        "blend_north": args.blend_north,
        "f_floor_deg": args.f_floor_deg,
        "newton_max_steps": args.newton_max_steps,
        "skip_existing": args.skip_existing,
    }

    started = time.time()
    tally: dict[str, int] = {}
    print(f"{len(specs)} events, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, spec, options): spec for spec in specs}
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            tally[record["status"]] = tally.get(record["status"], 0) + 1
            if record["status"] == "failed":
                print(
                    f"  {record['event_id']}: {record['error']}",
                    file=sys.stderr,
                    flush=True,
                )
            if done % 50 == 0 or done == len(specs):
                rate = done / max(time.time() - started, 1e-9) * 60.0
                print(
                    f"  {done}/{len(specs)}  {rate:.1f} events/min  {tally}",
                    flush=True,
                )

    elapsed = time.time() - started
    print(f"done in {elapsed / 60:.1f} min: {tally}", flush=True)
    return 1 if tally.get("failed") else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
