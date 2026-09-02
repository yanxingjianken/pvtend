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

def _add_ppvi_pieces_arg(p: argparse.ArgumentParser) -> None:
    """The PPVI flags shared by ``compute`` and ``ppvi``.

    The two subcommands must offer the same choices or a catalogue could be
    started with one decomposition and appended to in another.
    """
    p.add_argument(
        "--ppvi-pieces", choices=("per_level", "scale"), default="scale",
        help="How to split the PV/theta anomaly before inverting.  "
             "'scale' (default) gives four pieces surface / lower / upper_p / "
             "upper_e, splitting the upper levels into a planetary (zonal "
             "k<=4, inside the tracked object) and an eddy part.  'per_level' "
             "gives one piece per level, keys u/v_rot_anom_ppvi_{1000..100}. "
             "The two write different keys — do not mix them within one "
             "output directory.",
    )
    p.add_argument(
        "--ppvi-solver-grid", type=int, nargs=2, default=[128, 256],
        metavar=("NLAT", "NLON"),
        help="Gaussian solver grid of the spherical engine (default 128 256, "
             "truncation T84 after the dealiasing rule).  Not the data grid.",
    )
    p.add_argument(
        "--ppvi-newton-max-steps", type=int, default=60,
        help="Cap on the spherical engine's Newton iteration (default 60; "
             "real events take 4-10).",
    )


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
        "--source", choices=["era5", "cesm"], default="era5",
        help="Input dataset family (default: era5). 'cesm' reads the "
             "CESM2-LENS2 member-year archive and needs --member.",
    )
    compute.add_argument(
        "--member", type=int, default=None,
        help="LENS2 member number (91-100 for the 6-hourly smbb set). "
             "Required with --source cesm: the member selects the file and "
             "cannot be inferred from the event row, which is why the event "
             "CSVs are written one per member.",
    )
    compute.add_argument(
        "--era5-dir", required=True, type=Path,
        help="Directory with the input NetCDF files (ERA5 monthly, or the "
             "CESM member-year archive when --source cesm).",
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
    compute.add_argument(
        "--clim-helmholtz-dir", type=Path, default=None,
        help="Directory with pre-computed Helmholtz climatology NetCDFs.  "
             "Defaults to the same directory as --clim-path.",
    )
    compute.add_argument(
        "--with-ppvi", action=argparse.BooleanOptionalAction, default=True,
        help="Also compute the piecewise PV-inversion rotational winds "
             "(u/v_rot_anom_ppvi_*) in each NPZ written from scratch "
             "(default: enabled).  Use --no-with-ppvi to skip.",
    )
    compute.add_argument(
        "--max-tasks-per-child", type=int, default=16,
        help="Recycle each worker process after this many events to cap "
             "per-event memory growth (0 = never recycle).  Default 16.",
    )
    _add_ppvi_pieces_arg(compute)

    # ── ppvi ─────────────────────────────────────────────────────
    ppvi = sub.add_parser(
        "ppvi",
        help="Add piecewise PV-inversion rotational winds to NPZ files: "
             "append in-place to existing NPZ, or write base + PPVI "
             "together for missing NPZ.",
    )
    ppvi.add_argument(
        "--event-type", required=True, choices=["blocking", "prp"],
        help="Event type.",
    )
    ppvi.add_argument(
        "--events-csv", required=True, type=Path,
        help="CSV with columns: evt_name, track_id, lat0, lon0, base_ts.",
    )
    ppvi.add_argument(
        "--source", choices=["era5", "cesm"], default="era5",
        help="Input dataset family (default: era5). 'cesm' reads the "
             "CESM2-LENS2 member-year archive and needs --member.",
    )
    ppvi.add_argument(
        "--member", type=int, default=None,
        help="LENS2 member number (91-100 for the 6-hourly smbb set). "
             "Required with --source cesm: the member selects the file and "
             "cannot be inferred from the event row, which is why the event "
             "CSVs are written one per member.",
    )
    ppvi.add_argument(
        "--era5-dir", required=True, type=Path,
        help="Directory with the input NetCDF files (ERA5 monthly, or the "
             "CESM member-year archive when --source cesm).",
    )
    ppvi.add_argument(
        "--clim-path", required=True, type=Path,
        help="Climatology file or directory.",
    )
    ppvi.add_argument(
        "--out-dir", required=True, type=Path,
        help="Root NPZ directory. Existing base NPZ are appended in-place; "
             "missing ones are written together with their base fields.",
    )
    ppvi.add_argument(
        "--track-file", type=Path, default=None,
        help="Tracking data file for Lagrangian mode.",
    )
    ppvi.add_argument(
        "--dh-range", type=str, default="-12:13:1",
        help="Hour offsets as start:stop[:step]  (default: -12:13:1).",
    )
    ppvi.add_argument(
        "--center-mode", choices=["eulerian", "lagrangian"],
        default="eulerian",
        help="Centre tracking mode (default: eulerian).",
    )
    ppvi.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel worker processes (serial solver per event).",
    )
    ppvi.add_argument(
        "--year-range", type=str, default=None,
        help="Filter events to year range, e.g. '1990:2011' "
             "(start:stop_exclusive).",
    )
    ppvi.add_argument(
        "--stages", nargs="+", default=["onset", "peak", "decay"],
        help="Event stages to process (default: onset peak decay).",
    )
    ppvi.add_argument(
        "--skip-existing", action="store_true",
        help="Skip NPZ files that already contain PPVI fields.",
    )
    ppvi.add_argument(
        "--max-tasks-per-child", type=int, default=16,
        help="Recycle each worker process after this many events to cap "
             "per-event memory growth (0 = never recycle).  Default 16.",
    )
    ppvi.add_argument(
        "--clim-helmholtz-dir", type=Path, default=None,
        help="Directory with pre-computed Helmholtz climatology NetCDFs. "
             "Defaults to the same directory as --clim-path.",
    )
    _add_ppvi_pieces_arg(ppvi)

    # ── clim-helmholtz ───────────────────────────────────────────
    # Restored in 2.17.2: the block sat directly below the one-time `qsplit`
    # retrofit and was deleted with it in f421e06, leaving _cmd_clim_helmholtz
    # and its dispatch entry reachable only in principle -- `pvtend-pipeline
    # clim-helmholtz` failed on argparse's invalid-choice. The README's CLI
    # bullet never stopped listing it, and the ERA5 climatology directory it
    # produces (*_u_helmholtz.nc / *_v_helmholtz.nc, fed back through
    # --clim-helmholtz-dir) is still what the pipeline consumes.
    clim_helm = sub.add_parser(
        "clim-helmholtz",
        help="Pre-compute Helmholtz decomposition of climatological wind → NetCDFs.",
    )
    clim_helm.add_argument(
        "--clim-dir", required=True, type=Path,
        help="Directory with per-variable-per-month climatology NetCDFs.",
    )
    clim_helm.add_argument(
        "--output-dir", required=True, type=Path,
        help="Output directory for the 24 Helmholtz climatology files.",
    )
    clim_helm.add_argument(
        "--clim-stem", type=str, default="era5_hourly_clim_1990-2020",
        help="Filename stem for the climatology files (default: era5_hourly_clim_1990-2020).",
    )

    # ── classify ─────────────────────────────────────────────────
    classify = sub.add_parser(
        "classify",
        help="RWB classification (Pass 1) → variant tracksets PKL.",
    )
    classify.add_argument(
        "--npz-dir", required=False, type=Path, default=None,
        help="Root NPZ directory (with onset/peak/decay sub-dirs). Default "
             "mode. Mutually exclusive with --patches.",
    )
    classify.add_argument(
        "--patches", type=Path, default=None,
        help="Second option: classify a pre-extracted patch file (.npz or "
             ".nc) of single 2-D Z fields instead of the NPZ tree. The file "
             "must hold a Z field (var z/patch_z/patch_zwavg, shape (N,NY,NX)), "
             "relative coords (rel_lon/x_rel (NX,), rel_lat/y_rel (NY,)), and "
             "an identifier (var id (N,), or member+track). Writes a per-event "
             "label CSV to --output (single-field wavg mode; --threshold/--levels "
             "ignored).",
    )
    classify.add_argument(
        "--output", required=True, type=Path,
        help="Output pickle (NPZ-tree mode) or label CSV (--patches mode).",
    )
    classify.add_argument(
        "--stages", nargs="+", default=["onset", "peak", "decay"],
        help="Event stages to classify (default: onset peak decay).",
    )
    classify.add_argument(
        "--levels", nargs="+", type=_level_type, default=[500, 400, 300, 200],
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
    classify.add_argument(
        "--n-workers", type=int, default=1,
        help="Parallel worker processes for the per-file classification "
             "loop (NPZ-tree mode only; default 1 = serial).",
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
    composite.add_argument(
        "--n-workers", type=int, default=None,
        help="Parallel workers for composite accumulation (default: the "
             "PVTEND_COMPOSITE_WORKERS env var, else serial).",
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

    # ── blowup-scan ──────────────────────────────────────────────
    blowup = sub.add_parser(
        "blowup-scan",
        help=("[Deprecated in v2.10.8] Detect raw ERA5 ω blowups. "
              "Prefer scripts/aggregate_qg_blowup.py which scans NPZ "
              "max_abs_w_*_300 metrics — only QG-omega solver output is "
              "ever blowup-prone; raw ERA5 ω is observationally bounded."),
    )
    blowup.add_argument(
        "--era5", required=True, type=str,
        help="Glob expression for ERA5 ω files (e.g. /…/era5_w_*.nc).",
    )
    blowup.add_argument(
        "--level", type=float, default=300.0,
        help="Pressure level in hPa (default 300).",
    )
    blowup.add_argument(
        "--threshold", type=float, default=25.0,
        help=("Hard cutoff abs(omega) in Pa/s (default 25.0).  "
              "Empirical raw-ERA5 envelope at 300 hPa over 1990-2020 "
              "hourly: max=22.4, 99.9th=19.9 Pa/s; QG-solver output "
              ">25 Pa/s = solver pathology."),
    )
    blowup.add_argument(
        "--out", required=True, type=Path,
        help="Output CSV path.",
    )

    return parser


def _level_type(value: str) -> int | str:
    """Argparse type for --levels: return int for numbers, str for 'wavg'."""
    if value.lower() == "wavg":
        return "wavg"
    return int(value)


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
_WORKER_COMPUTER = None  # persistent TendencyComputer per worker process


def _pin_threads_for_pool() -> None:
    """Pin BLAS/FFT/OpenMP to 1 thread per worker process.

    Called in the **parent** before a ``ProcessPoolExecutor`` is created so
    spawned children inherit the environment **before** numpy/xarray are
    imported. With many workers each event is single-threaded; the batch is
    parallel at the process level. Prevents thread oversubscription on the
    384-core node (N_workers × BLAS-threads ≫ cores → cache thrash).
    """
    import os
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        # Assigned unconditionally: an inherited OMP_NUM_THREADS=8 from a
        # login profile would put N_workers × 8 threads on the node, the
        # exact oversubscription this function exists to prevent. Only the
        # pool path calls this, so single-process tooling is unaffected.
        os.environ[_v] = "1"
    # HDF5's own fcntl locking goes through NFS lockd — one extra lock
    # round-trip per open across every worker, plus the spurious
    # "unable to lock file" failure mode. Nothing in the pipeline writes an
    # input .nc concurrently, so the locks protect nothing here.
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# Whether the compute pool also computes PPVI fields (set per-run).
_WORKER_ALSO_PPVI = True


def _init_worker(cfg, also_ppvi=True):
    """Pool initializer — store config + a persistent computer in globals."""
    global _WORKER_CFG, _WORKER_COMPUTER, _WORKER_ALSO_PPVI
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    try:  # bound xarray's global open-file-handle cache so a long-lived
        # worker doesn't accumulate ~100 MB/handle across thousands of
        # per-event opens (mirrors the ppvi initializer).
        import xarray as _xr
        _xr.set_options(file_cache_maxsize=32)
    except Exception:
        pass
    try:  # one process = one thread is the design of this pool; dask's
        # default threaded scheduler would spin cpu_count() threads per
        # worker and hit the lock=False netCDF handles concurrently.
        import dask
        dask.config.set(scheduler="synchronous")
    except Exception:
        pass
    _WORKER_CFG = cfg
    _WORKER_ALSO_PPVI = also_ppvi
    from pvtend.tendency import TendencyComputer
    _WORKER_COMPUTER = TendencyComputer(cfg)
    try:  # warm the climatology once per worker (not once per event),
        # small-chunked: ``_get_clim`` honours chunks only on this first
        # call, and an unchunked warm pins the whole CESM slot file in the
        # backend cache (filtered to a no-op on the ERA5 layout).
        _WORKER_COMPUTER._get_clim(chunks={"day": 1, "hour": 1})
    except Exception:
        pass


def _process_one_event(arg_tuple):
    """Process a single event in a worker process (reuses cached computer)."""
    evt_name, track_id, lat0, lon0, base_ts = arg_tuple
    try:
        n = _WORKER_COMPUTER.process_event(
            evt_name=evt_name, track_id=track_id,
            lat0=lat0, lon0=lon0, base_ts=base_ts,
            also_ppvi=_WORKER_ALSO_PPVI)
        return n
    except Exception as exc:
        print(f"    ERROR [{track_id} {evt_name}]: {exc}", flush=True)
        return 0


def _init_ppvi_worker(cfg):
    """Pool initializer for the ppvi subcommand (persistent computer)."""
    global _WORKER_CFG, _WORKER_COMPUTER
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    try:  # bound xarray's global open-file-handle cache so a long-lived
        # worker doesn't accumulate ~100 MB/handle across thousands of
        # per-event ERA5 opens (this was the dominant per-worker RSS growth).
        import xarray as _xr
        _xr.set_options(file_cache_maxsize=32)
    except Exception:
        pass
    try:  # one process = one thread (see _init_worker)
        import dask
        dask.config.set(scheduler="synchronous")
    except Exception:
        pass
    _WORKER_CFG = cfg
    from pvtend.tendency import TendencyComputer
    _WORKER_COMPUTER = TendencyComputer(cfg)
    try:  # warm the climatology once per worker (small-chunked for PPVI so
        # per-event slices are released, not retained → bounded worker RSS)
        _WORKER_COMPUTER._get_clim(chunks={"day": 1, "hour": 1})
    except Exception:
        pass


def _ppvi_one_event(arg_tuple):
    """Append/write PPVI fields for a single event (reuses cached computer)."""
    evt_name, track_id, lat0, lon0, base_ts = arg_tuple
    try:
        return _ppvi_run_event(
            _WORKER_COMPUTER, evt_name, track_id, lat0, lon0, base_ts)
    except Exception as exc:
        print(f"    ERROR [{track_id} {evt_name}]: {exc}", flush=True)
        return 0


def _ppvi_run_event(computer, evt_name, track_id, lat0, lon0, base_ts):
    """Run PPVI for one event: append to existing NPZ, write-together fresh.

    1. ``compute_ppvi_for_event`` appends the PPVI fields to any base NPZ
       that already exists on disk (the common case).
    2. ``process_event(also_ppvi=True)`` then produces any **missing** base
       NPZ with the base tendency + PPVI fields written **together** in a
       single file (fresh run / new machine — no append).

    When ``skip_existing=False`` the base NPZ are recomputed from scratch,
    so only the write-together path runs (appending would be wasted work).
    """
    if not computer.cfg.skip_existing:
        return computer.process_event(
            evt_name=evt_name, track_id=track_id,
            lat0=lat0, lon0=lon0, base_ts=base_ts, also_ppvi=True)
    n = computer.compute_ppvi_for_event(
        evt_name=evt_name, track_id=track_id,
        lat0=lat0, lon0=lon0, base_ts=base_ts)
    n += computer.process_event(
        evt_name=evt_name, track_id=track_id,
        lat0=lat0, lon0=lon0, base_ts=base_ts, also_ppvi=True)
    return n


def _run_resilient_pool(event_args, n_workers, initializer, initargs,
                        worker_fn, *, max_attempts=4, failures_path=None,
                        unit="NPZ", max_tasks_per_child=None):
    """Run ``worker_fn`` over ``event_args`` in a fault-tolerant process pool.

    A single worker dying (e.g. a C-level segfault in a compiled kernel
    that Python cannot catch) marks a :class:`ProcessPoolExecutor` as
    *broken* and poisons **every** outstanding future.  A naive
    ``future.result()`` loop therefore aborts the whole run on the first bad
    event.  This supervisor instead rebuilds the pool over the still-pending
    events whenever it breaks, so completed work is preserved and the run
    makes forward progress.

    An event that repeatedly kills its worker (``>= max_attempts`` times) is
    *quarantined* — recorded to ``failures_path`` and skipped — so one
    pathological geometry can never wedge the run forever.

    Because the underlying work is idempotent on disk (``skip_existing``),
    re-submitting an already-finished event is cheap (it is detected and
    skipped), so exact in-memory bookkeeping is not safety-critical.

    Args:
        event_args: List of per-event argument tuples passed to ``worker_fn``.
        n_workers: Maximum worker processes per pool.
        initializer: Pool initializer callable.
        initargs: Tuple of arguments for ``initializer``.
        worker_fn: Callable run for each event; returns an int file count.
        max_attempts: Quarantine an event after this many failed attempts.
        failures_path: Optional path to write the quarantined event tuples.
        unit: Noun for progress messages (e.g. ``"NPZ"``).

    Returns:
        Tuple ``(n_total, quarantined_indices)``.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    n_events = len(event_args)
    done: set[int] = set()
    quarantined: set[int] = set()
    attempts = [0] * n_events
    n_total = 0
    n_completed = 0
    restart = 0
    broke_count = 0

    while True:
        pending = [i for i in range(n_events)
                   if i not in done and i not in quarantined]
        if not pending:
            break
        if restart:
            print(f"[pvtend] RESILIENT restart #{restart}: "
                  f"{len(pending)} events remaining "
                  f"({len(done)} done, {len(quarantined)} quarantined)",
                  flush=True)
        restart += 1

        broke = False
        _pool_kw = {}
        if max_tasks_per_child:
            # Recycle each worker after N events so accumulated per-event
            # memory (xarray/dask caches) is freed — a hard cap independent
            # of any single leak source. Requires a non-fork start method.
            import multiprocessing as _mp
            _pool_kw["max_tasks_per_child"] = max_tasks_per_child
            _pool_kw["mp_context"] = _mp.get_context("spawn")
        ex = ProcessPoolExecutor(
            max_workers=min(n_workers, len(pending)),
            initializer=initializer, initargs=initargs, **_pool_kw)
        futs = {ex.submit(worker_fn, event_args[i]): i for i in pending}
        try:
            for f in as_completed(futs):
                i = futs[f]
                try:
                    n_total += f.result()
                    done.add(i)
                    n_completed += 1
                    if n_completed % 50 == 0 or n_completed == n_events:
                        print(f"[pvtend] {n_completed}/{n_events} events done, "
                              f"{n_total} {unit} updated so far", flush=True)
                except BrokenProcessPool:
                    broke = True  # pool poisoned; harvest the rest, then rebuild
                except Exception as exc:  # noqa: BLE001 - worker-level failure
                    attempts[i] += 1
                    print(f"    ERROR event#{i}: {exc!r} "
                          f"(attempt {attempts[i]}/{max_attempts})", flush=True)
                    if attempts[i] >= max_attempts:
                        quarantined.add(i)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        if broke:
            broke_count += 1
            still = [i for i in range(n_events)
                     if i not in done and i not in quarantined]
            # A dead worker poisons the pool and drops its in-flight events, but
            # we cannot tell WHICH event killed it — so do NOT penalise every
            # pending event (that mass-quarantines healthy ones, which was the
            # cause of ~26% spurious migration loss). Just rebuild and retry;
            # healthy events succeed next round. Guard a truly poisonous set
            # with a generous global break cap.
            print(f"[pvtend] WARNING: worker pool broke (break #{broke_count}); "
                  f"{len(still)} events pending; rebuilding pool.", flush=True)
            if broke_count > 500:
                for i in still:
                    quarantined.add(i)
                print(f"[pvtend] aborting after {broke_count} pool breaks; "
                      f"quarantined {len(still)} remaining event(s).",
                      flush=True)

    if quarantined and failures_path is not None:
        try:
            with open(failures_path, "w") as fh:
                fh.write("# events quarantined after repeated worker deaths\n")
                fh.write("# evt_name, track_id, lat0, lon0, base_ts\n")
                for i in sorted(quarantined):
                    fh.write(f"{event_args[i]}\n")
            print(f"[pvtend] {len(quarantined)} event(s) quarantined; "
                  f"written to {failures_path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[pvtend] (could not write failures file: {exc})",
                  flush=True)

    return n_total, quarantined



# =====================================================================
# Subcommand implementations
# =====================================================================

def _load_event_args(events_csv: Path, year_range, stages):
    """Read an events CSV → list of (evt_name, track_id, lat0, lon0, base_ts).

    Handles legacy column names, year-range filtering, and stage filtering
    (shared by the ``compute`` and ``ppvi`` subcommands).
    """
    import pandas as pd

    events_df = pd.read_csv(events_csv)

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
    if year_range is not None:
        yr_parts = year_range.split(":")
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
            events_df[stage_col].isin(stages)
        ].reset_index(drop=True)
        print(f"[pvtend] Filtered to stages {stages}: "
              f"{len(events_df)} events")

    event_args = []
    for idx, row in events_df.iterrows():
        evt_name = str(row.get("evt_name", row.get("stage", "onset")))
        # track_id only has to name the NPZ uniquely within the output tree, and
        # `_out_path` interpolates it into an f-string, so a non-numeric id is
        # fine. The unconditional int() broke the CESM catalogue: track_id
        # restarts at 1 in every member there, so the event CSVs carry
        # "m091_t00002" to stop the ten members colliding in a shared output
        # directory. Numeric ids still become ints, so ERA5 filenames are
        # unchanged.
        raw_tid = row.get("track_id", idx)
        try:
            track_id = int(raw_tid)
        except (TypeError, ValueError):
            track_id = str(raw_tid)
        lat0 = float(row["lat0"])
        lon0 = float(row["lon0"])
        base_ts = pd.Timestamp(str(row.get("base_ts",
                                            row.get("timestamp"))))
        event_args.append((evt_name, track_id, lat0, lon0, base_ts))
    return event_args


def _assert_member_matches_events(event_args, member) -> None:
    """Refuse a --member that contradicts the event CSV's own track ids.

    The member selects the archive file while the CSV supplies the events;
    every timestamp exists in every member, so running member 92 against the
    m091 catalogue would silently write member-92 fields under m091 ids.
    Only m-prefixed (CESM) string ids carry the member and can be checked.
    """
    if member is None:
        return
    want = f"m{int(member):03d}_"
    bad = sorted({str(tid).split("_")[0] for _, tid, *_ in event_args
                  if isinstance(tid, str) and tid.startswith("m")
                  and not str(tid).startswith(want)})
    if bad:
        raise SystemExit(
            f"--member {member} but the events CSV carries track ids for "
            f"member(s) {bad} — wrong per-member CSV?"
        )


def _cmd_compute(args: argparse.Namespace) -> None:
    """Execute the ``compute`` subcommand."""
    from pvtend.tendency import TendencyComputer, TendencyConfig

    dh_values = _parse_dh_range(args.dh_range)

    qg = args.qg_method

    if args.source == "cesm" and args.member is None:
        raise SystemExit(
            "--source cesm requires --member: the member selects which "
            "lens2_smbb_m{member}_{year}_plev.nc to read and cannot be "
            "inferred from the event rows.")
    if args.source != "cesm" and args.member is not None:
        raise SystemExit("--member only applies to --source cesm")

    config = TendencyConfig(
        event_type=args.event_type,
        data_dir=args.era5_dir,
        source=args.source,
        member=args.member,
        clim_path=args.clim_path,
        output_dir=args.out_dir,
        csv_path=args.events_csv,
        track_file=args.track_file or Path(""),
        rel_hours=dh_values,
        qg_omega_method=qg,
        center_mode=args.center_mode,
        skip_existing=args.skip_existing,
        n_workers=args.n_workers,
        ppvi_pieces=args.ppvi_pieces,
        ppvi_solver_nlat=int(args.ppvi_solver_grid[0]),
        ppvi_solver_nlon=int(args.ppvi_solver_grid[1]),
        ppvi_newton_max_steps=int(args.ppvi_newton_max_steps),
        clim_helmholtz_dir=(
            args.clim_helmholtz_dir
            if args.clim_helmholtz_dir is not None
            else args.clim_path
            if args.clim_path.is_dir()
            else args.clim_path.parent
        ),
    )
    computer = TendencyComputer(config)

    event_args = _load_event_args(args.events_csv, args.year_range,
                                  args.stages)
    _assert_member_matches_events(event_args, getattr(args, "member", None))

    print(f"[pvtend] Processing {len(event_args)} events, "
          f"dh={dh_values[0]}..{dh_values[-1]}, qg_method={qg}, "
          f"ppvi={'on' if args.with_ppvi else 'off'} "
          f"({args.ppvi_pieces})")

    if config.n_workers > 1:
        _pin_threads_for_pool()
        print(f"[pvtend] Using {config.n_workers} parallel workers "
              f"(fault-tolerant pool)")
        n_total, _quarantined = _run_resilient_pool(
            event_args, config.n_workers,
            _init_worker, (config, args.with_ppvi),
            _process_one_event,
            failures_path=args.out_dir / "compute_failures.txt",
            unit="NPZ",
            max_tasks_per_child=args.max_tasks_per_child)
    else:
        n_total = 0
        for i, (evt_name, track_id, lat0, lon0, base_ts) in enumerate(
                event_args):
            print(f"  Event {i + 1}/{len(event_args)}: "
                  f"track_id={track_id}  {evt_name}  {base_ts}")
            try:
                n = computer.process_event(
                    evt_name=evt_name, track_id=track_id,
                    lat0=lat0, lon0=lon0, base_ts=base_ts,
                    also_ppvi=args.with_ppvi)
                n_total += n
            except Exception as exc:
                print(f"    ERROR: {exc}")
                continue

    print(f"[pvtend] Done — wrote {n_total} NPZ files.")


def _cmd_ppvi(args: argparse.Namespace) -> None:
    """Execute the ``ppvi`` subcommand (append existing / write fresh)."""
    from pvtend.tendency import TendencyComputer, TendencyConfig

    dh_values = _parse_dh_range(args.dh_range)

    if args.source == "cesm" and args.member is None:
        raise SystemExit(
            "--source cesm requires --member: the member selects which "
            "lens2_smbb_m{member}_{year}_plev.nc to read and cannot be "
            "inferred from the event rows.")
    if args.source != "cesm" and args.member is not None:
        raise SystemExit("--member only applies to --source cesm")

    config = TendencyConfig(
        event_type=args.event_type,
        data_dir=args.era5_dir,
        source=args.source,
        member=args.member,
        clim_path=args.clim_path,
        output_dir=args.out_dir,
        csv_path=args.events_csv,
        track_file=args.track_file or Path(""),
        rel_hours=dh_values,
        center_mode=args.center_mode,
        skip_existing=args.skip_existing,
        n_workers=args.n_workers,
        ppvi_pieces=args.ppvi_pieces,
        ppvi_solver_nlat=int(args.ppvi_solver_grid[0]),
        ppvi_solver_nlon=int(args.ppvi_solver_grid[1]),
        ppvi_newton_max_steps=int(args.ppvi_newton_max_steps),
        clim_helmholtz_dir=(
            args.clim_helmholtz_dir
            if args.clim_helmholtz_dir is not None
            else args.clim_path
            if args.clim_path.is_dir()
            else args.clim_path.parent
        ),
    )
    computer = TendencyComputer(config)

    event_args = _load_event_args(args.events_csv, args.year_range,
                                  args.stages)
    _assert_member_matches_events(event_args, getattr(args, "member", None))

    print(f"[pvtend] PPVI for {len(event_args)} events, "
          f"dh={dh_values[0]}..{dh_values[-1]}, "
          f"pieces={args.ppvi_pieces}")

    if config.n_workers > 1:
        _pin_threads_for_pool()
        print(f"[pvtend] Using {config.n_workers} parallel workers "
              f"(fault-tolerant pool)")
        n_total, _quarantined = _run_resilient_pool(
            event_args, config.n_workers,
            _init_ppvi_worker, (config,),
            _ppvi_one_event,
            failures_path=args.out_dir / "ppvi_failures.txt",
            unit="NPZ",
            # Recycle each worker after this many events so per-event xarray/
            # HDF5 cache growth (~0.5 GB/event) can't accumulate unbounded and
            # push the cgroup to its 2 TB limit. ~1% clim-rewarm overhead.
            max_tasks_per_child=args.max_tasks_per_child)
    else:
        n_total = 0
        for i, (evt_name, track_id, lat0, lon0, base_ts) in enumerate(
                event_args):
            print(f"  Event {i + 1}/{len(event_args)}: "
                  f"track_id={track_id}  {evt_name}  {base_ts}")
            try:
                n_total += _ppvi_run_event(
                    computer, evt_name, track_id, lat0, lon0, base_ts)
            except Exception as exc:
                print(f"    ERROR: {exc}")
                continue

    print(f"[pvtend] Done — updated {n_total} NPZ files with PPVI fields.")



def _cmd_clim_helmholtz(args: argparse.Namespace) -> None:
    """Execute the ``clim-helmholtz`` subcommand."""
    import numpy as np
    from pvtend.climatology import compute_helmholtz_climatology
    from pvtend.grid import default_nh_grid

    grid = default_nh_grid()
    lat = grid.lat
    lon = grid.lon

    written = compute_helmholtz_climatology(
        clim_dir=args.clim_dir,
        output_dir=args.output_dir,
        lat=lat,
        lon=lon,
        clim_stem=args.clim_stem,
    )
    print(f"[pvtend] Wrote {len(written)} Helmholtz climatology files "
          f"to {args.output_dir}")


def _cmd_classify_patches(args: argparse.Namespace) -> None:
    """Classify a pre-extracted patch file (.npz/.nc) of single 2-D Z fields.

    The "second option" for ``classify``: instead of the per-event NPZ tree,
    read a stack of block-relative weighted-average Z patches (e.g. built from
    CESM f09 nc) and write a per-event AWB/CWB/Omega/NEUTRAL label CSV.
    """
    import csv
    import numpy as np
    from pvtend.classify import classify_z_field, label_from_flags
    from pvtend.rwb import RWBConfig

    fp = args.patches

    def _pick(d, names):
        for n in names:
            if n in d:
                return np.asarray(d[n])
        raise KeyError(f"patch file missing any of {names}; has {list(d)}")

    if fp.suffix.lower() in (".nc", ".nc4", ".netcdf"):
        import xarray as xr
        ds = xr.open_dataset(fp)
        d = {k: ds[k].values for k in ds.variables}
        ds.close()
    else:
        with np.load(fp, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}

    Z = _pick(d, ("z", "patch_zwavg", "patch_z"))          # (N, NY, NX) or (NY, NX)
    if Z.ndim == 2:                                        # single patch → stack of 1
        Z = Z[None]
    if Z.ndim != 3:
        raise SystemExit(
            f"classify --patches: Z field has shape {Z.shape}; expected "
            f"(N, NY, NX) or (NY, NX). Check the field variable is not a "
            f"coordinate (rel_lon/rel_lat) — picked from {list(d)}.")
    n = Z.shape[0]
    x_rel = _pick(d, ("rel_lon", "x_rel"))
    y_rel = _pick(d, ("rel_lat", "y_rel"))
    try:
        if "id" in d:
            ident = [(int(i),) for i in np.asarray(d["id"]).ravel()]
            cols = ["id"]
        elif "member" in d and "track" in d:
            ident = list(zip(np.asarray(d["member"]).astype(int).tolist(),
                             np.asarray(d["track"]).astype(int).tolist()))
            cols = ["member", "track"]
        else:
            ident = [(i,) for i in range(n)]
            cols = ["id"]
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"classify --patches: non-integer identifier — {exc}")
    if len(ident) != n:
        raise SystemExit(
            f"classify --patches: {len(ident)} identifiers but {n} patches.")

    # Match the ERA5 classify-CLI RWB tuning (see _cmd_classify) for consistency.
    cfg = RWBConfig(area_min_deg2=20.0, try_levels=400)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"AWB": 0, "CWB": 0, "Omega": 0, "NEUTRAL": 0}
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([*cols, "label", "awb", "cwb"])
        for k in range(Z.shape[0]):
            awb, cwb = classify_z_field(Z[k], x_rel, y_rel, cfg)
            lab = label_from_flags(awb, cwb)
            counts[lab] += 1
            w.writerow([*ident[k], lab, int(awb), int(cwb)])
    print(f"[pvtend] classified {Z.shape[0]} patches from {fp.name}: "
          + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"[pvtend] labels saved to {args.output}")


def _cmd_classify(args: argparse.Namespace) -> None:
    """Execute the ``classify`` subcommand (Pass 1)."""
    from pvtend.classify import ClassifyConfig, run_pass1
    from pvtend.rwb import RWBConfig

    # Second option: a precomputed patch file (.npz/.nc) instead of the NPZ tree.
    if getattr(args, "patches", None) is not None and args.npz_dir is not None:
        raise SystemExit("classify: --npz-dir and --patches are mutually "
                         "exclusive; pass exactly one.")
    if getattr(args, "patches", None) is not None:
        _cmd_classify_patches(args)
        return
    if args.npz_dir is None:
        raise SystemExit("classify: one of --npz-dir or --patches is required.")

    # Clamp threshold to number of levels: with N levels you cannot demand
    # agreement from more than N. Prevents silent all-Neutral output when
    # users pass e.g. `--levels wavg` (N=1) with the default threshold=3.
    threshold = min(args.threshold, len(args.levels))
    if threshold != args.threshold:
        print(f"[pvtend] Warning: --threshold {args.threshold} > "
              f"len(--levels)={len(args.levels)}; clamping to {threshold}.",
              flush=True)

    cfg = ClassifyConfig(
        npz_dir=args.npz_dir,
        output_path=args.output,
        stages=args.stages,
        classify_levels=args.levels,
        classify_threshold=threshold,
        rwb_cfg=RWBConfig(area_min_deg2=20.0, try_levels=400),
        exclude_file=args.exclude_file,
        n_workers=args.n_workers,
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

    cfg_kwargs = dict(
        npz_dir=args.npz_dir,
        stages=args.stages,
        exclude_file=args.exclude_file,
    )
    if args.n_workers is not None:
        cfg_kwargs["n_workers"] = args.n_workers
    cfg = CompositeConfig(**cfg_kwargs)
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


def _cmd_blowup_scan(args: argparse.Namespace) -> None:
    """Execute the ``blowup-scan`` subcommand."""
    from pvtend.blowup import scan_omega_blowup

    level_pa = float(args.level) * 100.0
    print(f"[pvtend] Scanning ω blowups at {args.level:.0f} hPa, "
          f"|ω| > {args.threshold:.2f} Pa/s ...")
    df = scan_omega_blowup(
        era5_w_glob=args.era5,
        level_pa=level_pa,
        threshold=args.threshold,
        out_csv=args.out,
    )
    print(f"[pvtend] {len(df)} blowup timestamps written to {args.out}")


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
        "ppvi": _cmd_ppvi,
        "clim-helmholtz": _cmd_clim_helmholtz,
        "classify": _cmd_classify,
        "composite": _cmd_composite,
        "decompose": _cmd_decompose,
        "blowup-scan": _cmd_blowup_scan,
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
