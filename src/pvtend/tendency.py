"""PV tendency term computation for weather events.

Orchestrates the full computation pipeline for a single event:

1. Load ERA5 data for the time window
2. Subtract climatology to get anomalies
3. Compute all spatial/temporal derivatives
4. FFT Helmholtz decomposition on the full NH hemisphere
5. QG omega solver → omega_dry
6. Moist/dry decomposition → omega_moist, chi_moist
7. Extract event-centred patches
8. Compute PV cross-terms and vertical weighted averages
9. Write per-timestep NPZ files

The :class:`TendencyComputer` class is parameterized by event type
(blocking / PRP), eliminating the 95 % code duplication between the
original scripts.  Ported from
``tempest_extreme_4_basis/core/step2_compute_tendency_terms_blocking.py``
and ``step2_compute_tendency_terms_prp.py``.
"""

from __future__ import annotations

import gc
import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from .constants import (
    CP_DRY,
    DEFAULT_LEVELS,
    G0,
    H_SCALE,
    KAPPA,
    OMEGA_E,
    R_DRY,
    R_EARTH,
    WAVG_LEVELS,
    LAT_HALF,
    LON_HALF,
    GEO_SMOOTH_SIGMA,
    F_MIN_LAT,
    LAT_QG_LO,
    LAT_QG_HI,
    LAT_QG_POLAR,
    SP19_DRY_FRACTION,
    CLIM_VARIABLES,
    MONTH_ABBREVS,
)
from .derivatives import ddx, ddy, ddp, ddt
from .helmholtz import helmholtz_decomposition, solve_poisson_spherical_fft, gradient
from .climatology import load_helmholtz_climatology
from .omega import (
    solve_qg_omega_sip,
    _compute_diabatic_rhs_log20,
    _compute_diabatic_rhs_emanuel,
)

# Alias for brevity
LEVELS = DEFAULT_LEVELS

# Superseded grouped-piece PPVI keys (lower/middle/upper), now replaced by the
# per-level ``*_ppvi_{L}`` keys. Used to drop them when re-running PPVI in place.
_OLD_PIECE_KEY = re.compile(r"^[uv]_rot_anom_(lower|middle|upper)(_3d)?$")

# ── Full list of variables stored in each NPZ ──────────────────────────
VARS_3D: list[str] = [
    "z", "pv", "u", "v", "w", "t", "q", "t_dt",
    "pv_dt", "pv_total_dx", "pv_total_dy", "pv_total_dp",
    "u_bar", "v_bar", "w_bar", "pv_bar",
    "u_anom", "v_anom", "w_anom", "pv_anom",
    "pv_bar_dx", "pv_bar_dy", "pv_bar_dp", "pv_bar_dt",
    "pv_anom_dx", "pv_anom_dy", "pv_anom_dp", "pv_anom_dt",
    "theta", "theta_dt", "theta_dot", "Q",
    # Total-wind Helmholtz (v2.0)
    "u_rot", "u_div", "v_rot", "v_div",
    # Climatological Helmholtz (v2.0)
    "u_rot_bar", "u_div_bar", "v_rot_bar", "v_div_bar",
    # Anomaly Helmholtz (total − clim)
    "u_rot_anom", "u_div_anom", "u_har_anom",
    "v_rot_anom", "v_div_anom", "v_har_anom",
    "u_div_diabatic", "v_div_diabatic",
    "u_div_adiabatic", "v_div_adiabatic",
    "u_div_qg_diabatic", "v_div_qg_diabatic",
    "w_adiabatic", "w_diabatic", "w_qg_diabatic",
    # Second-order PV derivatives (full field)
    "pv_total_dx_dx", "pv_total_dy_dy", "pv_total_dx_dy",
]

# Variables extracted from the DS for each patch
_EXTRACT_VARS = [
    "z", "pv", "u", "v", "w", "t", "q", "t_dt",
    "pv_dt", "pv_total_dx", "pv_total_dy", "pv_total_dp",
    "u_bar", "v_bar", "w_bar", "pv_bar",
    "u_anom", "v_anom", "w_anom", "pv_anom",
    "pv_bar_dx", "pv_bar_dy", "pv_bar_dp", "pv_bar_dt",
    "pv_anom_dx", "pv_anom_dy", "pv_anom_dp", "pv_anom_dt",
    "theta", "theta_dt", "theta_dot", "Q",
    # Total-wind Helmholtz (v2.0)
    "u_rot", "u_div", "v_rot", "v_div",
    # Climatological Helmholtz (v2.0)
    "u_rot_bar", "u_div_bar", "v_rot_bar", "v_div_bar",
    # Anomaly Helmholtz (total − clim)
    "u_rot_anom", "u_div_anom", "u_har_anom",
    "v_rot_anom", "v_div_anom", "v_har_anom",
    "z_bar", "t_bar",
    # Second-order PV derivatives
    "pv_total_dx_dx", "pv_total_dy_dy", "pv_total_dx_dy",
]


def _log(msg: str) -> None:
    """Print with flush."""
    print(msg, flush=True)


# ============================================================================
#  Config
# ============================================================================
@dataclass
class TendencyConfig:
    """Configuration for PV tendency computation.

    Attributes:
        event_type: ``'blocking'`` or ``'prp'``.
        data_dir: Path to ERA5 monthly NetCDF files.
        clim_path: Path to climatology file or directory.
        clim_helmholtz_dir: Directory with pre-computed Helmholtz
            climatology files (from ``pvtend-pipeline clim-helmholtz``).
        output_dir: Root output directory for NPZ files.
        csv_path: Path to TempestExtremes event CSV.
        track_file: Path to tracking data file (for lagrangian mode).
        levels: Pressure levels [hPa].
        wavg_levels: Subset of *levels* used for vertical weighted averaging.
        rel_hours: Relative hour offsets from the event reference time.
        year_start: First event year.
        year_end: Last event year.
        lat_half: Half-width of the extraction patch in degrees latitude.
        lon_half: Half-width of the extraction patch in degrees longitude.
        partial_at_pole: Allow truncated patches near the poles.
        qg_omega_method: ``'log20'`` (SIP) or ``'sp19'`` (empirical scaling).
        center_mode: ``'eulerian'`` or ``'lagrangian'``.
        skip_existing: Skip events with existing NPZ output.
        engine: NetCDF engine passed to xarray.
        n_workers: Number of parallel workers for multiprocessing.
    """

    event_type: str = "blocking"
    data_dir: Path = Path("/net/flood/data2/users/x_yan/era")
    clim_path: Path = Path(
        "/net/flood/data2/users/x_yan/era/era5_hourly_clim_1990-2019.nc"
    )
    clim_helmholtz_dir: Path = Path(
        "/net/flood/data2/users/x_yan/era/clim"
    )
    output_dir: Path = Path(
        "/net/flood/data2/users/x_yan/composite_blocking_tempest"
    )
    csv_path: Path = Path("")
    track_file: Path = Path("")
    levels: list[int] = field(default_factory=lambda: list(LEVELS))
    wavg_levels: list[int] = field(default_factory=lambda: list(WAVG_LEVELS))
    rel_hours: list[int] = field(default_factory=lambda: list(range(-49, 25)))
    year_start: int = 1990
    year_end: int = 2020
    lat_half: float = LAT_HALF
    lon_half: float = LON_HALF
    partial_at_pole: bool = True
    #: Input dataset family: ``"era5"`` (per-variable monthly files, hourly,
    #: Gregorian) or ``"cesm"`` (per-member-year files, 6-hourly, noleap).
    source: str = "era5"
    #: LENS2 member number; required when ``source == "cesm"``, since the
    #: member selects the file and cannot be inferred from the event row.
    member: int | None = None
    qg_omega_method: str = "log20"
    center_mode: str = "eulerian"
    skip_existing: bool = True
    engine: str = "netcdf4"
    n_workers: int = 1


# ============================================================================
#  Climatology loading
# ============================================================================
def _find_per_var_month_files(parent: Path, stem: str) -> list[Path]:
    """Discover per-variable-per-month climatology files."""
    files = []
    for m_abbr in MONTH_ABBREVS:
        for var in CLIM_VARIABLES:
            f_raw = parent / f"{stem}_{m_abbr}_{var}.nc"
            if f_raw.exists():
                files.append(f_raw)
    return files


def load_climatology(
    clim_path: Path, engine: str = "netcdf4", chunks=None,
) -> xr.Dataset:
    """Load climatology, auto-detecting the file layout.

    Fallback order:
      1. Single merged file
      2. Per-var-per-month files
      3. Per-variable files

    ``chunks=None`` (default) keeps the lazy-backend layout. Pass a dask chunk
    spec (e.g. ``{"day":1,"hour":1}``) to back the variables with dask: each
    ``.sel(month,day,hour).values`` then computes and **releases** its slice
    instead of retaining it in the per-variable backend cache. This is what the
    PPVI workers use — without it a long-lived worker accumulates one ~0.4 GB
    month-chunk per accessed month/var (~19 GB/worker), throttling the cgroup.
    """
    clim_path = Path(clim_path)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*separate the stored chunks.*")
        if clim_path.is_file():
            _log(f"Loading climatology from single file: {clim_path}")
            ds = xr.open_dataset(clim_path, chunks=chunks, engine=engine,
                                 lock=False)
            # The CESM climatology is a single slot-indexed file still carrying
            # CAM names and the archive's 0…360 longitude. It has to be put on
            # the same footing as the state: `clim_bar` looks up "pv", and
            # xarray aligns on coordinate *values*, so a bar left on 0…360
            # would make `pv − pv_bar` NaN over the western hemisphere.
            if "slot" in ds.dims:
                ds = cesm_to_pipeline_names(ds)
            return ds

        parent = clim_path.parent
        stem = clim_path.stem.replace("_allvars", "")

        pvm_files = _find_per_var_month_files(parent, stem)
        if pvm_files:
            _log(f"Loading climatology from {len(pvm_files)} "
                 f"per-var-per-month files")
            return xr.open_mfdataset(
                [str(f) for f in pvm_files],
                chunks=chunks, engine=engine, lock=False,
                combine="by_coords", join="outer",
            )

        per_var = sorted(
            f for f in parent.glob(f"{stem}_*.nc")
            if "_smoothed" not in f.stem and "_allvars" not in f.stem
        )
        if per_var:
            _log(f"Loading climatology from {len(per_var)} per-variable files")
            return xr.open_mfdataset(per_var, chunks=chunks, engine=engine,
                                     lock=False)

    raise FileNotFoundError(
        f"Climatology missing: {clim_path} "
        f"(and no per-variable files matching '{stem}_*' in {parent})"
    )


# ============================================================================
#  ERA5 data loading helpers
# ============================================================================
def _ensure_valid_time(ds: xr.Dataset) -> xr.Dataset:
    if "valid_time" in ds.coords:
        return ds
    if "time" in ds.coords:
        return ds.rename({"time": "valid_time"})
    raise KeyError("Neither 'valid_time' nor 'time' coordinate present.")


def _drop_cds_artefacts(ds: xr.Dataset) -> xr.Dataset:
    to_drop = [v for v in ("number", "expver")
               if v in ds.coords or v in ds.data_vars]
    if to_drop:
        ds = ds.drop_vars(to_drop, errors="ignore")
    return ds


def _plev_name(ds: xr.Dataset) -> str:
    for nm in ("pressure_level", "level"):
        if nm in ds.dims or nm in ds.coords:
            return nm
    raise KeyError("No pressure level dimension.")


def month_keys_for_window(
    base_ts: pd.Timestamp, hmin: int = -49, hmax: int = 24,
) -> list[tuple[int, int]]:
    """Determine which (year, month) files are needed for the time window."""
    t0 = (pd.to_datetime(base_ts) + pd.Timedelta(hours=hmin)).to_period("M")
    t1 = (pd.to_datetime(base_ts) + pd.Timedelta(hours=hmax)).to_period("M")
    months = pd.period_range(t0, t1, freq="M")
    return [(int(p.year), int(p.month)) for p in months]


def open_months_ds(
    data_dir: Path, var_list: list[str], month_keys: list[tuple[int, int]],
    engine: str = "netcdf4", chunks=None,
) -> xr.Dataset:
    """Open multiple months of ERA5 data as a single dataset.

    ``chunks=None`` (default) loads each month eagerly — fine for the compute
    path that touches most timesteps of the window. Pass a dask chunk spec
    (e.g. ``{"valid_time": 1}``) to open lazily and only read the timesteps
    actually accessed — the PPVI path uses this so a worker materialises one
    (or a few) timesteps instead of the whole ~15 GB month, keeping per-worker
    RSS low enough to scale to many workers under the 2 TB cap.
    """
    data_dir = Path(data_dir)
    parts = []
    for v in var_list:
        fns = [
            str(data_dir / f"era5_{v}_{y}_{m:02d}.nc")
            for y, m in month_keys
            if (data_dir / f"era5_{v}_{y}_{m:02d}.nc").exists()
        ]
        if not fns:
            raise FileNotFoundError(f"No files for {v} in months {month_keys}")
        with warnings.catch_warnings():
            # Per-timestep chunking can split the stored netCDF chunks; that's
            # intentional here (memory over read-speed) — silence the notice.
            warnings.filterwarnings(
                "ignore", message=".*separate the stored chunks.*")
            dsv = xr.open_mfdataset(
                fns, combine="by_coords", parallel=False,
                chunks=chunks, engine=engine, lock=False,
            )
        dsv = _drop_cds_artefacts(dsv)
        dsv = _ensure_valid_time(dsv)
        if "level" in dsv.dims and "pressure_level" not in dsv.dims:
            dsv = dsv.rename({"level": "pressure_level"})
        dsv = dsv[[v]]
        parts.append(dsv)
    ds = xr.merge(parts, compat="no_conflicts", join="inner")
    ds = ds.assign_coords(
        longitude=((ds.longitude + 180) % 360) - 180,
    ).sortby("longitude")
    return ds


# ============================================================================
#  CESM2-LENS2 reader
# ============================================================================
#: CESM archive name -> the lowercase name the rest of the pipeline uses.
_CESM_RENAME = {"U": "u", "V": "v", "OMEGA": "w", "PV": "pv",
                "Z3": "z", "T": "t", "Q": "q"}
_CESM_COORDS = {"time": "valid_time", "lat": "latitude",
                "lon": "longitude", "plev": "pressure_level"}


def cesm_to_pipeline_names(ds: xr.Dataset) -> xr.Dataset:
    """CAM names/coords → the lowercase, −180…180 convention used downstream.

    Shared by the state reader and the climatology loader. The longitude
    normalisation is not cosmetic: xarray aligns on coordinate *values*, so a
    climatology left on the archive's 0…360 axis would only overlap the
    −180…180 state on its eastern half and ``pv − pv_bar`` would come back NaN
    over the Americas without raising anything.
    """
    ds = ds.rename({k: v for k, v in _CESM_RENAME.items() if k in ds.variables})
    ds = ds.rename({k: v for k, v in _CESM_COORDS.items()
                    if k in ds.variables or k in ds.dims})
    if "longitude" in ds.coords:
        ds = ds.assign_coords(
            longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
    return ds


def z_divisor(cfg) -> float:
    """What ``z`` must be divided by to get geopotential height [m].

    ERA5 ships geopotential [m²/s²], so ``H = z/g``. CESM ships ``Z3``, which
    is already height, so ``H = z``. :class:`grid.GridProfile` has recorded this
    as ``z_is_height`` since v2.13 but nothing ever read it, and the three
    ``z / G0`` sites would have made every CESM inversion height 9.8x too small
    — a base state that is wrong but finite, so the solver converges to it.
    """
    return 1.0 if getattr(cfg, "source", "era5") == "cesm" else G0


def noleap_slot_scalar(ts) -> int:
    """:func:`noleap_slot` for a single timestamp."""
    ts = pd.to_datetime(ts)
    doy = int(ts.dayofyear) - int(bool(ts.is_leap_year) and ts.month > 2)
    return (doy - 1) * 4 + int(ts.hour) // 6


def noleap_slot(valid_time: xr.DataArray) -> xr.DataArray:
    """6-hourly climatology slot for a Gregorian axis carrying noleap dates.

    ``slot = (dayofyr − 1)·4 + hour/6`` where ``dayofyr`` counts on the model's
    **noleap** calendar. The state axis has been converted to Gregorian (see
    :func:`open_cesm_years_ds`), and Gregorian day-of-year runs one ahead of
    noleap for every date after February in a leap year — 1 March 1988 is day
    61 there and day 60 in the model. Left uncorrected the climatology would be
    fetched a full day late for ten months of every fourth year, quietly, since
    a neighbouring day's climatology looks perfectly reasonable.
    """
    t = valid_time.dt
    doy = t.dayofyear - ((t.is_leap_year) & (t.month > 2)).astype(int)
    return ((doy - 1) * 4 + t.hour // 6).rename("slot")


def clim_bar(clim_ds: xr.Dataset, var: str, ds: xr.Dataset) -> xr.DataArray:
    """Climatological field sampled at each timestep of ``ds``.

    Dispatches on the climatology's own layout rather than on a config flag:
    the ERA5 climatology carries ``month``/``day``/``hour`` as dimensions and
    is addressed with vectorised ``.sel``; the CESM one is ``(slot, plev, lat,
    lon)`` and is addressed with vectorised ``.isel`` on the slot index.
    """
    if "slot" in clim_ds.dims:
        return clim_ds[var].isel(slot=noleap_slot(ds.valid_time))
    return clim_ds[var].sel(month=ds.valid_time.dt.month,
                            day=ds.valid_time.dt.day,
                            hour=ds.valid_time.dt.hour)


def year_keys_for_window(
    base_ts: pd.Timestamp, hmin: int = -49, hmax: int = 24,
) -> list[int]:
    """Years whose CESM member-year files are needed for the time window."""
    t0 = pd.to_datetime(base_ts) + pd.Timedelta(hours=hmin)
    t1 = pd.to_datetime(base_ts) + pd.Timedelta(hours=hmax)
    return list(range(int(t0.year), int(t1.year) + 1))


def open_cesm_years_ds(
    data_dir: Path, member: int, year_keys: list[int],
    engine: str = "netcdf4", chunks=None,
) -> xr.Dataset:
    """Open CESM2-LENS2 member-year files as one ERA5-shaped dataset.

    The archive differs from ERA5 in every surface detail: one file per
    (member, year) holding **all** variables rather than one per (variable,
    month); upper-case CAM names; ``time``/``lat``/``lon``/``plev`` instead of
    ``valid_time``/``latitude``/``longitude``/``pressure_level``; and a
    **noleap** calendar, which xarray decodes to ``CFTimeIndex`` rather than a
    ``DatetimeIndex``.

    Everything downstream of the reader is pandas-based (``pd.Timestamp`` /
    ``pd.Timedelta`` arithmetic, ``.dt`` accessors), so the calendar is
    converted here. That conversion is exact per timestamp — noleap dates all
    exist in the proleptic Gregorian calendar — but it leaves a **48 h gap
    across 28 February in leap years**, because the model has no 29 February
    and the converted axis therefore jumps 28 Feb 18:00 -> 1 Mar 00:00.
    ``xr.DataArray.differentiate`` uses the coordinate's own spacing, so a
    window straddling that seam would take a 6-hourly tendency over a 30 h
    interval. ``_open_window`` rejects such windows on the uniformity check
    rather than returning a quietly wrong LHS; it costs ~0.06 % of the
    catalogue (7 leap years x 10 members x ~1 day in 365).

    Args:
        data_dir: Directory of ``lens2_smbb_m{member}_{year}_plev.nc``.
        member: LENS2 member number (91-100 for the 6-hourly smbb set).
        year_keys: Years to open, from :func:`year_keys_for_window`.
        engine: NetCDF engine.
        chunks: As :func:`open_months_ds` — ``None`` loads eagerly, a dask
            chunk spec opens lazily for the low-RSS PPVI path.

    Returns:
        Dataset with ``u, v, w, pv, z, t, q`` on ``valid_time``,
        ``pressure_level``, ``latitude``, ``longitude``, longitude normalised
        to −180…180 and sorted, matching :func:`open_months_ds`.

    Raises:
        FileNotFoundError: If no member-year file exists for the window.
    """
    data_dir = Path(data_dir)
    fns = [str(data_dir / f"lens2_smbb_m{member}_{y}_plev.nc")
           for y in year_keys
           if (data_dir / f"lens2_smbb_m{member}_{y}_plev.nc").exists()]
    if not fns:
        raise FileNotFoundError(
            f"No CESM files for member {member}, years {year_keys} in {data_dir}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*separate the stored chunks.*")
        ds = xr.open_mfdataset(
            fns, combine="by_coords", parallel=False,
            chunks=chunks, engine=engine, lock=False,
        )

    ds = ds[[v for v in _CESM_RENAME if v in ds.variables]]
    ds = cesm_to_pipeline_names(ds)

    # noleap CFTimeIndex -> DatetimeIndex; see the docstring for the leap seam.
    idx = ds.indexes.get("valid_time")
    if isinstance(idx, xr.CFTimeIndex):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*non-standard calendar.*")
            try:
                new = idx.to_datetimeindex(time_unit="ns")
            except TypeError:                     # xarray < 2025.1
                new = idx.to_datetimeindex()
        ds = ds.assign_coords(valid_time=new)
    return ds


def fill_window_below_ground(ds: xr.Dataset) -> xr.Dataset:
    """Gap-fill below-ground NaN on an already-sliced time window.

    Applied to the window rather than to the whole opened file: a window is a
    handful of timesteps (~35 MB), a member-year is 1460.

    The archive stores NaN below ground on purpose, so the fill convention stays
    visible and swappable instead of being baked in. Nothing downstream
    tolerates it — one NaN destroys an entire spherical-harmonic transform, and
    the QG-omega SOR spreads it across the solve. The climatology is filled
    during the merge, so the state must be filled the same way or ``pv − pv_bar``
    becomes a difference between two conventions.

    A no-op on ERA5, whose pressure-level fields ECMWF already extrapolates
    below ground.
    """
    from .ppvi.solver import fill_below_ground_stack

    names = [v for v in ds.data_vars
             if ds[v].dims[-3:] == ("pressure_level", "latitude", "longitude")]
    if not names:
        return ds
    vals = {n: np.asarray(ds[n].values, dtype=np.float64) for n in names}
    if all(np.isfinite(v).all() for v in vals.values()):
        return ds

    plev = np.asarray(ds.pressure_level.values, dtype=float)
    for ti in range(ds.sizes["valid_time"]):
        filled = fill_below_ground_stack(
            {n: vals[n][ti] for n in names}, plev, height_key="z", temp_key="t")
        for n in names:
            vals[n][ti] = filled[n]
    for n in names:
        ds[n] = (ds[n].dims, vals[n].astype(np.float32))
    return ds


#: Hours of slack added to the ``rel_hours`` span when deciding **which files**
#: to open. The window actually read is ``rel_hours ± EDGE_PAD_H``, and the pad
#: is derived from the data's cadence — which is not known until a file is
#: open, so the file-key range cannot use the exact value. Being generous costs
#: nothing: an extra month or member-year is opened lazily and sliced away.
#:
#: Without it a narrow ``rel_hours`` silently under-opens at a file boundary.
#: With ``--dh-range 0:1:1`` the span is a single instant, so an event at
#: 31 December 18:00 asked only for that year, and the 1 January 00:00 needed
#: for the centred difference was in the next file. That cost 150 of the 176
#: failures in the first full CESM run; the same hole was latent on ERA5 at
#: month boundaries, hidden only because the default ``-49:25`` span already
#: reaches into the neighbouring month.
_FILE_KEY_PAD_H: int = 48


def open_source_ds(cfg, base_ts: pd.Timestamp, chunks=None) -> xr.Dataset:
    """Open the time window for whichever source ``cfg.source`` names."""
    hmin = int(min(cfg.rel_hours)) - _FILE_KEY_PAD_H
    hmax = int(max(cfg.rel_hours)) + _FILE_KEY_PAD_H
    if getattr(cfg, "source", "era5") == "cesm":
        if getattr(cfg, "member", None) is None:
            raise ValueError("source='cesm' needs cfg.member (LENS2 member number)")
        return open_cesm_years_ds(
            cfg.data_dir, int(cfg.member),
            year_keys_for_window(base_ts, hmin, hmax),
            engine=cfg.engine, chunks=chunks)
    return open_months_ds(
        cfg.data_dir, ["u", "v", "w", "pv", "z", "t", "q"],
        month_keys_for_window(base_ts, hmin, hmax),
        engine=cfg.engine, chunks=chunks)


# ============================================================================
#  Tracking data for Lagrangian mode
# ============================================================================
_TRACK_DF: pd.DataFrame | None = None


def _load_track_data(track_file: Path) -> pd.DataFrame:
    global _TRACK_DF
    if _TRACK_DF is None:
        _log(f"Loading tracking data from {track_file}...")
        df = pd.read_csv(
            track_file, sep=r"\s+",
            names=["track_id", "step", "time", "centlat", "centlon", "area"],
            skiprows=1,
        )
        df["time"] = df["time"].str.strip('"')
        df["time"] = pd.to_datetime(df["time"])
        _TRACK_DF = df
    return _TRACK_DF


def get_tracked_center(
    track_file: Path, track_id: int, target_time: pd.Timestamp,
) -> tuple[float | None, float | None]:
    """Look up Lagrangian centre from tracking data."""
    df = _load_track_data(track_file)
    mask = (df["track_id"] == track_id) & (df["time"] == target_time)
    matches = df[mask]
    if len(matches) == 0:
        return None, None
    row = matches.iloc[0]
    lat = float(row["centlat"])
    lon = float(row["centlon"])
    if lon > 180:
        lon -= 360
    return lat, lon


# ============================================================================
#  Geostrophic wind & gradient helpers
# ============================================================================
def _gaussian_smooth_2d(
    field_2d: np.ndarray, sigma: float = GEO_SMOOTH_SIGMA,
) -> np.ndarray:
    """NaN-tolerant Gaussian smoothing via normalised convolution."""
    mask = np.isnan(field_2d)
    filled = field_2d.copy()
    filled[mask] = 0.0
    weights = np.ones_like(field_2d)
    weights[mask] = 0.0
    s_field = gaussian_filter(filled, sigma=sigma, mode="wrap")
    s_weight = gaussian_filter(weights, sigma=sigma, mode="wrap")
    s_weight[s_weight < 1e-10] = np.nan
    return s_field / s_weight


def _grad_np_periodic_x(
    phi: np.ndarray, dx_arr: np.ndarray, dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient with periodic zonal boundary, one-sided meridional at poles."""
    nlat, nlon = phi.shape
    dphi_dx = np.empty_like(phi)
    dphi_dy = np.empty_like(phi)

    for j in range(nlat):
        dphi_dx[j, 1:-1] = (phi[j, 2:] - phi[j, :-2]) / (2 * dx_arr[j])
        dphi_dx[j, 0] = (phi[j, 1] - phi[j, -1]) / (2 * dx_arr[j])
        dphi_dx[j, -1] = (phi[j, 0] - phi[j, -2]) / (2 * dx_arr[j])

    dphi_dy[1:-1] = (phi[2:] - phi[:-2]) / (2 * dy)
    dphi_dy[0] = (phi[1] - phi[0]) / dy
    dphi_dy[-1] = (phi[-1] - phi[-2]) / dy
    return dphi_dx, dphi_dy


def _compute_geostrophic_wind(
    phi_3d: np.ndarray, lat: np.ndarray, lon: np.ndarray,
    sigma_smooth: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Geostrophic wind (u_g, v_g) from geopotential Φ on ascending lat."""
    nlev, nlat, nlon = phi_3d.shape
    lat_rad = np.deg2rad(lat)
    f_arr = 2 * OMEGA_E * np.sin(lat_rad)
    f_min = 2 * OMEGA_E * np.sin(np.deg2rad(F_MIN_LAT))
    f_arr = np.where(np.abs(f_arr) < f_min,
                     np.sign(f_arr + 1e-30) * f_min, f_arr)

    dlat = np.abs(lat[1] - lat[0]) if nlat > 1 else 1.5
    dlon = np.abs(lon[1] - lon[0]) if nlon > 1 else 1.5
    dy = np.deg2rad(dlat) * R_EARTH
    dx_arr = np.deg2rad(dlon) * R_EARTH * np.cos(lat_rad)
    dx_arr = np.maximum(dx_arr, dy * 0.01)

    u_g = np.zeros_like(phi_3d)
    v_g = np.zeros_like(phi_3d)

    for k in range(nlev):
        phi_k = phi_3d[k]
        if sigma_smooth > 0:
            phi_k = _gaussian_smooth_2d(phi_k, sigma=sigma_smooth)
        dphi_dx, dphi_dy = _grad_np_periodic_x(phi_k, dx_arr, dy)
        for j in range(nlat):
            u_g[k, j, :] = -dphi_dy[j, :] / f_arr[j]
            v_g[k, j, :] = dphi_dx[j, :] / f_arr[j]
    return u_g, v_g


# ============================================================================
#  Velocity potential solver (full NH, spherical Laplacian)
# ============================================================================
def _solve_chi_nh(
    omega_nh: np.ndarray,
    lat_nh: np.ndarray,
    lon_nh: np.ndarray,
    plevs_pa: np.ndarray,
    method: str = "spectral",
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ∇²χ = -∂ω/∂p on full NH, return u/v divergent wind.

    Default uses the spherical-harmonic Poisson inverse via
    :func:`pvtend.sh_ops.invert_laplacian_sh` with NH→global parity
    mirroring (scalar parity for χ), giving pole-closed χ at 90°N.
    Pass ``method="fd"`` to fall back to the legacy spherical-FFT
    Poisson solver.
    """
    nlev, _, _ = omega_nh.shape
    rhs = -ddp(omega_nh, plevs_pa)

    if method in ("spectral", "sh"):
        from .sh_ops import invert_laplacian_sh, gradient_sh, _SPHARM_AVAILABLE
        if not _SPHARM_AVAILABLE:
            import warnings
            warnings.warn(
                "pyspharm not installed; falling back to FD for "
                "u_div_from_omega. Install via `pip install pvtend[sh]`.",
                RuntimeWarning,
                stacklevel=2,
            )
            method = "fd"
        else:
            u_div_nh = np.zeros_like(omega_nh)
            v_div_nh = np.zeros_like(omega_nh)
            for k in range(nlev):
                chi_k = invert_laplacian_sh(
                    rhs[k], lat_nh, lon_nh, R_earth=R_EARTH, parity="scalar",
                )
                dchi_dx, dchi_dy = gradient_sh(chi_k, lat_nh, lon_nh, R_earth=R_EARTH)
                u_div_nh[k] = dchi_dx
                v_div_nh[k] = dchi_dy
            return u_div_nh, v_div_nh

    # ── Legacy FD path (preserved for regression) ──
    lat_rad = np.deg2rad(lat_nh)
    dlat = float(np.abs(np.diff(lat_nh).mean()))
    dlon = float(np.abs(np.diff(lon_nh).mean()))
    dy = np.deg2rad(dlat) * R_EARTH
    dx_arr = np.deg2rad(dlon) * R_EARTH * np.cos(lat_rad)
    dx_arr = np.maximum(dx_arr, dy * 0.1)
    dlon_rad = np.deg2rad(dlon)

    cos_phi = np.cos(lat_rad)
    area_weights = cos_phi / cos_phi.sum()
    nlon = omega_nh.shape[2]
    for k in range(nlev):
        weighted_mean = np.sum(area_weights[:, None] * rhs[k]) / nlon
        rhs[k] -= weighted_mean

    u_div_nh = np.zeros_like(omega_nh)
    v_div_nh = np.zeros_like(omega_nh)
    for k in range(nlev):
        chi_k = solve_poisson_spherical_fft(
            rhs[k], lat_nh, dy, dlon_rad, R_earth=R_EARTH
        )
        dchi_dx, dchi_dy = gradient(chi_k, dx_arr, dy)
        u_div_nh[k] = dchi_dx
        v_div_nh[k] = dchi_dy

    return u_div_nh, v_div_nh


# Backward-compatible alias
_solve_chi_moist_nh = _solve_chi_nh


# ============================================================================
#  Patch-level QG omega + moist/dry decomposition
# ============================================================================
def _qg_diabatic_adiabatic_on_patch(
    cube3d: dict[str, np.ndarray],
    lat_vec: np.ndarray,
    lon_vec: np.ndarray,
    plevs_hpa: np.ndarray,
    center_lat: float,
    qg_method: str = "log20",
    nh_data: dict | None = None,
) -> None:
    """QG omega + 4-way adiabatic/diabatic decomposition on the event patch.

    When *qg_method* is ``"log20"`` (default), performs three full SIP
    solves on the NH domain to separate vertical velocity into four
    components:

        ω_adiabatic    = QG omega (terms A+B only, no diabatic forcing)
        ω_qg_diabatic  = QG omega (A+B+C_log20) − ω_adiabatic
        ω_lhr_moist    = QG omega (A+B+C_em)     − ω_adiabatic
        ω_diabatic     = ω_total − ω_adiabatic    [total diabatic residual]

    C_log20 uses the full LOG20 J = J₁+J₂ with spherical Laplacian.
    C_em uses the Emanuel LHR formulation J_em = c_p θ̇_LHR T/θ.

    When *qg_method* is ``"sp19"`` (Steinfeld & Pfahl 2019), uses the
    empirical 1/3–2/3 scaling (no elliptic solve):

        ω_adiabatic = (1/3) ω_total
        ω_diabatic = ω_qg_diabatic = (2/3) ω_total

    For each omega component the divergent wind is recovered via
    Poisson inversion:  ∇²χ = −∂ω/∂p  →  (u_div, v_div) = ∇χ

    Modifies *cube3d* in-place, adding:
        w_adiabatic, w_diabatic, w_qg_diabatic, w_lhr_moist,
        u_div_diabatic, v_div_diabatic, u_div_adiabatic, v_div_adiabatic,
        u_div_qg_diabatic, v_div_qg_diabatic,
        u_div_lhr_moist, v_div_lhr_moist
    """
    nlevs, nlat, nlon = cube3d["z"].shape

    valid = ~np.isnan(lat_vec)
    n_valid = int(valid.sum())
    if n_valid < 3:
        zeros = np.zeros((nlevs, nlat, nlon), dtype=np.float32)
        for k in ("w_adiabatic", "w_diabatic", "w_qg_diabatic",
                  "w_lhr_moist",
                  "u_div_diabatic", "v_div_diabatic",
                  "u_div_adiabatic", "v_div_adiabatic",
                  "u_div_qg_diabatic", "v_div_qg_diabatic",
                  "u_div_lhr_moist", "v_div_lhr_moist"):
            cube3d[k] = zeros.copy()
        return

    lat_v = lat_vec[valid]
    psort = np.argsort(plevs_hpa)
    plevs_pa = plevs_hpa[psort] * 100.0

    def pick(arr3d):
        return np.nan_to_num(arr3d[psort][:, valid, :], nan=0.0)

    def unpack(arr_sv):
        out = np.zeros((nlevs, nlat, nlon), dtype=np.float32)
        for ki, si in enumerate(psort):
            out[si, valid, :] = arr_sv[ki]
        return out

    z_sv = pick(cube3d["z"])
    t_sv = pick(cube3d["t"])
    w_sv = pick(cube3d["w"])

    ug, vg = _compute_geostrophic_wind(z_sv, lat_v, lon_vec)

    # ---- SP19: empirical 1/3 dry, 2/3 moist (no elliptic solve) ----
    if qg_method == "sp19":
        from .constants import SP19_DRY_FRACTION
        cube3d["w_adiabatic"]  = SP19_DRY_FRACTION * cube3d["w"]
        cube3d["w_diabatic"] = cube3d["w"] - cube3d["w_adiabatic"]
        cube3d["w_qg_diabatic"] = cube3d["w_diabatic"].copy()

        # Poisson inversions on full NH for divergent wind recovery
        if nh_data is None:
            raise ValueError(
                "nh_data is required for divergent-wind Poisson inversion"
            )
        lat_nh = nh_data["lat"]
        lon_nh = nh_data["lon"]
        if lat_nh[0] > lat_nh[-1]:
            lat_nh_asc = lat_nh[::-1]
            flip_nh = True
        else:
            lat_nh_asc = lat_nh
            flip_nh = False

        def _prep_sp19(arr3d):
            out = arr3d[psort]
            if flip_nh:
                out = out[:, ::-1, :]
            return np.nan_to_num(out, nan=0.0)

        w_nh = _prep_sp19(nh_data["w"])
        w_adiabatic_nh = SP19_DRY_FRACTION * w_nh
        w_diabatic_nh = w_nh - w_adiabatic_nh

        udm_nh, vdm_nh = _solve_chi_nh(
            w_diabatic_nh, lat_nh_asc, lon_nh, plevs_pa)
        udd_nh, vdd_nh = _solve_chi_nh(
            w_adiabatic_nh, lat_nh_asc, lon_nh, plevs_pa)

        lat_idx = np.array([np.argmin(np.abs(lat_nh_asc - la))
                            for la in lat_v])

        def _circ_nearest_sp19(lv):
            d = np.abs((lon_nh - lv + 180) % 360 - 180)
            return int(np.argmin(d))

        lon_idx = np.array([_circ_nearest_sp19(lo) for lo in lon_vec])
        ix = np.ix_(np.arange(w_adiabatic_nh.shape[0]), lat_idx, lon_idx)

        cube3d["u_div_diabatic"]    = unpack(udm_nh[ix])
        cube3d["v_div_diabatic"]    = unpack(vdm_nh[ix])
        cube3d["u_div_qg_diabatic"] = cube3d["u_div_diabatic"].copy()
        cube3d["v_div_qg_diabatic"] = cube3d["v_div_diabatic"].copy()
        cube3d["u_div_adiabatic"]  = unpack(udd_nh[ix])
        cube3d["v_div_adiabatic"]  = unpack(vdd_nh[ix])
        return

    # ---- LOG20 (default): Full SIP solve on NH domain ----
    # nh_data is required — all solves run on the full NH domain
    if nh_data is None:
        raise ValueError(
            "nh_data is required for _qg_diabatic_adiabatic_on_patch; "
            "local-patch fallback has been removed"
        )

    # --- Full NH solves (spherical Poisson, periodic zonal BCs) ---
    lat_nh = nh_data["lat"]
    lon_nh = nh_data["lon"]
    if lat_nh[0] > lat_nh[-1]:
        lat_nh_asc = lat_nh[::-1]
        flip_nh = True
    else:
        lat_nh_asc = lat_nh
        flip_nh = False

    def _prep(arr3d):
        out = arr3d[psort]
        if flip_nh:
            out = out[:, ::-1, :]
        return np.nan_to_num(out, nan=0.0)

    z_nh = _prep(nh_data["z"])
    t_nh = _prep(nh_data["t"])
    w_nh = _prep(nh_data["w"])
    u_nh = _prep(nh_data["u"])
    v_nh = _prep(nh_data["v"])

    ug_nh, vg_nh = _compute_geostrophic_wind(z_nh, lat_nh_asc, lon_nh)

    # --- Compute local 3-D static stability for LOG20 J₂ ---
    nlev_nh = t_nh.shape[0]
    kappa_s = R_DRY / CP_DRY
    sigma_3d_nh = np.zeros_like(t_nh)
    for k in range(1, nlev_nh - 1):
        dp_s = plevs_pa[k + 1] - plevs_pa[k - 1]
        th_kp1 = t_nh[k + 1] * (1e5 / plevs_pa[k + 1]) ** kappa_s
        th_km1 = t_nh[k - 1] * (1e5 / plevs_pa[k - 1]) ** kappa_s
        dlnt = np.log(th_kp1) - np.log(th_km1)
        sigma_3d_nh[k] = -(R_DRY * t_nh[k] / plevs_pa[k]) * (dlnt / dp_s)
    sigma_3d_nh[0] = sigma_3d_nh[1]
    sigma_3d_nh[-1] = sigma_3d_nh[-2]
    sigma_3d_nh = np.maximum(sigma_3d_nh, 1e-7)

    # ── Solve 1: QG omega terms A+B → ω_dry ──
    od_nh, _ = solve_qg_omega_sip(
        ug_nh, vg_nh, t_nh,
        lat_nh_asc, lon_nh, plevs_pa,
        center_lat=center_lat,
        omega_b=w_nh,
        phi_3d=z_nh,
        bc_top=0.0, bc_bot=0.0)

    # ── Solve 2: Direct C-only QG (LOG20 full J) → ω_qg_diabatic ──
    # Exploits operator linearity: solve(C) = solve(A+B+C) - solve(A+B)
    # verified in research_questions/09_qg_moist_linearity to machine precision.
    # Zero wind → A=B=0; zero lateral BCs (omega_b=None).
    dTdt_raw = nh_data.get("t_dt")
    u_zero = np.zeros_like(u_nh)
    v_zero = np.zeros_like(v_nh)
    if dTdt_raw is not None:
        dTdt_nh = _prep(dTdt_raw)
        C_log20 = _compute_diabatic_rhs_log20(
            t_nh, dTdt_nh, u_nh, v_nh, w_nh,
            sigma_3d_nh, plevs_pa,
            lat_nh_asc, lon_nh)
        w_qg_diabatic_nh, _ = solve_qg_omega_sip(
            u_zero, v_zero, t_nh,
            lat_nh_asc, lon_nh, plevs_pa,
            center_lat=center_lat,
            omega_b=None,
            rhs_c=C_log20,
            phi_3d=z_nh,
            bc_top=0.0, bc_bot=0.0)
    else:
        w_qg_diabatic_nh = np.zeros_like(od_nh)

    # ── Solve 3: Direct C-only Emanuel (LHR) → ω_lhr_moist ──
    tdot_raw = nh_data.get("theta_dot")
    theta_raw = nh_data.get("theta")
    if tdot_raw is not None and theta_raw is not None:
        tdot_nh = _prep(tdot_raw)
        theta_nh = _prep(theta_raw)
        C_em = _compute_diabatic_rhs_emanuel(
            tdot_nh, t_nh, theta_nh,
            plevs_pa, lat_nh_asc, lon_nh)
        w_em_diabatic_nh, _ = solve_qg_omega_sip(
            u_zero, v_zero, t_nh,
            lat_nh_asc, lon_nh, plevs_pa,
            center_lat=center_lat,
            omega_b=None,
            rhs_c=C_em,
            phi_3d=z_nh,
            bc_top=0.0, bc_bot=0.0)
    else:
        w_em_diabatic_nh = np.zeros_like(od_nh)

    # Diabatic omega: ERA5 ω − ω_adiabatic (observational residual)
    w_diabatic_nh = w_nh - od_nh
    # w_qg_diabatic_nh and w_em_diabatic_nh already from direct C-only solves

    # Independent Poisson inversions on full NH (spherical Laplacian)
    udm_nh, vdm_nh = _solve_chi_nh(
        w_diabatic_nh, lat_nh_asc, lon_nh, plevs_pa)
    udd_nh, vdd_nh = _solve_chi_nh(
        od_nh, lat_nh_asc, lon_nh, plevs_pa)
    udqm_nh, vdqm_nh = _solve_chi_nh(
        w_qg_diabatic_nh, lat_nh_asc, lon_nh, plevs_pa)
    udem_nh, vdem_nh = _solve_chi_nh(
        w_em_diabatic_nh, lat_nh_asc, lon_nh, plevs_pa)

    # Extract patch from full NH solutions
    lat_idx = np.array([np.argmin(np.abs(lat_nh_asc - la))
                        for la in lat_v])

    def _circ_nearest(lv):
        d = np.abs((lon_nh - lv + 180) % 360 - 180)
        return int(np.argmin(d))

    lon_idx = np.array([_circ_nearest(lo) for lo in lon_vec])
    ix = np.ix_(np.arange(od_nh.shape[0]), lat_idx, lon_idx)

    od       = od_nh[ix]
    wqm_sv   = w_qg_diabatic_nh[ix]
    wem_sv   = w_em_diabatic_nh[ix]
    udm_sv   = udm_nh[ix]
    vdm_sv   = vdm_nh[ix]
    udd_sv   = udd_nh[ix]
    vdd_sv   = vdd_nh[ix]
    udqm_sv  = udqm_nh[ix]
    vdqm_sv  = vdqm_nh[ix]
    udem_sv  = udem_nh[ix]
    vdem_sv  = vdem_nh[ix]

    # --- Unpack & store ---
    cube3d["w_adiabatic"]            = unpack(od)
    cube3d["w_diabatic"]             = cube3d["w"] - cube3d["w_adiabatic"]
    cube3d["w_qg_diabatic"]          = unpack(wqm_sv)
    cube3d["w_lhr_moist"]            = unpack(wem_sv)
    cube3d["u_div_diabatic"]         = unpack(udm_sv)
    cube3d["v_div_diabatic"]         = unpack(vdm_sv)
    cube3d["u_div_qg_diabatic"]      = unpack(udqm_sv)
    cube3d["v_div_qg_diabatic"]      = unpack(vdqm_sv)
    cube3d["u_div_lhr_moist"]        = unpack(udem_sv)
    cube3d["v_div_lhr_moist"]        = unpack(vdem_sv)
    cube3d["u_div_adiabatic"]        = unpack(udd_sv)
    cube3d["v_div_adiabatic"]        = unpack(vdd_sv)


# ============================================================================
#  with_derivs_for_window  (the main "big array" builder)
# ============================================================================
def with_derivs_for_window(
    base_ts: pd.Timestamp,
    cfg: TendencyConfig,
    clim_ds: xr.Dataset,
) -> xr.Dataset:
    """Open data for base_ts window, compute all bars/anoms/derivatives,
    Helmholtz decomposition on full NH.

    Args:
        base_ts: Event reference timestamp.
        cfg: Tendency configuration.
        clim_ds: Pre-loaded climatology dataset.

    Returns:
        xr.Dataset with all original + derived fields on the ERA5 grid.
    """
    ds = open_source_ds(cfg, base_ts)

    # --- Restrict to the needed time window before any heavy compute ---
    # Only ``rel_hours`` around ``base_ts`` are ever written, yet the source
    # files hold a whole month (or year). Loading/deriving/Helmholtz-decomposing
    # all of it costs ~30× the memory (≈19 GB → <1 GB RSS) and compute. A
    # ±EDGE_PAD_H pad keeps the centred time-derivative (np.gradient) at the
    # extreme ``dh`` values bit-identical to the full-window result, since those
    # output steps then lie strictly inside the padded slice and use only their
    # immediate neighbours.
    #
    # The pad is derived from the data's own cadence rather than fixed. It used
    # to be a hardcoded 3 h, which silently assumed hourly ERA5: on 6-hourly
    # CESM a ``--dh-range 0:1:1`` request sliced [t−3 h, t+3 h], which contains
    # exactly ONE timestep, so the centred difference had no neighbours and the
    # tendency LHS was destroyed with nothing raised. Two steps of pad — one for
    # the gradient stencil, one of slack so an inclusive-endpoint slice cannot
    # land a needed step exactly on the boundary.
    ds = ds.sortby("valid_time")
    t_axis = np.asarray(ds.valid_time.values)
    dt_h = 1.0
    if t_axis.size > 1:
        dt_h = float(np.median(np.diff(t_axis).astype("timedelta64[m]")
                               .astype(np.float64)) / 60.0)
    EDGE_PAD_H = max(3.0, 2.0 * dt_h)
    t_lo = (pd.to_datetime(base_ts)
            + pd.Timedelta(hours=float(min(cfg.rel_hours)) - EDGE_PAD_H))
    t_hi = (pd.to_datetime(base_ts)
            + pd.Timedelta(hours=float(max(cfg.rel_hours)) + EDGE_PAD_H))
    ds = ds.sel(valid_time=slice(t_lo, t_hi))

    # A centred difference needs three points. Fewer means the window missed the
    # data (wrong month opened, event at the very start/end of the record), and
    # every tendency downstream would be silently one-sided or NaN.
    n_t = int(ds.sizes["valid_time"])
    if n_t < 3:
        raise ValueError(
            f"time window [{t_lo}, {t_hi}] around {base_ts} holds {n_t} "
            f"timestep(s) at Δt={dt_h:g} h; the centred time derivative needs "
            f"at least 3. Check that the source files cover the window."
        )

    # Every requested ``dh`` must have a neighbour on BOTH sides, or its
    # centred difference quietly degrades to a one-sided one. Checking the
    # window's endpoints rather than its length is what catches the CESM leap
    # seam: for an event at 28 Feb 18:00 the slice bound ``base + 12 h`` is
    # 29 February, a date the noleap model never wrote, so the slice simply
    # stops at 28 Feb 18:00 — three uniformly spaced steps with ``dh=0`` sitting
    # on the last one. Nothing about the returned axis looks wrong.
    t_win = np.asarray(ds.valid_time.values)
    need_lo = (pd.to_datetime(base_ts)
               + pd.Timedelta(hours=float(min(cfg.rel_hours)) - dt_h))
    need_hi = (pd.to_datetime(base_ts)
               + pd.Timedelta(hours=float(max(cfg.rel_hours)) + dt_h))
    have_lo, have_hi = pd.Timestamp(t_win[0]), pd.Timestamp(t_win[-1])
    if have_lo > need_lo or have_hi < need_hi:
        raise ValueError(
            f"time window around {base_ts} does not bracket every requested dh: "
            f"have [{have_lo}, {have_hi}], need [{need_lo}, {need_hi}] at "
            f"Δt={dt_h:g} h. Either the event sits at the edge of the record, "
            f"or (on CESM) the window runs into the 29 February that the noleap "
            f"calendar does not have."
        )

    # A gap *inside* the window is the same problem one step further in: the
    # derivative uses the coordinate's own spacing, so it would be taken over
    # the wrong interval. On CESM the converted noleap axis jumps 30 h from
    # 28 Feb 18:00 to 1 Mar 00:00, which a wide ``dh-range`` can span entirely.
    step_h = np.diff(t_win).astype("timedelta64[m]").astype(np.float64) / 60.0
    if step_h.size and not np.allclose(step_h, step_h[0], rtol=1e-6, atol=1e-6):
        bad = int(np.argmax(np.abs(step_h - np.median(step_h))))
        raise ValueError(
            f"non-uniform time axis in the window around {base_ts}: step {bad} "
            f"({pd.Timestamp(t_win[bad])} -> {pd.Timestamp(t_win[bad + 1])}) is "
            f"{step_h[bad]:g} h against a median of {np.median(step_h):g} h."
        )

    # ``differentiate`` needs at least edge_order + 1 = 2 points per dask chunk
    # along the axis it differentiates. ``open_mfdataset`` adopts the file's own
    # chunking, and the CESM archive is written (1, 9, 96, 288) — one timestep
    # per chunk — so every event failed with "Chunk size must be larger than
    # edge_order + 1". ERA5's monthly files chunk differently and never showed
    # it. The window is a handful of timesteps (~35 MB for 5), so collapsing the
    # time axis to one chunk is cheap and leaves the other dims lazy for the
    # low-RSS PPVI path.
    if ds.chunks and len(ds.chunks.get("valid_time", ())) > 1:
        ds = ds.chunk({"valid_time": -1})

    ds = fill_window_below_ground(ds)

    # --- lat metrics & Coriolis ---
    ds["latitude_rad"] = np.deg2rad(ds.latitude)
    lat_rad_vals = ds["latitude_rad"].values.copy()
    lat_rad_vals[np.abs(lat_rad_vals - np.pi / 2) < 0.01] = np.pi / 2 - 0.01
    ds["latitude_rad"] = xr.DataArray(lat_rad_vals, dims=["latitude"])
    ds["f"] = 2 * OMEGA_E * np.sin(ds["latitude_rad"])

    # --- climatology ---
    CLIM = clim_ds
    for v in ("pv", "u", "v", "w", "z", "t"):
        ds[f"{v}_bar"] = clim_bar(CLIM, v, ds)

    ds["pv_anom"] = ds["pv"] - ds["pv_bar"]
    ds["u_anom"] = ds["u"] - ds["u_bar"]
    ds["v_anom"] = ds["v"] - ds["v_bar"]
    ds["w_anom"] = ds["w"] - ds["w_bar"]

    # --- grid spacings ---
    dy_m = 2 * np.pi * R_EARTH / 360
    plev = _plev_name(ds)

    # --- Derivative helpers ---
    def _ddx_periodic_da(da, lon_name="longitude"):
        lon_vals = da[lon_name].values.astype(float)
        dlon_deg = float(np.nanmean(np.diff(lon_vals)))
        dx_m = np.deg2rad(abs(dlon_deg)) * R_EARTH * np.cos(ds["latitude_rad"])
        return (da.roll({lon_name: -1}, roll_coords=False)
                - da.roll({lon_name: 1}, roll_coords=False)) / (2.0 * dx_m)

    def _ddy_da(da, lat_name="latitude"):
        return da.differentiate(lat_name) / dy_m

    def _ddp_da(da, p_name=plev):
        return da.differentiate(p_name) / 100.0

    def _ddt_da(da, t_name="valid_time"):
        return da.differentiate(coord=t_name, datetime_unit="s")

    # --- PV derivatives ---
    ds["pv_anom_dx"] = _ddx_periodic_da(ds.pv_anom)
    ds["pv_anom_dy"] = _ddy_da(ds.pv_anom)
    ds["pv_anom_dp"] = _ddp_da(ds.pv_anom)
    ds["pv_bar_dx"] = _ddx_periodic_da(ds.pv_bar)
    ds["pv_bar_dy"] = _ddy_da(ds.pv_bar)
    ds["pv_bar_dp"] = _ddp_da(ds.pv_bar)
    ds["pv_total_dx"] = _ddx_periodic_da(ds.pv)
    ds["pv_total_dy"] = _ddy_da(ds.pv)
    ds["pv_total_dp"] = _ddp_da(ds.pv)
    ds["pv_anom_dt"] = _ddt_da(ds.pv_anom)
    ds["pv_bar_dt"] = _ddt_da(ds.pv_bar)
    ds["pv_dt"] = _ddt_da(ds.pv)

    # --- Second-order PV derivatives (full field, for 6-basis decomposition) ---
    ds["pv_total_dx_dx"] = _ddx_periodic_da(ds.pv_total_dx)
    ds["pv_total_dy_dy"] = _ddy_da(ds.pv_total_dy)
    ds["pv_total_dx_dy"] = _ddy_da(ds.pv_total_dx)

    # --- θ and Q terms (Emanuel 1987 / Tamarin & Kaspi 2016 LHR) ---
    kappa = 0.286
    L_V = 2.501e6       # latent heat of vapourisation [J/kg]
    R_V = 461.5         # gas constant for water vapour [J/(kg·K)]
    gamma_d = G0 / CP_DRY   # dry adiabatic lapse rate [K/m]

    ds["theta"] = ds["t"] * (1000.0 / ds[plev]) ** kappa
    ds["theta_dt"] = _ddt_da(ds["theta"])  # local tendency (diagnostic)
    ds["theta_dp"] = _ddp_da(ds.theta)

    # Eulerian temperature tendency (proxy for diabatic heating J/Cp)
    ds["t_dt"] = _ddt_da(ds["t"])

    # saturation vapour pressure (Bolton 1980) and specific humidity
    p_pa = ds[plev] * 100.0  # hPa → Pa
    es = 611.2 * np.exp(17.67 * (ds["t"] - 273.15) / (ds["t"] - 29.65))
    qs = 0.622 * es / (p_pa - 0.378 * es)

    # equivalent potential temperature (uses actual q from ERA5)
    ds["theta_e"] = ds["theta"] * np.exp(L_V * ds["q"] / (CP_DRY * ds["t"]))
    ds["theta_e_dp"] = _ddp_da(ds["theta_e"])

    # moist adiabatic lapse rate
    gamma_m = gamma_d * ((1.0 + L_V * qs / (R_DRY * ds["t"]))
                         / (1.0 + L_V**2 * qs / (CP_DRY * R_V * ds["t"]**2)))

    # LHR diabatic heating rate (only where ω < 0, i.e. ascending)
    theta_dot_raw = ds["w"] * (
        ds["theta_dp"]
        - (gamma_m / gamma_d) * (ds["theta"] / ds["theta_e"]) * ds["theta_e_dp"]
    )
    ds["theta_dot"] = theta_dot_raw.where(ds["w"] < 0, 0.0)
    ds["theta_dot_dp"] = _ddp_da(ds["theta_dot"])

    # relative vorticity ζ = ∂v/∂x − ∂u/∂y + u·tanφ/a  (spherical; the +u·tanφ/a curvature/metric
    # term is REQUIRED — without it ζ is the flat-earth form and (f+ζ) in the Q source below carries a
    # ~u·tanφ/a error, ~5% of f at 45° for u~30 m/s, growing poleward. Fixed 2026-06-23 per the cos-lat
    # metric audit; see ~/.github/copilot-instructions.md §14.2.)
    ds["v_dx"] = _ddx_periodic_da(ds.v)
    ds["u_dy"] = _ddy_da(ds.u)
    ds["zeta"] = ds["v_dx"] - ds["u_dy"] + ds["u"] * (np.tan(ds["latitude_rad"]) / R_EARTH)

    # Q = −g(f + ζ) ∂θ̇_LHR/∂p  (vertical stretching only)
    ds["Q"] = -G0 * (ds["f"] + ds["zeta"]) * ds["theta_dot_dp"]

    ds = ds.assign_coords(
        longitude=((ds.longitude + 180) % 360) - 180,
    ).sortby("longitude")

    # ================================================================
    #  FFT Helmholtz on the TOTAL NH wind field (v2.0)
    # ================================================================
    lat_nh = ds.latitude.values
    lon_nh = ds.longitude.values
    if lat_nh[0] > lat_nh[-1]:
        lat_asc = lat_nh[::-1]
        flip_lat = True
    else:
        lat_asc = lat_nh
        flip_lat = False

    ntimes = ds.sizes["valid_time"]
    nlevs = ds.sizes[plev]
    nlat_nh = ds.sizes["latitude"]
    nlon_nh = ds.sizes["longitude"]

    shape_4d = (ntimes, nlevs, nlat_nh, nlon_nh)
    u_rot_all = np.zeros(shape_4d, dtype=np.float32)
    u_div_all = np.zeros(shape_4d, dtype=np.float32)
    u_har_all = np.zeros(shape_4d, dtype=np.float32)
    v_rot_all = np.zeros(shape_4d, dtype=np.float32)
    v_div_all = np.zeros(shape_4d, dtype=np.float32)
    v_har_all = np.zeros(shape_4d, dtype=np.float32)

    # Helmholtz on total (u, v) — not (u_anom, v_anom)
    u_total_vals = ds["u"].values
    v_total_vals = ds["v"].values

    for ti in range(ntimes):
        for li in range(nlevs):
            u2d = u_total_vals[ti, li]
            v2d = v_total_vals[ti, li]
            if flip_lat:
                u2d = u2d[::-1]
                v2d = v2d[::-1]
            # method="spectral" → SH inversion via pyspharm/windspharm with
            # NH→global parity mirroring (u even, v odd), giving pole-closed
            # ψ/χ at 90 °N.  Falls back to the conservative-form spherical-FFT
            # solver if pyspharm is not installed (warning emitted).
            helm = helmholtz_decomposition(
                u2d, v2d, lat_asc, lon_nh,
                R_earth=R_EARTH, method="spectral")
            if flip_lat:
                for key in ("u_rot", "u_div", "u_har",
                            "v_rot", "v_div", "v_har"):
                    helm[key] = helm[key][::-1]
            u_rot_all[ti, li] = helm["u_rot"]
            u_div_all[ti, li] = helm["u_div"]
            u_har_all[ti, li] = helm["u_har"]
            v_rot_all[ti, li] = helm["v_rot"]
            v_div_all[ti, li] = helm["v_div"]
            v_har_all[ti, li] = helm["v_har"]

    _log(f"  FFT-NH Helmholtz (total wind) done: {ntimes} times × {nlevs} levels")

    # ── Store total Helmholtz fields ──
    dims4d = ("valid_time", plev, "latitude", "longitude")
    coords4d = ds["u"].coords
    for name, arr in [
        ("u_rot", u_rot_all), ("u_div", u_div_all),
        ("v_rot", v_rot_all), ("v_div", v_div_all),
    ]:
        ds[name] = xr.DataArray(arr, dims=dims4d, coords=coords4d)

    # ================================================================
    #  Load climatological Helmholtz & compute anomaly by subtraction
    # ================================================================
    HELM_BARS = ("u_rot_bar", "u_div_bar", "v_rot_bar", "v_div_bar")

    if all(b in clim_ds.variables for b in HELM_BARS):
        # CESM: the decomposition is folded into the climatology file itself.
        # It is linear, so decomposing the mean wind is identical to averaging
        # the decompositions, and doing it during the merge cost ~4 min against
        # a second pass over the 11 GB file plus a layout conversion. Addressed
        # by slot, exactly like every other bar.
        for name in HELM_BARS:
            ds[name] = clim_bar(clim_ds, name, ds)
    else:
        # ERA5: 24 separate per-month files, indexed (nday, 24, nlev, ny, nx).
        months_needed = sorted(set(ds.valid_time.dt.month.values.tolist()))
        clim_helm_cache: dict[int, dict[str, np.ndarray]] = {}
        for m in months_needed:
            clim_helm_cache[m] = load_helmholtz_climatology(
                cfg.clim_helmholtz_dir, m)

        bars_4d = {name: np.zeros(shape_4d, dtype=np.float32)
                   for name in HELM_BARS}
        times = pd.to_datetime(ds.valid_time.values)
        for ti, t in enumerate(times):
            ch = clim_helm_cache[t.month]
            di, hr = t.day - 1, t.hour        # 0-based day, 0-23 hour
            if di < ch["u_rot_bar"].shape[0] and hr < ch["u_rot_bar"].shape[1]:
                for name in HELM_BARS:
                    bars_4d[name][ti] = ch[name][di, hr]

        for name in HELM_BARS:
            ds[name] = xr.DataArray(bars_4d[name], dims=dims4d, coords=coords4d)

    # ── Anomaly Helmholtz by subtraction: u'_rot = u_rot − ū_rot ──
    ds["u_rot_anom"] = ds["u_rot"] - ds["u_rot_bar"]
    ds["u_div_anom"] = ds["u_div"] - ds["u_div_bar"]
    ds["u_har_anom"] = xr.DataArray(u_har_all, dims=dims4d, coords=coords4d)
    ds["v_rot_anom"] = ds["v_rot"] - ds["v_rot_bar"]
    ds["v_div_anom"] = ds["v_div"] - ds["v_div_bar"]
    ds["v_har_anom"] = xr.DataArray(v_har_all, dims=dims4d, coords=coords4d)

    _log("  Climatological Helmholtz loaded; anomaly Helmholtz computed by subtraction")

    return ds


# ============================================================================
#  Grid / patching utilities
# ============================================================================
class _GridInfo:
    """Container for grid metadata (cached per-worker)."""

    def __init__(self, lat, lon, lat_half, lon_half):
        dlat = float(abs(np.diff(lat).mean()))
        dlon = float(abs(np.diff(lon).mean()))
        self.lat = lat
        self.lon = lon
        self.LAT_PAD = int(round(lat_half / dlat))
        self.LON_PAD = int(round(lon_half / dlon))
        rlat = np.linspace(-lat_half, lat_half, 2 * self.LAT_PAD + 1)
        rlon = np.linspace(-lon_half, lon_half, 2 * self.LON_PAD + 1)
        self.Y_rel, self.X_rel = np.meshgrid(rlat, rlon, indexing="ij")
        self.lat_desc = bool(np.all(np.diff(lat) < 0))


def _nearest_idx(lat0, lon0, grid):
    ilat = int(np.abs(grid.lat - lat0).argmin())
    ilon = int(np.abs(grid.lon - lon0).argmin())
    ok = (ilat >= grid.LAT_PAD) and (ilat + grid.LAT_PAD < len(grid.lat))
    return ilat, ilon, ok


def _wrapped_lon_index(ilon, *, LON_PAD, nlon):
    start = ilon - LON_PAD
    return (np.arange(0, 2 * LON_PAD + 1) + start) % nlon


def _patch_lon1d(ds, ilon, grid):
    nlon = ds.sizes["longitude"]
    idx = _wrapped_lon_index(ilon, LON_PAD=grid.LON_PAD, nlon=nlon)
    lon_seg = ds.longitude.values[idx]
    return np.rad2deg(np.unwrap(np.deg2rad(lon_seg)))


def _patch_lat1d_full(ds, ilat, grid, eff_north, eff_south):
    nlat = ds.sizes["latitude"]
    full = 2 * grid.LAT_PAD + 1
    out = np.full((full,), np.nan, dtype=float)
    if grid.lat_desc:
        i0 = max(0, ilat - eff_north)
        i1 = min(nlat, ilat + eff_south + 1)
    else:
        i0 = max(0, ilat - eff_south)
        i1 = min(nlat, ilat + eff_north + 1)
    seg = ds.latitude.isel(latitude=slice(i0, i1)).values
    if grid.lat_desc:
        seg = seg[::-1]
    y_eff = seg.shape[0]
    y0 = grid.LAT_PAD - eff_south
    out[y0:y0 + y_eff] = seg
    return out


def _extract_cube_with_pads3d(ds, varnames, ts, ilat, ilon, levels, grid,
                              eff_north, eff_south):
    plev = _plev_name(ds)
    nlon = ds.sizes["longitude"]
    lon_idx = xr.DataArray(
        _wrapped_lon_index(ilon, LON_PAD=grid.LON_PAD, nlon=nlon),
        dims=("x",))
    ts_sel = ds.sel(valid_time=ts)
    nlat = ds.sizes["latitude"]
    if grid.lat_desc:
        i0 = max(0, ilat - eff_north)
        i1 = min(nlat, ilat + eff_south + 1)
    else:
        i0 = max(0, ilat - eff_south)
        i1 = min(nlat, ilat + eff_north + 1)
    parts = []
    for v in varnames:
        da = (ts_sel[v]
              .sel({plev: levels})
              .isel(latitude=slice(i0, i1))
              .isel(longitude=lon_idx))
        if grid.lat_desc:
            da = da.isel(latitude=slice(None, None, -1))
        parts.append(da)
    stacked = xr.concat(parts, dim="__var__").compute()
    Y_full = 2 * grid.LAT_PAD + 1
    Y_eff = stacked.sizes["latitude"]
    X = stacked.sizes["x"]
    L = stacked.sizes[plev]
    y0 = grid.LAT_PAD - eff_south
    out = {}
    for i, v in enumerate(varnames):
        arr = stacked.isel(__var__=i).values
        buf = np.full((L, Y_full, X), np.nan, dtype=arr.dtype)
        buf[:, y0:y0 + Y_eff, :] = arr
        out[v] = buf
    return out


# ============================================================================
#  TendencyComputer class
# ============================================================================
class TendencyComputer:
    """Computes PV tendency terms for weather events.

    Parameterized by :class:`TendencyConfig` — works for both blocking
    and PRP event types without code duplication.

    Example::

        cfg = TendencyConfig(
            event_type="blocking",
            csv_path=Path("events_blocking.csv"),
        )
        tc = TendencyComputer(cfg)
        n = tc.process_event("onset", track_id=42, lat0=55.0,
                             lon0=-30.0, base_ts=pd.Timestamp("2010-01-15"))
        print(f"Wrote {n} NPZ files.")
    """

    def __init__(self, config: TendencyConfig) -> None:
        self.cfg = config
        self._clim: xr.Dataset | None = None
        self._grid: _GridInfo | None = None

    def _get_clim(self, chunks=None) -> xr.Dataset:
        # ``chunks`` is honoured only on the first (cache-filling) call. PPVI
        # workers pass {"day":1,"hour":1} so per-event clim slices are released
        # rather than retained (see load_climatology); the compute path keeps
        # the default eager layout.
        if self._clim is None:
            self._clim = load_climatology(
                self.cfg.clim_path, self.cfg.engine, chunks=chunks)
        return self._clim

    def _init_grid(self, ds: xr.Dataset) -> _GridInfo:
        if self._grid is None:
            lat = ds.latitude.values
            lon = ds.longitude.values
            self._grid = _GridInfo(lat, lon, self.cfg.lat_half, self.cfg.lon_half)
        return self._grid

    # ── public API ─────────────────────────────────────────────────

    def process_event(
        self,
        evt_name: str,
        track_id: int,
        lat0: float,
        lon0: float,
        base_ts: pd.Timestamp,
        *,
        also_ppvi: bool = False,
        inv_lon_half: float = 90.0,
    ) -> int:
        """Process a single event and write NPZ files.

        Loads data **once** via :func:`with_derivs_for_window`, then
        iterates over ``cfg.rel_hours``, extracting a patch and computing
        QG omega + moist/dry + cross-terms + wavg per timestep.

        When ``also_ppvi=True`` the Wu piecewise PV-inversion rotational
        winds are computed for each ``dh`` and written **together** with
        the base tendency fields in a single NPZ write (used for fresh
        runs where no base NPZ exists yet — no append/re-write).

        Returns the number of NPZ files written.
        """
        written = 0
        _log(f"\n--- Processing event: {track_id} "
             f"at ({lat0}, {lon0}) ---")

        clim_ds = self._get_clim()

        # ── Event-level skip: all NPZs already exist? ──
        if self.cfg.skip_existing:
            all_exist = all(
                self._out_path(evt_name, dh, track_id,
                               base_ts + pd.Timedelta(hours=dh)).exists()
                for dh in self.cfg.rel_hours
            )
            if all_exist:
                _log(f"-> Event {track_id}: all {len(self.cfg.rel_hours)} "
                     f"NPZ(s) exist, skipping.")
                return 0

        ds = with_derivs_for_window(base_ts, self.cfg, clim_ds)
        grid = self._init_grid(ds)

        ilat, ilon, ok = _nearest_idx(lat0, lon0, grid)
        _log(f"Nearest grid index: ilat={ilat}, ilon={ilon}, ok={ok}")

        nlat = len(grid.lat)
        if grid.lat_desc:
            eff_north = min(grid.LAT_PAD, ilat)
            eff_south = min(grid.LAT_PAD, nlat - 1 - ilat)
        else:
            eff_south = min(grid.LAT_PAD, ilat)
            eff_north = min(grid.LAT_PAD, nlat - 1 - ilat)

        if not self.cfg.partial_at_pole and (
            eff_north < grid.LAT_PAD or eff_south < grid.LAT_PAD
        ):
            _log("-> Event skipped: Too close to boundary.")
            return 0
        if eff_north <= 0 and eff_south <= 0:
            _log("-> Event skipped: zero latitude rows.")
            return 0

        dt_index = pd.to_datetime(ds.valid_time.values)
        plev = _plev_name(ds)
        plevs_hpa = ds[plev].values
        levels = self.cfg.levels
        wavg_idx = [levels.index(l) for l in self.cfg.wavg_levels
                    if l in levels]

        # Static PPVI inversion geometry (only when writing fields together).
        ppvi_geom = None
        cplev = None
        if also_ppvi:
            ppvi_geom = self._ppvi_geom(ds, inv_lon_half)
            cplev = _plev_name(clim_ds)

        for dh in self.cfg.rel_hours:
            ts = base_ts + pd.Timedelta(hours=dh)
            if ts not in dt_index:
                continue

            out_fp = self._out_path(evt_name, dh, track_id, ts)
            if self.cfg.skip_existing and out_fp.exists():
                written += 1
                continue

            # Lagrangian centre
            if self.cfg.center_mode == "lagrangian":
                tracked_lat, tracked_lon = get_tracked_center(
                    self.cfg.track_file, track_id, ts)
                if tracked_lat is None:
                    current_lat, current_lon = lat0, lon0
                else:
                    current_lat, current_lon = tracked_lat, tracked_lon
            else:
                current_lat, current_lon = lat0, lon0

            ilat_c, ilon_c, ok_c = _nearest_idx(
                current_lat, current_lon, grid)
            if not ok_c and not self.cfg.partial_at_pole:
                continue

            if grid.lat_desc:
                en_c = min(grid.LAT_PAD, ilat_c)
                es_c = min(grid.LAT_PAD, nlat - 1 - ilat_c)
            else:
                es_c = min(grid.LAT_PAD, ilat_c)
                en_c = min(grid.LAT_PAD, nlat - 1 - ilat_c)
            if en_c <= 0 and es_c <= 0:
                continue

            lat_vec_full = _patch_lat1d_full(ds, ilat_c, grid, en_c, es_c)
            lon_unwrapped = _patch_lon1d(ds, ilon_c, grid)

            cube3d = _extract_cube_with_pads3d(
                ds, _EXTRACT_VARS, ts, ilat_c, ilon_c,
                levels, grid, en_c, es_c)

            # --- Patch-level QG omega + moist/dry ---
            nh_data = None
            if self.cfg.qg_omega_method == "log20":
                snap = ds.sel(valid_time=ts)
                nh_data = {
                    "z": snap["z"].values,
                    "t": snap["t"].values,
                    "w": snap["w"].values,
                    "t_dt": snap["t_dt"].values,
                    "u": snap["u"].values,
                    "v": snap["v"].values,
                    "theta_dot": snap["theta_dot"].values,
                    "theta": snap["theta"].values,
                    "lat": ds.latitude.values,
                    "lon": ds.longitude.values,
                }
            _qg_diabatic_adiabatic_on_patch(
                cube3d, lat_vec_full, lon_unwrapped,
                plevs_hpa, center_lat=current_lat,
                qg_method=self.cfg.qg_omega_method,
                nh_data=nh_data)

            # NaN safety: 0-fill these solver-output fields ONLY as input sanitization before the
            # nansum-based vertical mean `vwm` below — never 0-fill before a statistical mean
            # (see Rule 14). On the validated ERA5 path the wavg levels (300/250/200) are NaN-free,
            # so this is a no-op; dropping it to rely on nansum's exclude-NaN semantics is deferred
            # to v2.14 (needs the full test/CI checklist on the published package).
            for _key in ("w_adiabatic", "w_diabatic", "w_qg_diabatic",
                         "w_lhr_moist",
                         "u_div_diabatic", "v_div_diabatic",
                         "u_div_adiabatic", "v_div_adiabatic",
                         "u_div_qg_diabatic", "v_div_qg_diabatic",
                         "u_div_lhr_moist", "v_div_lhr_moist",
                         "q", "t_dt"):
                if _key in cube3d:
                    cube3d[_key] = np.nan_to_num(
                        cube3d[_key], nan=0.0, posinf=0.0, neginf=0.0)

            z_m_3d = cube3d["z"] / z_divisor(self.cfg)

            def vwm(arrL, *, z_m_3d=z_m_3d, wavg_idx=wavg_idx):
                arr_w = arrL[wavg_idx]
                z_w = z_m_3d[wavg_idx]
                wt = np.exp(-z_w / H_SCALE)
                num = np.nansum(arr_w * wt, axis=0)
                den = np.nansum(wt, axis=0)
                out = np.full_like(num, np.nan)
                mask = den > 0
                out[mask] = num[mask] / den[mask]
                return out

            vw = partial(vwm, z_m_3d=z_m_3d, wavg_idx=wavg_idx)

            # ────────────────────────────────────────────
            #  3-D cross terms (53-term v2.0 catalog)
            # ────────────────────────────────────────────
            # ── 12 base (bar/anom × bar/anom) ──
            uanom_pvbar_dx_3d = cube3d["u_anom"] * cube3d["pv_bar_dx"]
            uanom_pvanom_dx_3d = cube3d["u_anom"] * cube3d["pv_anom_dx"]
            ubar_pvanom_dx_3d = cube3d["u_bar"] * cube3d["pv_anom_dx"]
            ubar_pvbar_dx_3d = cube3d["u_bar"] * cube3d["pv_bar_dx"]

            vanom_pvbar_dy_3d = cube3d["v_anom"] * cube3d["pv_bar_dy"]
            vanom_pvanom_dy_3d = cube3d["v_anom"] * cube3d["pv_anom_dy"]
            vbar_pvanom_dy_3d = cube3d["v_bar"] * cube3d["pv_anom_dy"]
            vbar_pvbar_dy_3d = cube3d["v_bar"] * cube3d["pv_bar_dy"]

            wanom_pvbar_dp_3d = cube3d["w_anom"] * cube3d["pv_bar_dp"]
            wanom_pvanom_dp_3d = cube3d["w_anom"] * cube3d["pv_anom_dp"]
            wbar_pvanom_dp_3d = cube3d["w_bar"] * cube3d["pv_anom_dp"]
            wbar_pvbar_dp_3d = cube3d["w_bar"] * cube3d["pv_bar_dp"]

            # ── 16 Helmholtz primary (anom + bar rot/div) ──
            urot_anom_pvbar_dx_3d = cube3d["u_rot_anom"] * cube3d["pv_bar_dx"]
            urot_anom_pvanom_dx_3d = cube3d["u_rot_anom"] * cube3d["pv_anom_dx"]
            udiv_anom_pvbar_dx_3d = cube3d["u_div_anom"] * cube3d["pv_bar_dx"]
            udiv_anom_pvanom_dx_3d = cube3d["u_div_anom"] * cube3d["pv_anom_dx"]
            urot_bar_pvbar_dx_3d = cube3d["u_rot_bar"] * cube3d["pv_bar_dx"]
            urot_bar_pvanom_dx_3d = cube3d["u_rot_bar"] * cube3d["pv_anom_dx"]
            udiv_bar_pvbar_dx_3d = cube3d["u_div_bar"] * cube3d["pv_bar_dx"]
            udiv_bar_pvanom_dx_3d = cube3d["u_div_bar"] * cube3d["pv_anom_dx"]

            vrot_anom_pvbar_dy_3d = cube3d["v_rot_anom"] * cube3d["pv_bar_dy"]
            vrot_anom_pvanom_dy_3d = cube3d["v_rot_anom"] * cube3d["pv_anom_dy"]
            vdiv_anom_pvbar_dy_3d = cube3d["v_div_anom"] * cube3d["pv_bar_dy"]
            vdiv_anom_pvanom_dy_3d = cube3d["v_div_anom"] * cube3d["pv_anom_dy"]
            vrot_bar_pvbar_dy_3d = cube3d["v_rot_bar"] * cube3d["pv_bar_dy"]
            vrot_bar_pvanom_dy_3d = cube3d["v_rot_bar"] * cube3d["pv_anom_dy"]
            vdiv_bar_pvbar_dy_3d = cube3d["v_div_bar"] * cube3d["pv_bar_dy"]
            vdiv_bar_pvanom_dy_3d = cube3d["v_div_bar"] * cube3d["pv_anom_dy"]

            # ── 16 divergent adiabatic/diabatic horizontal ──
            udm_pvbar_dx_3d = cube3d["u_div_diabatic"] * cube3d["pv_bar_dx"]
            udm_pvanom_dx_3d = cube3d["u_div_diabatic"] * cube3d["pv_anom_dx"]
            udd_pvbar_dx_3d = cube3d["u_div_adiabatic"] * cube3d["pv_bar_dx"]
            udd_pvanom_dx_3d = cube3d["u_div_adiabatic"] * cube3d["pv_anom_dx"]

            vdm_pvbar_dy_3d = cube3d["v_div_diabatic"] * cube3d["pv_bar_dy"]
            vdm_pvanom_dy_3d = cube3d["v_div_diabatic"] * cube3d["pv_anom_dy"]
            vdd_pvbar_dy_3d = cube3d["v_div_adiabatic"] * cube3d["pv_bar_dy"]
            vdd_pvanom_dy_3d = cube3d["v_div_adiabatic"] * cube3d["pv_anom_dy"]

            udqm_pvbar_dx_3d = cube3d["u_div_qg_diabatic"] * cube3d["pv_bar_dx"]
            udqm_pvanom_dx_3d = cube3d["u_div_qg_diabatic"] * cube3d["pv_anom_dx"]
            vdqm_pvbar_dy_3d = cube3d["v_div_qg_diabatic"] * cube3d["pv_bar_dy"]
            vdqm_pvanom_dy_3d = cube3d["v_div_qg_diabatic"] * cube3d["pv_anom_dy"]

            udem_pvbar_dx_3d = cube3d["u_div_lhr_moist"] * cube3d["pv_bar_dx"]
            udem_pvanom_dx_3d = cube3d["u_div_lhr_moist"] * cube3d["pv_anom_dx"]
            vdem_pvbar_dy_3d = cube3d["v_div_lhr_moist"] * cube3d["pv_bar_dy"]
            vdem_pvanom_dy_3d = cube3d["v_div_lhr_moist"] * cube3d["pv_anom_dy"]

            # ── 8 alt vertical (adiabatic/diabatic/qg/lhr omega) ──
            w_dry_pvbar_dp_3d = cube3d["w_adiabatic"] * cube3d["pv_bar_dp"]
            w_dry_pvanom_dp_3d = cube3d["w_adiabatic"] * cube3d["pv_anom_dp"]
            w_moist_pvbar_dp_3d = cube3d["w_diabatic"] * cube3d["pv_bar_dp"]
            w_moist_pvanom_dp_3d = cube3d["w_diabatic"] * cube3d["pv_anom_dp"]
            w_qgm_pvbar_dp_3d = cube3d["w_qg_diabatic"] * cube3d["pv_bar_dp"]
            w_qgm_pvanom_dp_3d = cube3d["w_qg_diabatic"] * cube3d["pv_anom_dp"]
            w_em_pvbar_dp_3d = cube3d["w_lhr_moist"] * cube3d["pv_bar_dp"]
            w_em_pvanom_dp_3d = cube3d["w_lhr_moist"] * cube3d["pv_anom_dp"]

            # ── 1 diabatic (Q_LHR) — already in cube3d["Q"] ──

            # ────────────────────────────────────────────
            #  Write NPZ (atomic via tempfile)
            # ────────────────────────────────────────────
            out_fp.parent.mkdir(parents=True, exist_ok=True)
            _log(f"  Writing {out_fp} ...")
            # 300-hPa level index for QG-omega blowup watch (Phase 7 v2)
            _lvl300 = int(np.argmin(np.abs(np.asarray(levels) - 300)))
            record = dict(
                    # ── Metadata ──
                    Y_rel=grid.Y_rel, X_rel=grid.X_rel,
                    levels=np.array(levels, dtype=np.int32),
                    wavg_levels=np.array(self.cfg.wavg_levels, dtype=np.int32),
                    H_SCALE=float(H_SCALE), G0=float(G0),
                    lat_vec=lat_vec_full.astype(float),
                    lon_vec_unwrapped=lon_unwrapped.astype(float),
                    # Stored as-is: CESM ids are strings ('m091_t00002')
                    # because track_id restarts per LENS2 member. np.savez
                    # keeps a str as a 0-d unicode array, so readers get
                    # str(npz['track_id']) either way.
                    track_id=track_id,
                    lat0=float(lat0), lon0=float(lon0),
                    center_lat=float(current_lat),
                    center_lon=float(current_lon),
                    center_mode=self.cfg.center_mode,
                    ts=str(ts), dh=int(dh),

                    # ── 2-D wavg fields ──
                    pv_dt=vw(cube3d["pv_dt"]),
                    pv=vw(cube3d["pv"]),
                    z=vw(z_m_3d),
                    u=vw(cube3d["u"]), v=vw(cube3d["v"]),
                    w=vw(cube3d["w"]),
                    pv_dx=vw(cube3d["pv_total_dx"]),
                    pv_dy=vw(cube3d["pv_total_dy"]),
                    pv_dp=vw(cube3d["pv_total_dp"]),
                    pv_dx_dx=vw(cube3d["pv_total_dx_dx"]),
                    pv_dy_dy=vw(cube3d["pv_total_dy_dy"]),
                    pv_dx_dy=vw(cube3d["pv_total_dx_dy"]),
                    u_bar=vw(cube3d["u_bar"]),
                    v_bar=vw(cube3d["v_bar"]),
                    w_bar=vw(cube3d["w_bar"]),
                    pv_bar=vw(cube3d["pv_bar"]),
                    u_anom=vw(cube3d["u_anom"]),
                    v_anom=vw(cube3d["v_anom"]),
                    w_anom=vw(cube3d["w_anom"]),
                    pv_anom=vw(cube3d["pv_anom"]),
                    pv_bar_dx=vw(cube3d["pv_bar_dx"]),
                    pv_bar_dy=vw(cube3d["pv_bar_dy"]),
                    pv_bar_dp=vw(cube3d["pv_bar_dp"]),
                    pv_bar_dt=vw(cube3d["pv_bar_dt"]),
                    pv_anom_dx=vw(cube3d["pv_anom_dx"]),
                    pv_anom_dy=vw(cube3d["pv_anom_dy"]),
                    pv_anom_dp=vw(cube3d["pv_anom_dp"]),
                    pv_anom_dt=vw(cube3d["pv_anom_dt"]),
                    t=vw(cube3d["t"]),
                    theta=vw(cube3d["theta"]),
                    theta_dt=vw(cube3d["theta_dt"]),
                    theta_dot=vw(cube3d["theta_dot"]),
                    Q=vw(cube3d["Q"]),
                    # Total Helmholtz (v2.0)
                    u_rot=vw(cube3d["u_rot"]),
                    u_div=vw(cube3d["u_div"]),
                    v_rot=vw(cube3d["v_rot"]),
                    v_div=vw(cube3d["v_div"]),
                    # Climatological Helmholtz (v2.0)
                    u_rot_bar=vw(cube3d["u_rot_bar"]),
                    u_div_bar=vw(cube3d["u_div_bar"]),
                    v_rot_bar=vw(cube3d["v_rot_bar"]),
                    v_div_bar=vw(cube3d["v_div_bar"]),
                    # Anomaly Helmholtz
                    u_rot_anom=vw(cube3d["u_rot_anom"]),
                    u_div_anom=vw(cube3d["u_div_anom"]),
                    u_har_anom=vw(cube3d["u_har_anom"]),
                    v_rot_anom=vw(cube3d["v_rot_anom"]),
                    v_div_anom=vw(cube3d["v_div_anom"]),
                    v_har_anom=vw(cube3d["v_har_anom"]),
                    u_div_diabatic=vw(cube3d["u_div_diabatic"]),
                    v_div_diabatic=vw(cube3d["v_div_diabatic"]),
                    u_div_adiabatic=vw(cube3d["u_div_adiabatic"]),
                    v_div_adiabatic=vw(cube3d["v_div_adiabatic"]),
                    w_adiabatic=vw(cube3d["w_adiabatic"]),
                    w_diabatic=vw(cube3d["w_diabatic"]),
                    w_qg_diabatic=vw(cube3d["w_qg_diabatic"]),
                    u_div_qg_diabatic=vw(cube3d["u_div_qg_diabatic"]),
                    v_div_qg_diabatic=vw(cube3d["v_div_qg_diabatic"]),
                    w_lhr_moist=vw(cube3d["w_lhr_moist"]),
                    # ── QG-omega solver blowup watch (Phase 7 v2) ──
                    # Patch max |ω| at 300 hPa for each solver-derived ω.
                    # Empirical raw-ERA5 envelope at 300 hPa over
                    # 1990-2020 hourly: max=22.4 Pa/s, 99.9th=19.9 Pa/s.
                    # We flag QG/Emanuel solver output > 25 Pa/s as
                    # solver pathology (raw ω essentially never exceeds
                    # this; 25 Pa/s = raw_max + ~10 % headroom).
                    max_abs_w_adiabatic_300=np.float32(
                        np.nanmax(np.abs(cube3d["w_adiabatic"][_lvl300]))),
                    max_abs_w_diabatic_300=np.float32(
                        np.nanmax(np.abs(cube3d["w_diabatic"][_lvl300]))),
                    max_abs_w_qg_diabatic_300=np.float32(
                        np.nanmax(np.abs(cube3d["w_qg_diabatic"][_lvl300]))),
                    max_abs_w_lhr_moist_300=np.float32(
                        np.nanmax(np.abs(cube3d["w_lhr_moist"][_lvl300]))),
                    u_div_lhr_moist=vw(cube3d["u_div_lhr_moist"]),
                    v_div_lhr_moist=vw(cube3d["v_div_lhr_moist"]),
                    q=vw(cube3d["q"]),
                    t_dt=vw(cube3d["t_dt"]),

                    # ── 2-D cross terms (53-term v2.0 catalog) ──
                    # 12 base
                    u_anom_pv_bar_dx=vw(uanom_pvbar_dx_3d),
                    u_anom_pv_anom_dx=vw(uanom_pvanom_dx_3d),
                    u_bar_pv_anom_dx=vw(ubar_pvanom_dx_3d),
                    u_bar_pv_bar_dx=vw(ubar_pvbar_dx_3d),
                    v_anom_pv_bar_dy=vw(vanom_pvbar_dy_3d),
                    v_anom_pv_anom_dy=vw(vanom_pvanom_dy_3d),
                    v_bar_pv_anom_dy=vw(vbar_pvanom_dy_3d),
                    v_bar_pv_bar_dy=vw(vbar_pvbar_dy_3d),
                    w_anom_pv_bar_dp=vw(wanom_pvbar_dp_3d),
                    w_anom_pv_anom_dp=vw(wanom_pvanom_dp_3d),
                    w_bar_pv_anom_dp=vw(wbar_pvanom_dp_3d),
                    w_bar_pv_bar_dp=vw(wbar_pvbar_dp_3d),
                    # 16 Helmholtz (anom + bar rot/div)
                    u_rot_anom_pv_bar_dx=vw(urot_anom_pvbar_dx_3d),
                    u_rot_anom_pv_anom_dx=vw(urot_anom_pvanom_dx_3d),
                    u_div_anom_pv_bar_dx=vw(udiv_anom_pvbar_dx_3d),
                    u_div_anom_pv_anom_dx=vw(udiv_anom_pvanom_dx_3d),
                    u_rot_bar_pv_bar_dx=vw(urot_bar_pvbar_dx_3d),
                    u_rot_bar_pv_anom_dx=vw(urot_bar_pvanom_dx_3d),
                    u_div_bar_pv_bar_dx=vw(udiv_bar_pvbar_dx_3d),
                    u_div_bar_pv_anom_dx=vw(udiv_bar_pvanom_dx_3d),
                    v_rot_anom_pv_bar_dy=vw(vrot_anom_pvbar_dy_3d),
                    v_rot_anom_pv_anom_dy=vw(vrot_anom_pvanom_dy_3d),
                    v_div_anom_pv_bar_dy=vw(vdiv_anom_pvbar_dy_3d),
                    v_div_anom_pv_anom_dy=vw(vdiv_anom_pvanom_dy_3d),
                    v_rot_bar_pv_bar_dy=vw(vrot_bar_pvbar_dy_3d),
                    v_rot_bar_pv_anom_dy=vw(vrot_bar_pvanom_dy_3d),
                    v_div_bar_pv_bar_dy=vw(vdiv_bar_pvbar_dy_3d),
                    v_div_bar_pv_anom_dy=vw(vdiv_bar_pvanom_dy_3d),
                    # 16 divergent adiabatic/diabatic horizontal
                    u_div_diabatic_pv_bar_dx=vw(udm_pvbar_dx_3d),
                    u_div_diabatic_pv_anom_dx=vw(udm_pvanom_dx_3d),
                    u_div_adiabatic_pv_bar_dx=vw(udd_pvbar_dx_3d),
                    u_div_adiabatic_pv_anom_dx=vw(udd_pvanom_dx_3d),
                    v_div_diabatic_pv_bar_dy=vw(vdm_pvbar_dy_3d),
                    v_div_diabatic_pv_anom_dy=vw(vdm_pvanom_dy_3d),
                    v_div_adiabatic_pv_bar_dy=vw(vdd_pvbar_dy_3d),
                    v_div_adiabatic_pv_anom_dy=vw(vdd_pvanom_dy_3d),
                    u_div_qg_diabatic_pv_bar_dx=vw(udqm_pvbar_dx_3d),
                    u_div_qg_diabatic_pv_anom_dx=vw(udqm_pvanom_dx_3d),
                    v_div_qg_diabatic_pv_bar_dy=vw(vdqm_pvbar_dy_3d),
                    v_div_qg_diabatic_pv_anom_dy=vw(vdqm_pvanom_dy_3d),
                    u_div_lhr_moist_pv_bar_dx=vw(udem_pvbar_dx_3d),
                    u_div_lhr_moist_pv_anom_dx=vw(udem_pvanom_dx_3d),
                    v_div_lhr_moist_pv_bar_dy=vw(vdem_pvbar_dy_3d),
                    v_div_lhr_moist_pv_anom_dy=vw(vdem_pvanom_dy_3d),
                    # 8 alt vertical
                    w_adiabatic_pv_bar_dp=vw(w_dry_pvbar_dp_3d),
                    w_adiabatic_pv_anom_dp=vw(w_dry_pvanom_dp_3d),
                    w_diabatic_pv_bar_dp=vw(w_moist_pvbar_dp_3d),
                    w_diabatic_pv_anom_dp=vw(w_moist_pvanom_dp_3d),
                    w_qg_diabatic_pv_bar_dp=vw(w_qgm_pvbar_dp_3d),
                    w_qg_diabatic_pv_anom_dp=vw(w_qgm_pvanom_dp_3d),
                    w_lhr_moist_pv_bar_dp=vw(w_em_pvbar_dp_3d),
                    w_lhr_moist_pv_anom_dp=vw(w_em_pvanom_dp_3d),

                    # ── 3-D per-level cubes ──
                    z_3d=z_m_3d, pv_3d=cube3d["pv"],
                    u_3d=cube3d["u"], v_3d=cube3d["v"],
                    w_3d=cube3d["w"], t_3d=cube3d["t"],
                    pv_dt_3d=cube3d["pv_dt"],
                    pv_dx_3d=cube3d["pv_total_dx"],
                    pv_dy_3d=cube3d["pv_total_dy"],
                    pv_dp_3d=cube3d["pv_total_dp"],
                    pv_dx_dx_3d=cube3d["pv_total_dx_dx"],
                    pv_dy_dy_3d=cube3d["pv_total_dy_dy"],
                    pv_dx_dy_3d=cube3d["pv_total_dx_dy"],
                    u_bar_3d=cube3d["u_bar"],
                    v_bar_3d=cube3d["v_bar"],
                    w_bar_3d=cube3d["w_bar"],
                    pv_bar_3d=cube3d["pv_bar"],
                    u_anom_3d=cube3d["u_anom"],
                    v_anom_3d=cube3d["v_anom"],
                    w_anom_3d=cube3d["w_anom"],
                    pv_anom_3d=cube3d["pv_anom"],
                    pv_bar_dx_3d=cube3d["pv_bar_dx"],
                    pv_bar_dy_3d=cube3d["pv_bar_dy"],
                    pv_bar_dp_3d=cube3d["pv_bar_dp"],
                    pv_bar_dt_3d=cube3d["pv_bar_dt"],
                    pv_anom_dx_3d=cube3d["pv_anom_dx"],
                    pv_anom_dy_3d=cube3d["pv_anom_dy"],
                    pv_anom_dp_3d=cube3d["pv_anom_dp"],
                    pv_anom_dt_3d=cube3d["pv_anom_dt"],
                    theta_3d=cube3d["theta"],
                    theta_dt_3d=cube3d["theta_dt"],
                    theta_dot_3d=cube3d["theta_dot"],
                    Q_3d=cube3d["Q"],
                    # Total Helmholtz 3-D
                    u_rot_3d=cube3d["u_rot"],
                    u_div_3d_helm=cube3d["u_div"],
                    v_rot_3d=cube3d["v_rot"],
                    v_div_3d_helm=cube3d["v_div"],
                    # Clim Helmholtz 3-D
                    u_rot_bar_3d=cube3d["u_rot_bar"],
                    u_div_bar_3d=cube3d["u_div_bar"],
                    v_rot_bar_3d=cube3d["v_rot_bar"],
                    v_div_bar_3d=cube3d["v_div_bar"],
                    # Anomaly Helmholtz 3-D
                    u_rot_anom_3d=cube3d["u_rot_anom"],
                    u_div_anom_3d=cube3d["u_div_anom"],
                    u_har_anom_3d=cube3d["u_har_anom"],
                    v_rot_anom_3d=cube3d["v_rot_anom"],
                    v_div_anom_3d=cube3d["v_div_anom"],
                    v_har_anom_3d=cube3d["v_har_anom"],
                    u_div_diabatic_3d=cube3d["u_div_diabatic"],
                    v_div_diabatic_3d=cube3d["v_div_diabatic"],
                    u_div_adiabatic_3d=cube3d["u_div_adiabatic"],
                    v_div_adiabatic_3d=cube3d["v_div_adiabatic"],
                    w_adiabatic_3d=cube3d["w_adiabatic"],
                    w_diabatic_3d=cube3d["w_diabatic"],
                    w_qg_diabatic_3d=cube3d["w_qg_diabatic"],
                    u_div_qg_diabatic_3d=cube3d["u_div_qg_diabatic"],
                    v_div_qg_diabatic_3d=cube3d["v_div_qg_diabatic"],
                    w_lhr_moist_3d=cube3d["w_lhr_moist"],
                    u_div_lhr_moist_3d=cube3d["u_div_lhr_moist"],
                    v_div_lhr_moist_3d=cube3d["v_div_lhr_moist"],
                    q_3d=cube3d["q"],
                    t_dt_3d=cube3d["t_dt"],
                    # Cross-terms 3-D
                    u_anom_pv_bar_dx_3d=uanom_pvbar_dx_3d,
                    u_anom_pv_anom_dx_3d=uanom_pvanom_dx_3d,
                    u_bar_pv_anom_dx_3d=ubar_pvanom_dx_3d,
                    u_bar_pv_bar_dx_3d=ubar_pvbar_dx_3d,
                    v_anom_pv_bar_dy_3d=vanom_pvbar_dy_3d,
                    v_anom_pv_anom_dy_3d=vanom_pvanom_dy_3d,
                    v_bar_pv_anom_dy_3d=vbar_pvanom_dy_3d,
                    v_bar_pv_bar_dy_3d=vbar_pvbar_dy_3d,
                    w_anom_pv_bar_dp_3d=wanom_pvbar_dp_3d,
                    w_anom_pv_anom_dp_3d=wanom_pvanom_dp_3d,
                    w_bar_pv_anom_dp_3d=wbar_pvanom_dp_3d,
                    w_bar_pv_bar_dp_3d=wbar_pvbar_dp_3d,
                    u_rot_anom_pv_bar_dx_3d=urot_anom_pvbar_dx_3d,
                    u_rot_anom_pv_anom_dx_3d=urot_anom_pvanom_dx_3d,
                    u_div_anom_pv_bar_dx_3d=udiv_anom_pvbar_dx_3d,
                    u_div_anom_pv_anom_dx_3d=udiv_anom_pvanom_dx_3d,
                    u_rot_bar_pv_bar_dx_3d=urot_bar_pvbar_dx_3d,
                    u_rot_bar_pv_anom_dx_3d=urot_bar_pvanom_dx_3d,
                    u_div_bar_pv_bar_dx_3d=udiv_bar_pvbar_dx_3d,
                    u_div_bar_pv_anom_dx_3d=udiv_bar_pvanom_dx_3d,
                    v_rot_anom_pv_bar_dy_3d=vrot_anom_pvbar_dy_3d,
                    v_rot_anom_pv_anom_dy_3d=vrot_anom_pvanom_dy_3d,
                    v_div_anom_pv_bar_dy_3d=vdiv_anom_pvbar_dy_3d,
                    v_div_anom_pv_anom_dy_3d=vdiv_anom_pvanom_dy_3d,
                    v_rot_bar_pv_bar_dy_3d=vrot_bar_pvbar_dy_3d,
                    v_rot_bar_pv_anom_dy_3d=vrot_bar_pvanom_dy_3d,
                    v_div_bar_pv_bar_dy_3d=vdiv_bar_pvbar_dy_3d,
                    v_div_bar_pv_anom_dy_3d=vdiv_bar_pvanom_dy_3d,
                    u_div_diabatic_pv_bar_dx_3d=udm_pvbar_dx_3d,
                    u_div_diabatic_pv_anom_dx_3d=udm_pvanom_dx_3d,
                    u_div_adiabatic_pv_bar_dx_3d=udd_pvbar_dx_3d,
                    u_div_adiabatic_pv_anom_dx_3d=udd_pvanom_dx_3d,
                    v_div_diabatic_pv_bar_dy_3d=vdm_pvbar_dy_3d,
                    v_div_diabatic_pv_anom_dy_3d=vdm_pvanom_dy_3d,
                    v_div_adiabatic_pv_bar_dy_3d=vdd_pvbar_dy_3d,
                    v_div_adiabatic_pv_anom_dy_3d=vdd_pvanom_dy_3d,
                    w_adiabatic_pv_bar_dp_3d=w_dry_pvbar_dp_3d,
                    w_adiabatic_pv_anom_dp_3d=w_dry_pvanom_dp_3d,
                    w_diabatic_pv_bar_dp_3d=w_moist_pvbar_dp_3d,
                    w_diabatic_pv_anom_dp_3d=w_moist_pvanom_dp_3d,
                    u_div_qg_diabatic_pv_bar_dx_3d=udqm_pvbar_dx_3d,
                    u_div_qg_diabatic_pv_anom_dx_3d=udqm_pvanom_dx_3d,
                    v_div_qg_diabatic_pv_bar_dy_3d=vdqm_pvbar_dy_3d,
                    v_div_qg_diabatic_pv_anom_dy_3d=vdqm_pvanom_dy_3d,
                    w_qg_diabatic_pv_bar_dp_3d=w_qgm_pvbar_dp_3d,
                    w_qg_diabatic_pv_anom_dp_3d=w_qgm_pvanom_dp_3d,
                    u_div_lhr_moist_pv_bar_dx_3d=udem_pvbar_dx_3d,
                    u_div_lhr_moist_pv_anom_dx_3d=udem_pvanom_dx_3d,
                    v_div_lhr_moist_pv_bar_dy_3d=vdem_pvbar_dy_3d,
                    v_div_lhr_moist_pv_anom_dy_3d=vdem_pvanom_dy_3d,
                    w_lhr_moist_pv_bar_dp_3d=w_em_pvbar_dp_3d,
                    w_lhr_moist_pv_anom_dp_3d=w_em_pvanom_dp_3d,
                )

            # ── Optionally compute + merge PPVI fields (write together) ──
            if also_ppvi:
                try:
                    record.update(self._ppvi_compute_keys(
                        record, ts, ds, clim_ds, cplev, ppvi_geom))
                except Exception as exc:  # noqa: BLE001 - log & continue
                    _log(f"  PPVI (inline) dh={dh:+d} FAILED: {exc!r}")

            with tempfile.NamedTemporaryFile(
                dir=out_fp.parent, prefix=out_fp.stem + ".",
                suffix=".npz", delete=False,
            ) as tf:
                tmp_name = tf.name
                np.savez_compressed(tf, **record)
            os.replace(tmp_name, str(out_fp))
            written += 1
            gc.collect()

        _log(f"-> Event {track_id}: wrote {written} NPZ(s).")
        return written

    # ── piecewise PV inversion pass ────────────────────────────────

    # Fixed Wu inversion conventions. 9 levels 1000→100 hPa (matches the npz
    # ``levels`` and the audited per-level PPVI); index 0 = 1000 hPa bottom-θ,
    # index 8 = 100 hPa top-θ, 1..7 = interior PV.
    # 9-level Wu grid.  NL is a recompile-time PARAMETER in wuppvi.f and this
    # list MUST match solver.PR -- they are edited together or pass D silently
    # mismatches.  NL=20 was tried on 2026-08-10 and is BROKEN in this port:
    # the f2py build uses -frecursive (stack locals, no zero-init) whereas the
    # original relied on -fno-automatic static zero-init, and only 7 of ~35
    # local arrays in BALP are explicitly zeroed.  At NL=9 the leftover stack
    # happens to be harmless; at NL=20 pass D returns ~95 % NaN.
    _WU_PLEVS = [1000, 850, 700, 500, 400, 300, 250, 200, 100]
    _WU2SI = 1.0e-8          # Wu pseudo-PV → SI Ertel PV
    _WU_MI = 9999.90         # Wu missing-value sentinel
    _BAND_N = 85.5           # fixed inversion latitude band (north)
    _BAND_S = 10.5           # fixed inversion latitude band (south)

    def _ppvi_geom(self, ds: xr.Dataset, inv_lon_half: float) -> dict:
        """Build the static inversion geometry for an event.

        Returns a dict of grid / header info reused across all ``dh`` of a
        single event (depends only on the ERA5 grid + ``inv_lon_half``).
        """
        dlat = float(abs(np.diff(ds.latitude.values).mean()))
        dlon = float(abs(np.diff(ds.longitude.values).mean()))
        lat_all = ds.latitude.values
        lon_all = ds.longitude.values
        band_idx = np.where(
            (lat_all <= self._BAND_N + 1e-6)
            & (lat_all >= self._BAND_S - 1e-6))[0]
        band_lats = lat_all[band_idx]
        ny = len(band_idx)
        inv_lon_pad = int(round(inv_lon_half / dlon))
        nx = 2 * inv_lon_pad + 1
        zhdr = np.array(
            [self._BAND_S, 0.0, self._BAND_N, (nx - 1) * dlon,
             dlat, dlon, nx, ny], dtype=np.float32)
        wu_wavg_idx = [self._WU_PLEVS.index(l) for l in self.cfg.wavg_levels
                       if l in self._WU_PLEVS]
        return dict(
            plev=_plev_name(ds), lat_all=lat_all, lon_all=lon_all,
            nlon=len(lon_all), band_idx=band_idx, band_lats=band_lats,
            ny=ny, inv_lon_pad=inv_lon_pad, nx=nx, zhdr=zhdr,
            dlat=dlat, dlon=dlon, wu_wavg_idx=wu_wavg_idx,
        )

    def _ppvi_compute_keys(
        self, store: dict, ts: pd.Timestamp, ds: xr.Dataset,
        clim_ds: xr.Dataset, cplev: str, geom: dict,
    ) -> dict:
        """Compute the PPVI rotational-wind / PV-anomaly keys for one ``dh``.

        ``store`` must already provide the base-patch metadata
        (``center_lon``, ``lat_vec``, ``levels``) and the observed
        rotational-wind anomalies (``u_rot_anom_3d``, ``v_rot_anom_3d``).
        Returns a dict of new keys (no file IO).
        """
        from .ppvi import PIECES, invert_piecewise, psi_to_winds

        WU_PLEVS = self._WU_PLEVS
        WU2SI = self._WU2SI
        MI = self._WU_MI
        BAND_N = self._BAND_N
        plev = geom["plev"]
        band_idx = geom["band_idx"]
        band_lats = geom["band_lats"]
        ny = geom["ny"]
        nx = geom["nx"]
        inv_lon_pad = geom["inv_lon_pad"]
        zhdr = geom["zhdr"]
        lon_all = geom["lon_all"]
        nlon = geom["nlon"]
        dlat = geom["dlat"]
        dlon = geom["dlon"]
        wu_wavg_idx = geom["wu_wavg_idx"]

        center_lon = float(store["center_lon"])
        lat_vec = np.asarray(store["lat_vec"], dtype=float)
        npz_levels = list(np.asarray(store["levels"]).tolist())
        obs_u = np.asarray(store["u_rot_anom_3d"], dtype=float)
        obs_v = np.asarray(store["v_rot_anom_3d"], dtype=float)
        n_npz_lev, yp, xp = obs_u.shape

        ilon_c = int(np.argmin(np.abs(lon_all - center_lon)))
        lon_idx_inv = _wrapped_lon_index(
            ilon_c, LON_PAD=inv_lon_pad, nlon=nlon)

        # ── Extract event + climatological-mean cubes (NL, ny, nx) ──
        mo, dy, hr = int(ts.month), int(ts.day), int(ts.hour)

        def _ev(var):
            return (ds[var].sel(valid_time=ts)
                    .sel({plev: WU_PLEVS})
                    .isel(latitude=band_idx, longitude=lon_idx_inv)
                    .values.astype(np.float64))

        def _mn(var):
            # Same layout dispatch as `clim_bar`, on a single timestamp.
            if "slot" in clim_ds.dims:
                base = clim_ds[var].isel(slot=noleap_slot_scalar(ts))
            else:
                base = clim_ds[var].sel(month=mo, day=dy, hour=hr)
            return (base.sel({cplev: WU_PLEVS})
                    .isel(latitude=band_idx, longitude=lon_idx_inv)
                    .values.astype(np.float64))

        z_e, t_e, u_e, v_e = _ev("z"), _ev("t"), _ev("u"), _ev("v")
        z_m, t_m, u_m, v_m = _mn("z"), _mn("t"), _mn("u"), _mn("v")
        _zd = z_divisor(self.cfg)
        H_e, H_m = z_e / _zd, z_m / _zd

        # ── Run the Wu piecewise inversion ──
        res = invert_piecewise(
            H_m, t_m, u_m, v_m, H_e, t_e, u_e, v_e, zhdr)

        # Rotational winds per piece on the inversion grid (NL, ny, nx).
        piece_u: dict[str, np.ndarray] = {}
        piece_v: dict[str, np.ndarray] = {}
        for name in PIECES:
            ur, vr = psi_to_winds(
                res["psi_pieces"][name], band_lats, dlat, dlon)
            piece_u[name], piece_v[name] = ur, vr

        # Wu PV anomaly (interior levels valid; sentinel → NaN).
        q_e = np.asarray(res["Q_event"], dtype=float)
        q_m = np.asarray(res["Q_mean"], dtype=float)
        bad = (np.abs(q_e) >= MI * 0.99) | (np.abs(q_m) >= MI * 0.99)
        pv_anom_wu = (q_e - q_m) * WU2SI
        pv_anom_wu[bad] = np.nan

        # ── Crop inversion grid (ny, nx) → event patch (yp, xp) ──
        # Latitude: match each patch row to a fixed-band row by value.
        row_for = np.full(yp, -1, dtype=int)
        for j, plat in enumerate(lat_vec):
            if not np.isfinite(plat):
                continue
            r = int(round((BAND_N - plat) / dlat))
            if 0 <= r < ny and abs(band_lats[r] - plat) < 0.25:
                row_for[j] = r
        # Longitude: map patch columns to inversion columns by grid index.
        patch_lon_pad = (xp - 1) // 2
        lon_idx_patch = _wrapped_lon_index(
            ilon_c, LON_PAD=patch_lon_pad, nlon=nlon)
        pos_of = {int(g): k for k, g in enumerate(lon_idx_inv)}
        col_for = np.array(
            [pos_of.get(int(g), -1) for g in lon_idx_patch], dtype=int)

        vr_mask = row_for >= 0
        vc_mask = col_for >= 0
        rr = row_for[vr_mask]
        cc = col_for[vc_mask]
        ridx = np.where(vr_mask)[0]
        cidx = np.where(vc_mask)[0]
        wu_pos = [npz_levels.index(p) for p in WU_PLEVS]

        def _crop(cube):
            out = np.full((cube.shape[0], yp, xp), np.nan, dtype=float)
            if rr.size and cc.size:
                sub = cube[:, rr][:, :, cc]
                out[np.ix_(np.arange(cube.shape[0]), ridx, cidx)] = sub
            return out

        def _pad_levels(cube8):
            """(NL, yp, xp) Wu levels → (n_npz_lev, yp, xp), NaN elsewhere."""
            out = np.full((n_npz_lev, yp, xp), np.nan, dtype=float)
            out[wu_pos] = cube8
            return out

        piece_u_c = {n: _crop(piece_u[n]) for n in PIECES}
        piece_v_c = {n: _crop(piece_v[n]) for n in PIECES}
        pv_anom_wu_c = _crop(pv_anom_wu)

        # Residual vs observed anomaly rotational wind (Wu levels only).
        sum_u = sum(piece_u_c[n] for n in PIECES)
        sum_v = sum(piece_v_c[n] for n in PIECES)
        resid_u = obs_u[wu_pos] - sum_u
        resid_v = obs_v[wu_pos] - sum_v

        # ── Weighted vertical average → 2-D (over wavg_levels) ──
        z_patch = _crop(z_e) / z_divisor(self.cfg)   # geopotential height [m]
        zw = z_patch[wu_wavg_idx]
        wt = np.exp(-zw / H_SCALE)
        den = np.nansum(wt, axis=0)
        dmask = den > 0

        def _wavg(cube8):
            num = np.nansum(cube8[wu_wavg_idx] * wt, axis=0)
            out = np.full_like(num, np.nan)
            out[dmask] = num[dmask] / den[dmask]
            return out

        # ── Assemble new keys ──
        # Per-level decomposition: the solver piece name is the 1-based Wu
        # level index as a string; map it to the hPa level for the npz key,
        # e.g. piece "7" → 250 hPa → ``u_rot_anom_ppvi_250(_3d)``.
        new: dict[str, np.ndarray] = {}
        for n in PIECES:
            L = WU_PLEVS[int(n) - 1]
            new[f"u_rot_anom_ppvi_{L}_3d"] = _pad_levels(piece_u_c[n])
            new[f"v_rot_anom_ppvi_{L}_3d"] = _pad_levels(piece_v_c[n])
            new[f"u_rot_anom_ppvi_{L}"] = _wavg(piece_u_c[n])
            new[f"v_rot_anom_ppvi_{L}"] = _wavg(piece_v_c[n])
        new["u_rot_anom_residual_ppvi_3d"] = _pad_levels(resid_u)
        new["v_rot_anom_residual_ppvi_3d"] = _pad_levels(resid_v)
        new["u_rot_anom_residual_ppvi"] = _wavg(resid_u)
        new["v_rot_anom_residual_ppvi"] = _wavg(resid_v)
        new["pv_anom_wu_3d"] = _pad_levels(pv_anom_wu_c)
        new["pv_anom_wu"] = _wavg(pv_anom_wu_c)
        return new

    def compute_ppvi_for_event(
        self,
        evt_name: str,
        track_id: int,
        lat0: float,
        lon0: float,
        base_ts: pd.Timestamp,
        *,
        inv_lon_half: float = 90.0,
    ) -> int:
        """Append Wu piecewise PV-inversion rotational winds to NPZ files.

        For each ``dh`` in :attr:`cfg.rel_hours` this reads the **existing**
        NPZ (produced by :meth:`process_event`), runs the Wu piecewise PV
        inversion on a **fixed** Northern-Hemisphere latitude band
        (85.5°N → 10.5°N, ``NY=51``) and an event-centred longitude window
        of half-width ``inv_lon_half`` (default ±90° → ``NX=121``), then
        crops the resulting balanced rotational winds back to the event
        patch and **appends** the new fields in-place.

        Only NPZs that already exist on disk are touched here; missing NPZs
        should be produced by :meth:`process_event` with ``also_ppvi=True``
        (the ``ppvi`` CLI command does both — replace in existing, write
        fresh ones together).

        This **replaces** any superseded grouped-piece keys
        (``[uv]_rot_anom_{lower,middle,upper}(_3d)``) with the per-level
        decomposition. New keys (each also with a ``_3d`` variant), for each
        Wu level ``L`` in 1000…100 hPa:
        ``u_rot_anom_ppvi_{L}``, ``v_rot_anom_ppvi_{L}``,
        plus ``u/v_rot_anom_residual_ppvi`` (observed − Σ of all per-level
        pieces) and ``pv_anom_wu`` (Wu PV anomaly in SI, interior levels only).

        Returns the number of NPZ files updated.
        """
        _log(f"\n--- PPVI (append) for event: {track_id} "
             f"at ({lat0}, {lon0}) ---")

        # Small-chunked clim: release per-event month slices (avoid the
        # ~19 GB/worker accumulation that throttled the cgroup at scale).
        clim_ds = self._get_clim(chunks={"day": 1, "hour": 1})

        # ── Which dh have an existing NPZ that still needs PPVI fields? ──
        todo: list[tuple[int, pd.Timestamp, Path]] = []
        for dh in self.cfg.rel_hours:
            ts = base_ts + pd.Timedelta(hours=dh)
            fp = self._out_path(evt_name, dh, track_id, ts)
            if not fp.exists():
                continue
            if self.cfg.skip_existing:
                with np.load(fp, allow_pickle=True) as z:
                    if "u_rot_anom_ppvi_250_3d" in z.files:
                        continue
            todo.append((dh, ts, fp))
        if not todo:
            _log(f"-> Event {track_id}: no existing NPZ needs PPVI.")
            return 0

        # ── Load only z, t, u, v for the window ──
        month_keys = month_keys_for_window(
            base_ts, hmin=min(self.cfg.rel_hours),
            hmax=max(self.cfg.rel_hours))
        # Lazy open (per-timestep chunks): a PPVI worker only reads the few
        # timesteps it inverts, not the whole ~15 GB month — this keeps
        # per-worker RSS at ~1-2 GB so many workers fit under the 2 TB cap.
        ds = open_months_ds(
            self.cfg.data_dir, ["z", "t", "u", "v"], month_keys,
            engine=self.cfg.engine, chunks={"valid_time": 1})
        cplev = _plev_name(clim_ds)
        geom = self._ppvi_geom(ds, inv_lon_half)
        dt_index = pd.to_datetime(ds.valid_time.values)

        updated = 0
        for dh, ts, fp in todo:
            if ts not in dt_index:
                continue
            tmp_name = None
            try:
                with np.load(fp, allow_pickle=True) as z:
                    store = {k: z[k] for k in z.files}
                # Drop superseded grouped-piece keys so the npz is *replaced*,
                # not appended (old lower/middle/upper PPVI fields).
                for k in [k for k in store if _OLD_PIECE_KEY.match(k)]:
                    store.pop(k, None)
                new = self._ppvi_compute_keys(
                    store, ts, ds, clim_ds, cplev, geom)
                store.update(new)
                if "u_rot_anom_ppvi_250_3d" not in store:
                    raise RuntimeError("PPVI produced no per-level keys")
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".npz", dir=str(fp.parent),
                    delete=False,
                ) as tf:
                    tmp_name = tf.name
                    np.savez_compressed(tf, **store)
                os.replace(tmp_name, str(fp))
                tmp_name = None
                updated += 1
            except Exception as exc:  # noqa: BLE001 - log & continue
                if tmp_name and os.path.exists(tmp_name):
                    os.unlink(tmp_name)
                _log(f"-> PPVI dh={dh:+d} FAILED: {exc!r}")
            gc.collect()

        # Release the ERA5 file handles / dask buffers for this event so a
        # long-lived worker doesn't accumulate them across thousands of events
        # (this leak drove per-worker RSS to ~8 GB and throttled the cgroup).
        try:
            ds.close()
        except Exception:  # noqa: BLE001
            pass
        del ds
        gc.collect()
        _log(f"-> Event {track_id}: PPVI appended to {updated} NPZ(s).")
        return updated

    # ── output path ────────────────────────────────────────────────

    def _out_path(
        self, evt: str, dh: int, track_id: int | str, ts: pd.Timestamp,
    ) -> Path:
        """Compute the output NPZ file path."""
        return (
            Path(self.cfg.output_dir)
            / evt
            / f"dh={dh:+d}"
            / f"track_{track_id}_{ts.strftime('%Y%m%d%H')}_dh{dh:+d}.npz"
        )
