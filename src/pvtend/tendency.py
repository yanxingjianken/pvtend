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
import tempfile
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
    clim_path: Path, engine: str = "netcdf4",
) -> xr.Dataset:
    """Load climatology, auto-detecting the file layout.

    Fallback order:
      1. Single merged file
      2. Per-var-per-month files
      3. Per-variable files
    """
    clim_path = Path(clim_path)
    if clim_path.is_file():
        _log(f"Loading climatology from single file: {clim_path}")
        return xr.open_dataset(clim_path, chunks=None, engine=engine,
                               lock=False)

    parent = clim_path.parent
    stem = clim_path.stem.replace("_allvars", "")

    pvm_files = _find_per_var_month_files(parent, stem)
    if pvm_files:
        _log(f"Loading climatology from {len(pvm_files)} "
             f"per-var-per-month files")
        return xr.open_mfdataset(
            [str(f) for f in pvm_files],
            chunks=None, engine=engine, lock=False,
            combine="by_coords", join="outer",
        )

    per_var = sorted(
        f for f in parent.glob(f"{stem}_*.nc")
        if "_smoothed" not in f.stem and "_allvars" not in f.stem
    )
    if per_var:
        _log(f"Loading climatology from {len(per_var)} per-variable files")
        return xr.open_mfdataset(per_var, chunks=None, engine=engine,
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
    engine: str = "netcdf4",
) -> xr.Dataset:
    """Open multiple months of ERA5 data as a single dataset."""
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
        dsv = xr.open_mfdataset(
            fns, combine="by_coords", parallel=False,
            chunks=None, engine=engine, lock=False,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ∇²χ = -∂ω/∂p on full NH, return u/v divergent wind.

    Uses the spherical Laplacian (conservative form) with area-weighted
    RHS mean removal.  The full NH domain with periodic zonal BCs gives
    a much better-conditioned inversion than the local patch.
    """
    nlev, nlat, nlon = omega_nh.shape
    lat_rad = np.deg2rad(lat_nh)
    dlat = float(np.abs(np.diff(lat_nh).mean()))
    dlon = float(np.abs(np.diff(lon_nh).mean()))
    dy = np.deg2rad(dlat) * R_EARTH
    dx_arr = np.deg2rad(dlon) * R_EARTH * np.cos(lat_rad)
    dx_arr = np.maximum(dx_arr, dy * 0.1)
    dlon_rad = np.deg2rad(dlon)

    rhs = -ddp(omega_nh, plevs_pa)

    # Area-weighted mean removal
    cos_phi = np.cos(lat_rad)
    area_weights = cos_phi / cos_phi.sum()
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
    month_keys = month_keys_for_window(
        base_ts, hmin=cfg.rel_hours[0], hmax=cfg.rel_hours[-1])
    ds = open_months_ds(cfg.data_dir, ["u", "v", "w", "pv", "z", "t", "q"],
                        month_keys, engine=cfg.engine)

    # --- lat metrics & Coriolis ---
    ds["latitude_rad"] = np.deg2rad(ds.latitude)
    lat_rad_vals = ds["latitude_rad"].values.copy()
    lat_rad_vals[np.abs(lat_rad_vals - np.pi / 2) < 0.01] = np.pi / 2 - 0.01
    ds["latitude_rad"] = xr.DataArray(lat_rad_vals, dims=["latitude"])
    ds["f"] = 2 * OMEGA_E * np.sin(ds["latitude_rad"])

    # --- climatology ---
    CLIM = clim_ds
    mo = ds.valid_time.dt.month
    dy = ds.valid_time.dt.day
    hr = ds.valid_time.dt.hour
    ds["pv_bar"] = CLIM["pv"].sel(month=mo, day=dy, hour=hr)
    ds["u_bar"] = CLIM["u"].sel(month=mo, day=dy, hour=hr)
    ds["v_bar"] = CLIM["v"].sel(month=mo, day=dy, hour=hr)
    ds["w_bar"] = CLIM["w"].sel(month=mo, day=dy, hour=hr)
    ds["z_bar"] = CLIM["z"].sel(month=mo, day=dy, hour=hr)
    ds["t_bar"] = CLIM["t"].sel(month=mo, day=dy, hour=hr)

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

    # relative vorticity ζ = ∂v/∂x − ∂u/∂y
    ds["v_dx"] = _ddx_periodic_da(ds.v)
    ds["u_dy"] = _ddy_da(ds.u)
    ds["zeta"] = ds["v_dx"] - ds["u_dy"]

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
            helm = helmholtz_decomposition(
                u2d, v2d, lat_asc, lon_nh,
                R_earth=R_EARTH, method="spherical")
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
    # Gather unique months needed in this time window
    months_needed = sorted(set(ds.valid_time.dt.month.values.tolist()))
    clim_helm_cache: dict[int, dict[str, np.ndarray]] = {}
    for m in months_needed:
        clim_helm_cache[m] = load_helmholtz_climatology(
            cfg.clim_helmholtz_dir, m)

    # Build 4-D bar arrays by matching each timestep to its month/hour/level
    u_rot_bar_4d = np.zeros(shape_4d, dtype=np.float32)
    u_div_bar_4d = np.zeros(shape_4d, dtype=np.float32)
    v_rot_bar_4d = np.zeros(shape_4d, dtype=np.float32)
    v_div_bar_4d = np.zeros(shape_4d, dtype=np.float32)

    times = pd.to_datetime(ds.valid_time.values)
    for ti, t in enumerate(times):
        m = t.month
        hr = t.hour
        day = t.day  # 1-based calendar day
        ch = clim_helm_cache[m]
        # Climatology files now have shape (nday, 24, nlev, nlat, nlon)
        # Index by (day-1, hour) for daily-hourly resolution
        di = day - 1  # 0-based day index
        if di < ch["u_rot_bar"].shape[0] and hr < ch["u_rot_bar"].shape[1]:
            u_rot_bar_4d[ti] = ch["u_rot_bar"][di, hr]
            u_div_bar_4d[ti] = ch["u_div_bar"][di, hr]
            v_rot_bar_4d[ti] = ch["v_rot_bar"][di, hr]
            v_div_bar_4d[ti] = ch["v_div_bar"][di, hr]

    for name, arr in [
        ("u_rot_bar", u_rot_bar_4d), ("u_div_bar", u_div_bar_4d),
        ("v_rot_bar", v_rot_bar_4d), ("v_div_bar", v_div_bar_4d),
    ]:
        ds[name] = xr.DataArray(arr, dims=dims4d, coords=coords4d)

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

    def _get_clim(self) -> xr.Dataset:
        if self._clim is None:
            self._clim = load_climatology(self.cfg.clim_path, self.cfg.engine)
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
    ) -> int:
        """Process a single event and write NPZ files.

        Loads data **once** via :func:`with_derivs_for_window`, then
        iterates over ``cfg.rel_hours``, extracting a patch and computing
        QG omega + moist/dry + cross-terms + wavg per timestep.

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

            # NaN safety on adiabatic/diabatic decomposition outputs
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

            z_m_3d = cube3d["z"] / G0

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
            with tempfile.NamedTemporaryFile(
                dir=out_fp.parent, prefix=out_fp.stem + ".",
                suffix=".npz", delete=False,
            ) as tf:
                tmp_name = tf.name
                np.savez_compressed(
                    tf,
                    # ── Metadata ──
                    Y_rel=grid.Y_rel, X_rel=grid.X_rel,
                    levels=np.array(levels, dtype=np.int32),
                    wavg_levels=np.array(self.cfg.wavg_levels, dtype=np.int32),
                    H_SCALE=float(H_SCALE), G0=float(G0),
                    lat_vec=lat_vec_full.astype(float),
                    lon_vec_unwrapped=lon_unwrapped.astype(float),
                    track_id=int(track_id),
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
            os.replace(tmp_name, str(out_fp))
            written += 1
            gc.collect()

        _log(f"-> Event {track_id}: wrote {written} NPZ(s).")
        return written

    # ── output path ────────────────────────────────────────────────

    def _out_path(
        self, evt: str, dh: int, track_id: int, ts: pd.Timestamp,
    ) -> Path:
        """Compute the output NPZ file path."""
        return (
            Path(self.cfg.output_dir)
            / evt
            / f"dh={dh:+d}"
            / f"track_{track_id}_{ts.strftime('%Y%m%d%H')}_dh{dh:+d}.npz"
        )
