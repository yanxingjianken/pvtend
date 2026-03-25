"""Hourly climatology computation with Fourier smoothing.

Computes the hourly climatology for each variable and month from ERA5
(or other reanalysis) data.  The climatology is smoothed in time
(day-of-year) using a low-pass Fourier filter (4 harmonics) and
optionally in the zonal direction (2 zonal modes).

Output: per-variable, per-month NetCDF files suitable for anomaly
computation in the step2 tendency analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import xarray as xr

from .constants import CLIM_VARIABLES, MONTH_ABBREVS, R_EARTH
from .helmholtz import helmholtz_decomposition


# ═══════════════════════════════════════════════════════════════
#  Fourier smoothing utilities
# ═══════════════════════════════════════════════════════════════


def fourier_filter_1d(
    arr: np.ndarray,
    n_modes: int,
    axis: int = 0,
) -> np.ndarray:
    """Low-pass filter via FFT truncation along one axis.

    Retains only the first *n_modes* Fourier coefficients (plus the DC
    component) and inverse-transforms back.

    Args:
        arr: Input array.
        n_modes: Number of Fourier modes to retain (excluding DC).
        axis: Axis along which to filter.

    Returns:
        Filtered array, same shape as input.
    """
    spec = np.fft.rfft(arr, axis=axis)

    # Zero out modes beyond n_modes
    n_total = spec.shape[axis]
    if n_modes + 1 < n_total:
        slices: list[slice] = [slice(None)] * spec.ndim
        slices[axis] = slice(n_modes + 1, None)
        spec[tuple(slices)] = 0.0

    return np.fft.irfft(spec, n=arr.shape[axis], axis=axis)


def smooth_climatology(
    data: np.ndarray,
    n_time_modes: int = 4,
    n_zonal_modes: int = 2,
    smooth_zonal: bool = True,
) -> np.ndarray:
    """Apply combined time + zonal Fourier smoothing to climatology.

    Args:
        data: Climatology array with axes ``(..., nlat, nlon)``.
            Typically ``(12, 31, 24, nlev, nlat, nlon)`` for
            ``(month, day, hour, level, lat, lon)``.
        n_time_modes: Number of temporal harmonics to retain.
        n_zonal_modes: Number of zonal harmonics to retain.
        smooth_zonal: Whether to also smooth in the zonal direction.

    Returns:
        Smoothed climatology, same shape.
    """
    # Time smoothing (along day-of-month axis, assumed axis=1)
    out = fourier_filter_1d(data, n_time_modes, axis=1)

    # Zonal smoothing (along last axis)
    if smooth_zonal:
        out = fourier_filter_1d(out, n_zonal_modes, axis=-1)

    return out


# ═══════════════════════════════════════════════════════════════
#  Main climatology computation
# ═══════════════════════════════════════════════════════════════


def compute_climatology(
    data_dir: str | Path,
    output_dir: str | Path,
    years: Sequence[int],
    variables: Sequence[str] = CLIM_VARIABLES,
    per_month: bool = True,
    engine: str = "netcdf4",
) -> list[Path]:
    """Compute per-variable per-month hourly climatology.

    Accumulates hourly data from monthly ERA5 files, averages
    over years, then applies Fourier smoothing.

    Args:
        data_dir: Directory with ERA5 monthly NetCDF files.
        output_dir: Where to write climatology files.
        years: Years to include in the climatology.
        variables: Variable names to process.
        per_month: If True, output one file per ``(variable, month)``.
        engine: NetCDF engine.

    Returns:
        List of output file paths.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: list[Path] = []
    stem = f"era5_hourly_clim_{years[0]}-{years[-1]}"

    for month in range(1, 13):
        month3 = MONTH_ABBREVS[month - 1]

        for var in variables:
            print(f"  Computing clim: month={month3}, var={var} ...")

            # Accumulate across years
            accum: Optional[np.ndarray] = None
            count = 0

            for year in years:
                try:
                    from .preprocessing import load_era5_monthly

                    ds = load_era5_monthly(
                        data_dir, year, month, variables=[var], engine=engine
                    )
                except FileNotFoundError:
                    continue

                arr = ds[var].values  # (nt, nlev, nlat, nlon)
                if accum is None:
                    accum = np.zeros_like(arr, dtype=np.float64)
                accum += arr
                count += 1

            if count == 0:
                print(f"    WARNING: no data for {month3}/{var}")
                continue

            mean = (accum / count).astype(np.float32)

            # Apply Fourier smoothing along time axis
            smoothed = fourier_filter_1d(mean, 4, axis=0)

            # Save
            if per_month:
                # Raw climatology
                raw_path = output_dir / f"{stem}_{month}_{var}.nc"
                _save_clim_nc(mean, var, raw_path)

                # Smoothed climatology
                smooth_path = output_dir / f"{stem}_{month}_{var}_smooth.nc"
                _save_clim_nc(smoothed.astype(np.float32), var, smooth_path)

                output_files.extend([raw_path, smooth_path])

    return output_files


# ═══════════════════════════════════════════════════════════════
#  Loading helpers
# ═══════════════════════════════════════════════════════════════


def load_climatology(
    clim_path: str | Path,
    engine: str = "netcdf4",
    prefer_smooth: bool = False,
) -> xr.Dataset:
    """Load climatology, auto-detecting file layout.

    Supports three layouts (most → least granular):

    1. **Per-var-per-month** files: raw ``{stem}_{month}_{var}.nc``
       (or smoothed ``*_smooth.nc`` if *prefer_smooth* is True)
    2. **Per-variable** files: ``{stem}_{var}.nc``
    3. **Single merged** file

    Args:
        clim_path: Path to climatology file or directory stem.
        engine: NetCDF engine.
        prefer_smooth: If True, prefer smoothed files; if False, prefer raw.

    Returns:
        ``xr.Dataset`` with climatology fields.

    Raises:
        FileNotFoundError: If no climatology files are found.
    """
    clim_path = Path(clim_path)

    # Direct file
    if clim_path.is_file():
        return xr.open_dataset(clim_path, chunks=None, engine=engine)

    # Directory with per-var-per-month files
    parent = clim_path.parent if clim_path.suffix else clim_path
    stem = clim_path.stem.replace("_allvars", "") if clim_path.suffix else ""

    if parent.is_dir():
        if prefer_smooth:
            # Try smoothed per-variable-per-month files first
            pvm_files = sorted(parent.glob(f"{stem}*_smooth.nc"))
            if pvm_files:
                return xr.open_mfdataset(
                    [str(f) for f in pvm_files],
                    chunks=None,
                    engine=engine,
                    combine="by_coords",
                    join="outer",
                )

        # Try raw per-variable-per-month files (exclude _smooth)
        raw_files = sorted(
            f for f in parent.glob(f"{stem}_*.nc")
            if "_smooth" not in f.stem and "_allvars" not in f.stem
        )
        if raw_files:
            return xr.open_mfdataset(
                [str(f) for f in raw_files],
                chunks=None,
                engine=engine,
                combine="by_coords",
                join="outer",
            )

        # If prefer_smooth was False and no raw files, try smooth as fallback
        if not prefer_smooth:
            pvm_files = sorted(parent.glob(f"{stem}*_smooth.nc"))
            if pvm_files:
                return xr.open_mfdataset(
                    [str(f) for f in pvm_files],
                    chunks=None,
                    engine=engine,
                    combine="by_coords",
                    join="outer",
                )

    raise FileNotFoundError(f"Climatology not found at {clim_path}")


# ═══════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════


def _save_clim_nc(data: np.ndarray, var_name: str, path: Path) -> None:
    """Save a climatology array as NetCDF.

    Args:
        data: Climatology values, arbitrary number of dimensions.
        var_name: Variable name to use in the dataset.
        path: Output file path.
    """
    dim_names = ["time"] + [f"dim_{i}" for i in range(data.ndim - 1)]
    ds = xr.Dataset({var_name: (dim_names, data)})
    ds.to_netcdf(path)


# ═══════════════════════════════════════════════════════════════
#  Climatological Helmholtz decomposition
# ═══════════════════════════════════════════════════════════════


def compute_helmholtz_climatology(
    clim_dir: str | Path,
    output_dir: str | Path,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    clim_stem: str = "era5_hourly_clim_1990-2020",
    engine: str = "netcdf4",
) -> list[Path]:
    """Pre-compute Helmholtz decomposition of the climatological wind.

    For each of the 12 months, loads the hourly climatological (u_bar,
    v_bar) fields, applies ``helmholtz_decomposition`` at every
    (hour, level) slice, and writes two NetCDF files per month
    containing u_rot_bar/u_div_bar and v_rot_bar/v_div_bar.

    Args:
        clim_dir: Directory containing per-variable-per-month climatology
            NetCDF files (e.g. ``era5_hourly_clim_1990-2020_jan_u.nc``).
        output_dir: Where to write the 24 Helmholtz climatology files.
        lat: Latitude array [degrees], matching the climatology grid.
        lon: Longitude array [degrees], matching the climatology grid.
        clim_stem: Filename stem for the climatology files.
        engine: NetCDF engine.

    Returns:
        List of 24 output file paths.
    """
    clim_dir = Path(clim_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure ascending latitude for Helmholtz solver
    if lat[0] > lat[-1]:
        lat_asc = lat[::-1]
        flip_lat = True
    else:
        lat_asc = lat
        flip_lat = False

    output_files: list[Path] = []

    for month in range(1, 13):
        month3 = MONTH_ABBREVS[month - 1]
        print(f"  Helmholtz clim: month={month3} ...", flush=True)

        # Load climatological u and v for this month
        u_path = clim_dir / f"{clim_stem}_{month3}_u.nc"
        v_path = clim_dir / f"{clim_stem}_{month3}_v.nc"
        if not u_path.exists() or not v_path.exists():
            print(f"    WARNING: missing {u_path} or {v_path}, skipping")
            continue

        ds_u = xr.open_dataset(u_path, engine=engine)
        ds_v = xr.open_dataset(v_path, engine=engine)

        # Climatology files are 6-D: (month, day, hour, level, lat, lon).
        # Average over month (single element) but KEEP day dimension so
        # tendency.py can index by (day, hour) → (day, hour, level, lat, lon).
        u_bar = ds_u["u"].mean(dim="month").values  # (nday, 24, nlev, nlat, nlon)
        v_bar = ds_v["v"].mean(dim="month").values

        nday = u_bar.shape[0]
        nt = u_bar.shape[1]    # 24 hours
        nlev = u_bar.shape[2]  # 9 pressure levels

        u_rot_bar = np.zeros_like(u_bar)
        u_div_bar = np.zeros_like(u_bar)
        v_rot_bar = np.zeros_like(v_bar)
        v_div_bar = np.zeros_like(v_bar)

        for di in range(nday):
            for ti in range(nt):
                for li in range(nlev):
                    u2d = u_bar[di, ti, li]
                    v2d = v_bar[di, ti, li]
                    if flip_lat:
                        u2d = u2d[::-1]
                        v2d = v2d[::-1]
                    helm = helmholtz_decomposition(
                        u2d, v2d, lat_asc, lon,
                        R_earth=R_EARTH, method="spherical",
                    )
                    if flip_lat:
                        u_rot_bar[di, ti, li] = helm["u_rot"][::-1]
                        u_div_bar[di, ti, li] = helm["u_div"][::-1]
                        v_rot_bar[di, ti, li] = helm["v_rot"][::-1]
                        v_div_bar[di, ti, li] = helm["v_div"][::-1]
                    else:
                        u_rot_bar[di, ti, li] = helm["u_rot"]
                        u_div_bar[di, ti, li] = helm["u_div"]
                        v_rot_bar[di, ti, li] = helm["v_rot"]
                        v_div_bar[di, ti, li] = helm["v_div"]

        ds_u.close()
        ds_v.close()

        # Write u-components
        u_out_path = output_dir / f"{clim_stem}_{month3}_u_helmholtz.nc"
        dims = ["day", "hour", "pressure_level", "latitude", "longitude"]
        ds_out = xr.Dataset({
            "u_rot_bar": (dims, u_rot_bar.astype(np.float32)),
            "u_div_bar": (dims, u_div_bar.astype(np.float32)),
        })
        ds_out.to_netcdf(u_out_path)
        output_files.append(u_out_path)

        # Write v-components
        v_out_path = output_dir / f"{clim_stem}_{month3}_v_helmholtz.nc"
        ds_out = xr.Dataset({
            "v_rot_bar": (dims, v_rot_bar.astype(np.float32)),
            "v_div_bar": (dims, v_div_bar.astype(np.float32)),
        })
        ds_out.to_netcdf(v_out_path)
        output_files.append(v_out_path)

        print(f"    Wrote {u_out_path.name}, {v_out_path.name}")

    return output_files


def load_helmholtz_climatology(
    clim_dir: str | Path,
    month: int,
    *,
    clim_stem: str = "era5_hourly_clim_1990-2020",
    engine: str = "netcdf4",
) -> dict[str, np.ndarray]:
    """Load pre-computed Helmholtz-decomposed climatological wind fields.

    Args:
        clim_dir: Directory containing Helmholtz climatology files.
        month: Month number (1–12).
        clim_stem: Filename stem.
        engine: NetCDF engine.

    Returns:
        Dictionary with keys ``u_rot_bar``, ``u_div_bar``,
        ``v_rot_bar``, ``v_div_bar`` — each shape
        ``(nday, 24, nlev, nlat, nlon)``  (day × hour × level × lat × lon).

    Raises:
        FileNotFoundError: If the Helmholtz climatology files are missing.
    """
    clim_dir = Path(clim_dir)

    month3 = MONTH_ABBREVS[month - 1]
    u_path = clim_dir / f"{clim_stem}_{month3}_u_helmholtz.nc"
    v_path = clim_dir / f"{clim_stem}_{month3}_v_helmholtz.nc"

    if not u_path.exists() or not v_path.exists():
        raise FileNotFoundError(
            f"Helmholtz climatology not found: {u_path} and/or {v_path}. "
            f"Run `pvtend-pipeline clim-helmholtz` first."
        )

    ds_u = xr.open_dataset(u_path, engine=engine)
    ds_v = xr.open_dataset(v_path, engine=engine)

    result = {
        "u_rot_bar": ds_u["u_rot_bar"].values,
        "u_div_bar": ds_u["u_div_bar"].values,
        "v_rot_bar": ds_v["v_rot_bar"].values,
        "v_div_bar": ds_v["v_div_bar"].values,
    }

    ds_u.close()
    ds_v.close()

    return result
