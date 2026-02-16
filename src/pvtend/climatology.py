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

from .constants import CLIM_VARIABLES, MONTH_ABBREVS


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
) -> xr.Dataset:
    """Load climatology, auto-detecting file layout.

    Supports three layouts (most → least granular):

    1. **Per-var-per-month** files: ``{stem}_{month}_{var}_smooth.nc``
    2. **Per-variable** files: ``{stem}_{var}.nc``
    3. **Single merged** file

    Args:
        clim_path: Path to climatology file or directory stem.
        engine: NetCDF engine.

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
        # Try smoothed per-variable-per-month files
        pvm_files = sorted(parent.glob(f"{stem}*_smooth.nc"))
        if pvm_files:
            return xr.open_mfdataset(
                [str(f) for f in pvm_files],
                chunks=None,
                engine=engine,
                combine="by_coords",
                join="outer",
            )

        # Try per-variable files
        per_var = sorted(parent.glob(f"{stem}_*.nc"))
        if per_var:
            return xr.open_mfdataset(
                per_var,
                chunks=None,
                engine=engine,
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
