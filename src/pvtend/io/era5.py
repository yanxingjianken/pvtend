"""ERA5 monthly NetCDF file loading.

Supports the file naming convention era5_{var}_{YYYY}_{MM}.nc
and handles CDS artefacts (expver, number dimensions).
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence

import xarray as xr


def load_era5_month(
    data_dir: str | Path,
    year: int,
    month: int,
    variables: Sequence[str],
    *,
    engine: str = "netcdf4",
    chunks: dict | None = None,
) -> xr.Dataset:
    """Load ERA5 monthly data for specified variables.

    Parameters:
        data_dir: Directory with era5_{var}_{YYYY}_{MM}.nc files.
        year: Year.
        month: Month (1-12).
        variables: Variable names to load.
        engine: NetCDF engine.
        chunks: Dask chunks (None for eager loading).

    Returns:
        Merged xarray Dataset.
    """
    data_dir = Path(data_dir)
    parts = []
    for var in variables:
        fp = data_dir / f"era5_{var}_{year}_{month:02d}.nc"
        if not fp.exists():
            raise FileNotFoundError(f"Missing: {fp}")
        ds = xr.open_dataset(fp, engine=engine, chunks=chunks)
        ds = _drop_cds_artefacts(ds)
        ds = _ensure_valid_time(ds)
        if "level" in ds.dims and "pressure_level" not in ds.dims:
            ds = ds.rename({"level": "pressure_level"})
        ds = ds[[var]]
        parts.append(ds)

    merged = xr.merge(parts, compat="no_conflicts", join="inner")
    merged = merged.assign_coords(
        longitude=((merged.longitude + 180) % 360) - 180,
    ).sortby("longitude")
    return merged


def open_months_dataset(
    data_dir: str | Path,
    var_list: Sequence[str],
    month_keys: Sequence[tuple[int, int]],
    *,
    engine: str = "netcdf4",
    chunks: dict | None = None,
) -> xr.Dataset:
    """Open multiple months of ERA5 data as a single dataset.

    Parameters:
        data_dir: Directory with ERA5 monthly files.
        var_list: Variable names.
        month_keys: List of (year, month) tuples.
        engine: NetCDF engine.
        chunks: Dask chunks.

    Returns:
        Merged xarray Dataset spanning all requested months.
    """
    data_dir = Path(data_dir)
    parts = []
    for var in var_list:
        files = []
        for y, m in month_keys:
            fp = data_dir / f"era5_{var}_{y}_{m:02d}.nc"
            if fp.exists():
                files.append(str(fp))
        if not files:
            raise FileNotFoundError(f"No files for {var} in {month_keys}")
        dsv = xr.open_mfdataset(
            files, combine="by_coords", parallel=False,
            chunks=chunks, engine=engine,
        )
        dsv = _drop_cds_artefacts(dsv)
        dsv = _ensure_valid_time(dsv)
        if "level" in dsv.dims and "pressure_level" not in dsv.dims:
            dsv = dsv.rename({"level": "pressure_level"})
        dsv = dsv[[var]]
        parts.append(dsv)

    ds = xr.merge(parts, compat="no_conflicts", join="inner")
    ds = ds.assign_coords(
        longitude=((ds.longitude + 180) % 360) - 180,
    ).sortby("longitude")
    return ds


def _drop_cds_artefacts(ds: xr.Dataset) -> xr.Dataset:
    """Remove CDS artefact variables (number, expver)."""
    to_drop = [v for v in ("number", "expver")
               if v in ds.coords or v in ds.data_vars]
    if to_drop:
        ds = ds.drop_vars(to_drop, errors="ignore")
    return ds


def _ensure_valid_time(ds: xr.Dataset) -> xr.Dataset:
    """Normalize time dimension name to 'valid_time'."""
    if "valid_time" in ds.coords:
        return ds
    if "time" in ds.coords:
        return ds.rename({"time": "valid_time"})
    raise KeyError("Neither 'valid_time' nor 'time' coordinate present.")
