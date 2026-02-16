"""Preprocessing: load reanalysis data, crop to NH, regrid to 1.5°.

Supports ERA5 and other reanalysis products with standard CF-conventions.
Input can be any resolution >= NH coverage; output is always a regular
1.5° Northern-Hemisphere grid at hourly resolution.

Required variables: u, v, w, t, pv, z on pressure levels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import xarray as xr

from .constants import DEFAULT_LEVELS, TARGET_LAT, TARGET_LON, CLIM_VARIABLES
from .grid import crop_to_nh, bilinear_interpolate, NHGrid, default_nh_grid


def load_era5_monthly(
    data_dir: str | Path,
    year: int,
    month: int,
    variables: Sequence[str] = CLIM_VARIABLES,
    levels: Sequence[int] = DEFAULT_LEVELS,
    engine: str = "netcdf4",
) -> xr.Dataset:
    """Load ERA5 monthly files for the given year/month.

    Expects files named like ``era5_{month3}_{year}.nc`` or
    ``era5_{year}_{month:02d}.nc`` in `data_dir`.

    Parameters:
        data_dir: Directory containing ERA5 NetCDF files.
        year: Year to load.
        month: Month (1-12) to load.
        variables: Variable names to select.
        levels: Pressure levels to select [hPa].
        engine: xarray engine for reading NetCDF.

    Returns:
        xr.Dataset with dimensions (valid_time, pressure_level, latitude, longitude).
    """
    data_dir = Path(data_dir)
    # Try common naming patterns
    from .constants import MONTH_ABBREVS
    month3 = MONTH_ABBREVS[month - 1]

    candidates = [
        data_dir / f"era5_{month3}_{year}.nc",
        data_dir / f"era5_{year}_{month:02d}.nc",
        data_dir / f"era5_{year}{month:02d}.nc",
    ]

    for fpath in candidates:
        if fpath.is_file():
            ds = xr.open_dataset(fpath, engine=engine, chunks=None)
            # Normalize dimension names
            ds = _normalize_dims(ds)
            # Select levels
            plev = _plev_name(ds)
            if plev and levels:
                ds = ds.sel({plev: levels})
            return ds

    raise FileNotFoundError(
        f"No ERA5 file found for {year}-{month:02d} in {data_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def regrid_to_nh(
    ds: xr.Dataset,
    target_lat: np.ndarray = TARGET_LAT,
    target_lon: np.ndarray = TARGET_LON,
) -> xr.Dataset:
    """Regrid dataset to standard NH grid via bilinear interpolation.

    Parameters:
        ds: Input dataset with (latitude, longitude) dimensions.
        target_lat: Target latitude array.
        target_lon: Target longitude array.

    Returns:
        Regridded dataset on the target NH grid.
    """
    src_lat = ds.latitude.values
    src_lon = ds.longitude.values

    # Crop to NH first
    nh_mask = src_lat >= -5  # include a small buffer below equator
    if not nh_mask.all():
        ds = ds.isel(latitude=nh_mask)
        src_lat = ds.latitude.values

    out_vars = {}
    for vname in ds.data_vars:
        da = ds[vname]
        lat_dim = [d for d in da.dims if "lat" in d.lower()]
        if not lat_dim:
            out_vars[vname] = da
            continue

        data = da.values
        regridded = bilinear_interpolate(src_lat, src_lon, data,
                                         target_lat, target_lon)

        # Replace lat/lon dims
        new_dims = []
        for d in da.dims:
            if "lat" in d.lower():
                new_dims.append("latitude")
            elif "lon" in d.lower():
                new_dims.append("longitude")
            else:
                new_dims.append(d)

        coords = {d: da.coords[d].values for d in da.dims
                  if "lat" not in d.lower() and "lon" not in d.lower()}
        coords["latitude"] = target_lat
        coords["longitude"] = target_lon
        out_vars[vname] = xr.DataArray(regridded, dims=new_dims, coords=coords)

    return xr.Dataset(out_vars, attrs=ds.attrs)


def _normalize_dims(ds: xr.Dataset) -> xr.Dataset:
    """Normalize dimension names to standard CF names."""
    rename = {}
    for dim in ds.dims:
        dl = dim.lower()
        if "lat" in dl and dim != "latitude":
            rename[dim] = "latitude"
        elif "lon" in dl and dim != "longitude":
            rename[dim] = "longitude"
        elif dl in ("time", "valid_time") and dim != "valid_time":
            rename[dim] = "valid_time"
        elif ("pressure" in dl or "level" in dl or "plev" in dl) and dim != "pressure_level":
            rename[dim] = "pressure_level"
    if rename:
        ds = ds.rename(rename)
    return ds


def _plev_name(ds: xr.Dataset) -> Optional[str]:
    """Auto-detect pressure level dimension name.

    Parameters:
        ds: Dataset to inspect.

    Returns:
        Name of the pressure level dimension, or None if not found.
    """
    for name in ("pressure_level", "level", "plev", "lev", "isobaricInhPa"):
        if name in ds.dims or name in ds.coords:
            return name
    return None
