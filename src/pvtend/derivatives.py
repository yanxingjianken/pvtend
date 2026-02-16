"""Spatial and temporal derivative operators on lat-lon-pressure grids.

All operators respect:
- Periodic zonal boundary conditions (globe wraps in longitude)
- One-sided differences at polar boundaries (meridional)
- Non-uniform pressure level spacing (pressure derivatives)

Two interfaces are provided:
1. NumPy functions for raw arrays: ``ddx``, ``ddy``, ``ddp``, ``ddt``
2. xarray wrappers: ``ddx_da``, ``ddy_da``, ``ddp_da``, ``ddt_da``
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .constants import R_EARTH


# ═══════════════════════════════════════════════════════════════
#  NumPy array operators
# ═══════════════════════════════════════════════════════════════


def ddx(
    field: np.ndarray,
    dx_arr: np.ndarray,
    periodic: bool = True,
) -> np.ndarray:
    """Zonal derivative ∂f/∂x with periodic wrapping.

    Uses centred finite differences in the interior.  At the zonal
    boundaries the field wraps around the globe when *periodic* is
    True (default); otherwise one-sided differences are applied.

    Args:
        field: 2-D array ``(nlat, nlon)`` or 3-D ``(nlev, nlat, nlon)``.
        dx_arr: Zonal grid spacing per latitude [m], shape ``(nlat,)``.
        periodic: If True, wrap zonally (col 0 ↔ col -1).

    Returns:
        Same shape as *field*, in units of ``[field_units / m]``.
    """
    if field.ndim == 3:
        return np.stack(
            [ddx(field[k], dx_arr, periodic) for k in range(field.shape[0])]
        )

    nlat, nlon = field.shape
    out = np.empty_like(field)

    for j in range(nlat):
        # Interior: centred differences
        out[j, 1:-1] = (field[j, 2:] - field[j, :-2]) / (2 * dx_arr[j])

        if periodic:
            out[j, 0] = (field[j, 1] - field[j, -1]) / (2 * dx_arr[j])
            out[j, -1] = (field[j, 0] - field[j, -2]) / (2 * dx_arr[j])
        else:
            out[j, 0] = (field[j, 1] - field[j, 0]) / dx_arr[j]
            out[j, -1] = (field[j, -1] - field[j, -2]) / dx_arr[j]

    return out


def ddy(field: np.ndarray, dy: float) -> np.ndarray:
    """Meridional derivative ∂f/∂y with one-sided differences at boundaries.

    Uses centred differences in the interior and forward/backward
    differences at the first/last latitude row.

    Args:
        field: 2-D ``(nlat, nlon)`` or 3-D ``(nlev, nlat, nlon)``.
        dy: Meridional grid spacing [m].

    Returns:
        Same shape as *field*, in ``[field_units / m]``.
    """
    if field.ndim == 3:
        return np.stack(
            [ddy(field[k], dy) for k in range(field.shape[0])]
        )

    out = np.empty_like(field)
    out[1:-1] = (field[2:] - field[:-2]) / (2 * dy)
    out[0] = (field[1] - field[0]) / dy
    out[-1] = (field[-1] - field[-2]) / dy
    return out


def ddp(field: np.ndarray, plevs_pa: np.ndarray) -> np.ndarray:
    """Pressure derivative ∂f/∂p for non-uniform levels.

    Uses centred differences in the interior and one-sided differences
    at the top/bottom pressure boundaries.

    Args:
        field: Array with pressure as axis 0, shape ``(nlev, ...)``.
        plevs_pa: Pressure levels in Pa, shape ``(nlev,)``.

    Returns:
        Same shape as *field*, in ``[field_units / Pa]``.
    """
    nlev = field.shape[0]
    out = np.zeros_like(field)

    # Interior: centred differences
    for k in range(1, nlev - 1):
        dp = plevs_pa[k + 1] - plevs_pa[k - 1]
        out[k] = (field[k + 1] - field[k - 1]) / dp

    # Boundaries: one-sided differences
    dp0 = plevs_pa[1] - plevs_pa[0]
    dpn = plevs_pa[-1] - plevs_pa[-2]
    out[0] = (field[1] - field[0]) / dp0
    out[-1] = (field[-1] - field[-2]) / dpn

    return out


def ddt(field: np.ndarray, dt_s: float) -> np.ndarray:
    """Time derivative ∂f/∂t via centred differences.

    Uses centred differences in the interior and one-sided differences
    at the first and last time steps.

    Args:
        field: Array with time as axis 0, shape ``(nt, ...)``.
        dt_s: Time step in seconds.

    Returns:
        Same shape as *field*, in ``[field_units / s]``.
    """
    nt = field.shape[0]
    out = np.zeros_like(field)
    out[1:-1] = (field[2:] - field[:-2]) / (2 * dt_s)
    out[0] = (field[1] - field[0]) / dt_s
    out[-1] = (field[-1] - field[-2]) / dt_s
    return out


def gradient_periodic(
    phi: np.ndarray,
    dx_arr: np.ndarray,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal gradient with periodic zonal wrapping.

    Convenience wrapper combining :func:`ddx` (periodic) and :func:`ddy`.

    Args:
        phi: 2-D field ``(nlat, nlon)``.
        dx_arr: Zonal spacing per latitude [m], shape ``(nlat,)``.
        dy: Meridional spacing [m].

    Returns:
        Tuple ``(dphi_dx, dphi_dy)``, each ``(nlat, nlon)``.
    """
    return ddx(phi, dx_arr, periodic=True), ddy(phi, dy)


# ═══════════════════════════════════════════════════════════════
#  xarray DataArray wrappers
# ═══════════════════════════════════════════════════════════════


def ddx_da(da: xr.DataArray, lon_name: str = "longitude") -> xr.DataArray:
    """Periodic zonal derivative via ``DataArray.roll()``.

    Uses centred differences with periodic roll so that the first and
    last longitude columns reference each other.

    Args:
        da: xarray DataArray with a longitude dimension.
        lon_name: Name of the longitude dimension.

    Returns:
        ``d(da)/dx`` in SI units ``[field / m]``.
    """
    lon_vals = da[lon_name].values.astype(float)
    dlon_deg = float(np.nanmean(np.diff(lon_vals)))

    # Compute dx from latitude
    lat_name = _find_lat_dim(da)
    lat_rad = np.deg2rad(da[lat_name].values)
    dx_m = np.abs(np.deg2rad(dlon_deg)) * R_EARTH * np.cos(lat_rad)

    # Broadcast dx_m to match da's dims
    dx_da = xr.DataArray(dx_m, dims=[lat_name], coords={lat_name: da[lat_name]})

    return (
        da.roll({lon_name: -1}, roll_coords=False)
        - da.roll({lon_name: 1}, roll_coords=False)
    ) / (2.0 * dx_da)


def ddy_da(da: xr.DataArray, lat_name: str = "latitude") -> xr.DataArray:
    """Meridional derivative ``d()/dy`` in SI units.

    Uses ``xr.DataArray.differentiate`` along latitude, then converts
    from degrees to metres.

    Args:
        da: xarray DataArray with a latitude dimension.
        lat_name: Name of the latitude dimension.

    Returns:
        ``d(da)/dy`` in ``[field / m]``.
    """
    dlat = float(np.abs(np.diff(da[lat_name].values).mean()))
    dy_m = np.deg2rad(dlat) * R_EARTH
    return da.differentiate(lat_name) / dy_m


def ddp_da(da: xr.DataArray, p_name: str = "pressure_level") -> xr.DataArray:
    """Pressure derivative ``d()/dp`` where coordinate is in hPa.

    Uses ``xr.DataArray.differentiate`` along the pressure coordinate,
    then converts from hPa to Pa (factor 100).

    Args:
        da: xarray DataArray with a pressure level dimension.
        p_name: Name of the pressure dimension.

    Returns:
        ``d(da)/dp`` in ``[field / Pa]``.
    """
    return da.differentiate(p_name) / 100.0  # hPa → Pa


def ddt_da(da: xr.DataArray, t_name: str = "valid_time") -> xr.DataArray:
    """Time derivative ``d()/dt`` in s⁻¹.

    Uses ``xr.DataArray.differentiate`` with ``datetime_unit='s'``.

    Args:
        da: xarray DataArray with a time dimension.
        t_name: Name of the time dimension.

    Returns:
        ``d(da)/dt`` in ``[field / s]``.
    """
    return da.differentiate(coord=t_name, datetime_unit="s")


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _find_lat_dim(da: xr.DataArray) -> str:
    """Auto-detect latitude dimension name.

    Args:
        da: xarray DataArray to inspect.

    Returns:
        The name of the latitude dimension.

    Raises:
        ValueError: If no dimension containing ``'lat'`` is found.
    """
    for name in da.dims:
        if "lat" in name.lower():
            return name
    raise ValueError(f"No latitude dimension found in {da.dims}")
