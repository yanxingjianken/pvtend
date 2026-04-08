"""Real-data u_div additivity test using ERA5 omega.

Verifies that the Poisson solver is linear:
  u_div(ω_total) ≈ u_div(ω_adiabatic) + u_div(ω_diabatic)

for an empirical adiabatic/diabatic split (1/3 adiabatic, 2/3 diabatic).

Requires ERA5 data at /net/flood/data2/users/x_yan/era/ — skipped
in CI or when data is absent.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

ERA5_W = "/net/flood/data2/users/x_yan/era/era5_w_2010_01.nc"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(ERA5_W).exists(),
    reason="ERA5 omega file not found (requires local data)",
)


def test_poisson_linearity_real_omega():
    """u_div(ω_dry) + u_div(ω_moist) == u_div(ω_total) to machine precision."""
    from pvtend.moist_dry import solve_chi_from_omega, verify_div_additivity

    ds = xr.open_dataset(ERA5_W, engine="netcdf4")
    # Take a single timestep
    w = ds["w"].isel(valid_time=0).values  # (nlev, nlat, nlon)
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    plevs = ds["pressure_level"].values

    # Ensure ascending latitude (Poisson solver expects it)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        w = w[:, ::-1, :]

    # Ensure ascending pressure levels (Pa)
    plevs_pa = plevs * 100.0
    if plevs_pa[0] > plevs_pa[-1]:
        plevs_pa = plevs_pa[::-1]
        w = w[::-1, :, :]

    # NaN→0 guard
    w = np.nan_to_num(w, nan=0.0).astype(np.float64)

    # Empirical adiabatic/diabatic split
    w_adiabatic = w / 3.0
    w_diabatic = w - w_adiabatic

    # Independent Poisson inversions
    _, u_div_total, v_div_total = solve_chi_from_omega(w, lat, lon, plevs_pa)
    _, u_div_adiabatic, v_div_adiabatic = solve_chi_from_omega(w_adiabatic, lat, lon, plevs_pa)
    _, u_div_diabatic, v_div_diabatic = solve_chi_from_omega(w_diabatic, lat, lon, plevs_pa)

    # Verify additivity
    err_u = verify_div_additivity(u_div_total, u_div_adiabatic, u_div_diabatic)
    err_v = verify_div_additivity(v_div_total, v_div_adiabatic, v_div_diabatic)

    # Machine precision: should be < 1e-10 m/s
    assert err_u < 1e-10, f"u_div additivity error too large: {err_u:.3e}"
    assert err_v < 1e-10, f"v_div additivity error too large: {err_v:.3e}"

    ds.close()
    print(f"PASS  max|u_div_adiabatic + u_div_diabatic - u_div| = {err_u:.3e}")
    print(f"PASS  max|v_div_adiabatic + v_div_diabatic - v_div| = {err_v:.3e}")
