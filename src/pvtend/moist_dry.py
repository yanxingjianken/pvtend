"""Moist/dry omega decomposition and divergent wind recovery.

Decomposes total vertical velocity (omega) into dry and moist components:

    ω_total  = ω_dry + ω_moist
    ω_dry    = QG omega (from solve_qg_omega)
    ω_moist  = ω_total − ω_dry

The moist divergent wind is recovered via:
    ∇²χ_moist = −∂ω_moist/∂p
    (u_div_moist, v_div_moist) = ∇χ_moist

The dry divergent wind is the residual:
    u_div_dry = u_div − u_div_moist
"""

from __future__ import annotations

import numpy as np

from .constants import R_EARTH
from .helmholtz import gradient, solve_poisson_fft


def solve_chi_moist(
    omega_moist: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    plevs_pa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve ∇²χ_moist = -∂ω_moist/∂p at each level.

    Computes the velocity potential of the moist divergent wind by
    solving a Poisson equation on each pressure level.  The RHS is
    the negative vertical derivative of the moist omega field.

    Args:
        omega_moist: Moist omega [Pa/s], shape ``(nlev, nlat, nlon)``.
        lat: Latitude [degrees], ascending, shape ``(nlat,)``.
        lon: Longitude [degrees], shape ``(nlon,)``.
        plevs_pa: Pressure levels [Pa], ascending, shape ``(nlev,)``.

    Returns:
        Tuple of ``(chi_moist, u_div_moist, v_div_moist)``, each with
        shape ``(nlev, nlat, nlon)``.

    Raises:
        ValueError: If ``omega_moist`` is not 3-D.
    """
    if omega_moist.ndim != 3:
        raise ValueError(
            f"omega_moist must be 3-D (nlev, nlat, nlon), got {omega_moist.ndim}-D"
        )

    nlev, nlat, nlon = omega_moist.shape
    lat_rad = np.deg2rad(lat)
    dlat = np.abs(lat[1] - lat[0]) if nlat > 1 else 1.5
    dlon = np.abs(lon[1] - lon[0]) if nlon > 1 else 1.5
    dy = np.deg2rad(dlat) * R_EARTH
    dx_arr = np.deg2rad(dlon) * R_EARTH * np.cos(lat_rad)
    dx_arr = np.maximum(dx_arr, dy * 0.1)  # guard near poles

    chi_m = np.zeros_like(omega_moist)
    u_div_m = np.zeros_like(omega_moist)
    v_div_m = np.zeros_like(omega_moist)

    # ── Compute ∂ω_moist/∂p using centred finite differences ──
    domega_dp = np.zeros_like(omega_moist)
    dp = np.diff(plevs_pa)

    # Interior levels: centred difference
    for k in range(1, nlev - 1):
        domega_dp[k] = (omega_moist[k + 1] - omega_moist[k - 1]) / (
            plevs_pa[k + 1] - plevs_pa[k - 1]
        )

    # Boundary levels: one-sided difference
    if nlev > 1:
        domega_dp[0] = (omega_moist[1] - omega_moist[0]) / dp[0]
        domega_dp[-1] = (omega_moist[-1] - omega_moist[-2]) / dp[-1]

    # RHS of Poisson equation: -∂ω_moist/∂p
    rhs_poisson = -domega_dp

    # ── Solve level-by-level ──
    for k in range(nlev):
        chi_k = solve_poisson_fft(rhs_poisson[k], dx_arr, dy)
        chi_m[k] = chi_k
        dchi_dx, dchi_dy = gradient(chi_k, dx_arr, dy)
        u_div_m[k] = dchi_dx
        v_div_m[k] = dchi_dy

    return chi_m, u_div_m, v_div_m


def decompose_omega(
    omega_total: np.ndarray,
    omega_dry: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    plevs_pa: np.ndarray,
    u_div: np.ndarray | None = None,
    v_div: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Full moist/dry omega decomposition with divergent wind recovery.

    Given total omega and its QG (dry) component, this function:

    1. Computes the moist residual: ``ω_moist = ω_total − ω_dry``.
    2. Solves for the moist velocity potential ``χ_moist`` via Poisson
       equation on each pressure level.
    3. Recovers the moist divergent wind ``(u_div_moist, v_div_moist)``
       as the gradient of ``χ_moist``.
    4. Optionally decomposes the total divergent wind into moist and dry
       parts by subtraction.

    Args:
        omega_total: Total vertical velocity [Pa/s],
            shape ``(nlev, nlat, nlon)``.
        omega_dry: QG omega (dry component) [Pa/s],
            shape ``(nlev, nlat, nlon)``.
        lat: Latitude [degrees], ascending, shape ``(nlat,)``.
        lon: Longitude [degrees], shape ``(nlon,)``.
        plevs_pa: Pressure levels [Pa], ascending, shape ``(nlev,)``.
        u_div: Total divergent u-component [m/s] (optional),
            shape ``(nlev, nlat, nlon)``.
        v_div: Total divergent v-component [m/s] (optional),
            shape ``(nlev, nlat, nlon)``.

    Returns:
        Dictionary containing:

        - ``"omega_moist"``: Moist omega residual.
        - ``"chi_moist"``: Moist velocity potential.
        - ``"u_div_moist"``: Moist divergent u-component.
        - ``"v_div_moist"``: Moist divergent v-component.
        - ``"u_div_dry"``: Dry divergent u (only if *u_div* provided).
        - ``"v_div_dry"``: Dry divergent v (only if *v_div* provided).
    """
    omega_moist = omega_total - omega_dry
    chi_m, u_div_m, v_div_m = solve_chi_moist(
        omega_moist, lat, lon, plevs_pa
    )

    result: dict[str, np.ndarray] = {
        "omega_moist": omega_moist,
        "chi_moist": chi_m,
        "u_div_moist": u_div_m,
        "v_div_moist": v_div_m,
    }

    if u_div is not None:
        result["u_div_dry"] = u_div - u_div_m
    if v_div is not None:
        result["v_div_dry"] = v_div - v_div_m

    return result
