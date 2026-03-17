"""Moist/dry omega decomposition and divergent wind recovery.

Decomposes total vertical velocity (omega) into dry and moist components:

    ω_total  = ω_dry + ω_moist
    ω_dry    = QG omega (from solve_qg_omega)
    ω_moist  = ω_total − ω_dry

Each divergent wind component (moist, dry, qg-moist) is recovered
**independently** via its own Poisson inversion:

    ∇²χ_X = −∂ω_X/∂p
    (u_div_X, v_div_X) = ∇χ_X

where X ∈ {moist, dry, qg_moist}.

Total-field approximation
~~~~~~~~~~~~~~~~~~~~~~~~~
The QG omega solve and Poisson inversion are performed on **total
fields** (ω, not ω'), exploiting |ω'| >> |ω̄| in midlatitude synoptic
systems.  Because the climatological mean vertical velocity is
negligibly small compared with the anomaly, the total-field moist
omega closely approximates the anomaly moist omega:

    ω_moist ≈ ω'_moist

The same linear approximation propagates through the Poisson inversion
to the horizontal divergent wind:

    u_div_moist ≈ u'_div_moist
"""

from __future__ import annotations

import numpy as np

from .constants import R_EARTH
from .derivatives import ddp
from .helmholtz import gradient, solve_poisson_spherical_fft


def solve_chi_from_omega(
    omega: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    plevs_pa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve ∇²χ = -∂ω/∂p at each level (spherical Laplacian).

    Computes the velocity potential of the divergent wind associated
    with *omega* by solving a Poisson equation on each pressure level
    using the full spherical Laplacian (conservative form).

    The area-weighted mean of the RHS is removed on each level before
    solving to ensure compatibility with the Dirichlet boundary
    conditions (Fredholm solvability).

    Args:
        omega: Vertical velocity component [Pa/s],
            shape ``(nlev, nlat, nlon)``.
        lat: Latitude [degrees], ascending, shape ``(nlat,)``.
        lon: Longitude [degrees], shape ``(nlon,)``.
        plevs_pa: Pressure levels [Pa], ascending, shape ``(nlev,)``.

    Returns:
        Tuple of ``(chi, u_div, v_div)``, each with
        shape ``(nlev, nlat, nlon)``.

    Raises:
        ValueError: If ``omega`` is not 3-D.
    """
    if omega.ndim != 3:
        raise ValueError(
            f"omega must be 3-D (nlev, nlat, nlon), got {omega.ndim}-D"
        )

    nlev, nlat, nlon = omega.shape
    lat_rad = np.deg2rad(lat)
    dlat = np.abs(lat[1] - lat[0]) if nlat > 1 else 1.5
    dlon = np.abs(lon[1] - lon[0]) if nlon > 1 else 1.5
    dy = np.deg2rad(dlat) * R_EARTH
    dx_arr = np.deg2rad(dlon) * R_EARTH * np.cos(lat_rad)
    dx_arr = np.maximum(dx_arr, dy * 0.1)  # guard near poles
    dlon_rad = np.deg2rad(dlon)

    chi_out = np.zeros_like(omega)
    u_div_out = np.zeros_like(omega)
    v_div_out = np.zeros_like(omega)

    # ── Compute ∂ω/∂p using centred finite differences ──
    domega_dp = ddp(omega, plevs_pa)

    # RHS of Poisson equation: -∂ω/∂p
    rhs_poisson = -domega_dp

    # ── Area-weighted mean removal for RHS compatibility ──
    cos_phi = np.cos(lat_rad)  # (nlat,)
    area_weights = cos_phi / cos_phi.sum()  # normalised weights
    for k in range(nlev):
        weighted_mean = np.sum(
            area_weights[:, None] * rhs_poisson[k]
        ) / nlon
        rhs_poisson[k] -= weighted_mean

    # ── Solve level-by-level with spherical Laplacian ──
    for k in range(nlev):
        chi_k = solve_poisson_spherical_fft(
            rhs_poisson[k], lat, dy, dlon_rad, R_earth=R_EARTH
        )
        chi_out[k] = chi_k
        dchi_dx, dchi_dy = gradient(chi_k, dx_arr, dy)
        u_div_out[k] = dchi_dx
        v_div_out[k] = dchi_dy

    return chi_out, u_div_out, v_div_out


# Backward-compatible alias
solve_chi_moist = solve_chi_from_omega


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
    2. Solves independently for **both** moist and dry velocity
       potentials via Poisson equation on each pressure level.
    3. Recovers the divergent wind for each component as the
       gradient of its respective velocity potential.

    Args:
        omega_total: Total vertical velocity [Pa/s],
            shape ``(nlev, nlat, nlon)``.
        omega_dry: QG omega (dry component) [Pa/s],
            shape ``(nlev, nlat, nlon)``.
        lat: Latitude [degrees], ascending, shape ``(nlat,)``.
        lon: Longitude [degrees], shape ``(nlon,)``.
        plevs_pa: Pressure levels [Pa], ascending, shape ``(nlev,)``.
        u_div: Total divergent u-component [m/s] (optional),
            shape ``(nlev, nlat, nlon)``.  No longer used for dry
            residual computation but accepted for API compatibility.
        v_div: Total divergent v-component [m/s] (optional),
            shape ``(nlev, nlat, nlon)``.  No longer used for dry
            residual computation but accepted for API compatibility.

    Returns:
        Dictionary containing:

        - ``"omega_moist"``: Moist omega residual.
        - ``"chi_moist"``: Moist velocity potential.
        - ``"u_div_moist"``: Moist divergent u-component.
        - ``"v_div_moist"``: Moist divergent v-component.
        - ``"chi_dry"``: Dry velocity potential.
        - ``"u_div_dry"``: Dry divergent u-component.
        - ``"v_div_dry"``: Dry divergent v-component.
    """
    omega_moist = omega_total - omega_dry

    chi_m, u_div_m, v_div_m = solve_chi_from_omega(
        omega_moist, lat, lon, plevs_pa
    )
    chi_d, u_div_d, v_div_d = solve_chi_from_omega(
        omega_dry, lat, lon, plevs_pa
    )

    return {
        "omega_moist": omega_moist,
        "chi_moist": chi_m,
        "u_div_moist": u_div_m,
        "v_div_moist": v_div_m,
        "chi_dry": chi_d,
        "u_div_dry": u_div_d,
        "v_div_dry": v_div_d,
    }
