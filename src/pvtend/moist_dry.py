"""Adiabatic/diabatic omega decomposition and divergent wind recovery.

Decomposes total vertical velocity (omega) into adiabatic and diabatic
components:

    ω_total     = ω_adiabatic + ω_diabatic
    ω_adiabatic = QG omega (from solve_qg_omega)
    ω_diabatic  = ω_total − ω_adiabatic

Each divergent wind component (diabatic, adiabatic, qg-diabatic) is
recovered **independently** via its own Poisson inversion:

    ∇²χ_X = −∂ω_X/∂p
    (u_div_X, v_div_X) = ∇χ_X

where X ∈ {diabatic, adiabatic, qg_diabatic}.

Total-field approximation
~~~~~~~~~~~~~~~~~~~~~~~~~
The QG omega solve and Poisson inversion are performed on **total
fields** (ω, not ω'), exploiting |ω'| >> |ω̄| in midlatitude synoptic
systems.  Because the climatological mean vertical velocity is
negligibly small compared with the anomaly, the total-field diabatic
omega closely approximates the anomaly diabatic omega:

    ω_diabatic ≈ ω'_diabatic

The same linear approximation propagates through the Poisson inversion
to the horizontal divergent wind:

    u_div_diabatic ≈ u'_div_diabatic
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
    method: str = "spectral",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve ∇²χ = -∂ω/∂p at each level.

    Computes the velocity potential of the divergent wind associated
    with *omega* by solving a Poisson equation on each pressure level.

    By default (``method="spectral"``) the solve uses spherical
    harmonics via :func:`pvtend.sh_ops.invert_laplacian_sh`, which is
    pole-closed and supports auto NH→global parity mirroring (scalar
    parity for χ). Pass ``method="fd"`` to fall back to the legacy
    conservative-form spherical-FFT Poisson solver.

    Args:
        omega: Vertical velocity component [Pa/s],
            shape ``(nlev, nlat, nlon)``.
        lat: Latitude [degrees], ascending, shape ``(nlat,)``.
        lon: Longitude [degrees], shape ``(nlon,)``.
        plevs_pa: Pressure levels [Pa], ascending, shape ``(nlev,)``.
        method: ``"spectral"`` (default) or ``"fd"``.

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

    nlev, _, _ = omega.shape

    # ── Compute ∂ω/∂p using centred finite differences ──
    domega_dp = ddp(omega, plevs_pa)
    # RHS of Poisson equation: -∂ω/∂p
    rhs_poisson = -domega_dp

    if method in ("spectral", "sh"):
        from .sh_ops import invert_laplacian_sh, gradient_sh, _SPHARM_AVAILABLE
        if not _SPHARM_AVAILABLE:
            import warnings
            warnings.warn(
                "pyspharm not installed; falling back to FD for "
                "solve_chi_from_omega. Install via `pip install pvtend[sh]`.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _solve_chi_from_omega_fd(rhs_poisson, lat, lon)

        chi_out = np.zeros_like(omega)
        u_div_out = np.zeros_like(omega)
        v_div_out = np.zeros_like(omega)
        for k in range(nlev):
            chi_k = invert_laplacian_sh(
                rhs_poisson[k], lat, lon, R_earth=R_EARTH, parity="scalar",
            )
            chi_out[k] = chi_k
            dchi_dx, dchi_dy = gradient_sh(chi_k, lat, lon, R_earth=R_EARTH)
            u_div_out[k] = dchi_dx
            v_div_out[k] = dchi_dy
        return chi_out, u_div_out, v_div_out

    # ── Legacy FD path ──
    return _solve_chi_from_omega_fd(rhs_poisson, lat, lon)


def _solve_chi_from_omega_fd(
    rhs_poisson: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy spherical-FFT Poisson path (Dirichlet BCs).

    Kept for regression testing against the pre-SH baseline.
    """
    nlev, nlat, nlon = rhs_poisson.shape
    lat_rad = np.deg2rad(lat)
    dlat = np.abs(lat[1] - lat[0]) if nlat > 1 else 1.5
    dlon = np.abs(lon[1] - lon[0]) if nlon > 1 else 1.5
    dy = np.deg2rad(dlat) * R_EARTH
    dx_arr = np.deg2rad(dlon) * R_EARTH * np.cos(lat_rad)
    dx_arr = np.maximum(dx_arr, dy * 0.1)  # guard near poles
    dlon_rad = np.deg2rad(dlon)

    chi_out = np.zeros_like(rhs_poisson)
    u_div_out = np.zeros_like(rhs_poisson)
    v_div_out = np.zeros_like(rhs_poisson)

    rhs = rhs_poisson.copy()
    cos_phi = np.cos(lat_rad)
    area_weights = cos_phi / cos_phi.sum()
    for k in range(nlev):
        weighted_mean = np.sum(area_weights[:, None] * rhs[k]) / nlon
        rhs[k] -= weighted_mean

    for k in range(nlev):
        chi_k = solve_poisson_spherical_fft(
            rhs[k], lat, dy, dlon_rad, R_earth=R_EARTH
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
    omega_adiabatic: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    plevs_pa: np.ndarray,
    u_div: np.ndarray | None = None,
    v_div: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Full adiabatic/diabatic omega decomposition with divergent wind recovery.

    Given total omega and its QG (adiabatic) component, this function:

    1. Computes the diabatic residual: ``ω_diabatic = ω_total − ω_adiabatic``.
    2. Solves independently for **both** diabatic and adiabatic velocity
       potentials via Poisson equation on each pressure level.
    3. Recovers the divergent wind for each component as the
       gradient of its respective velocity potential.

    Args:
        omega_total: Total vertical velocity [Pa/s],
            shape ``(nlev, nlat, nlon)``.
        omega_adiabatic: QG omega (adiabatic component) [Pa/s],
            shape ``(nlev, nlat, nlon)``.
        lat: Latitude [degrees], ascending, shape ``(nlat,)``.
        lon: Longitude [degrees], shape ``(nlon,)``.
        plevs_pa: Pressure levels [Pa], ascending, shape ``(nlev,)``.
        u_div: Total divergent u-component [m/s] (optional),
            shape ``(nlev, nlat, nlon)``.  No longer used for adiabatic
            residual computation but accepted for API compatibility.
        v_div: Total divergent v-component [m/s] (optional),
            shape ``(nlev, nlat, nlon)``.  No longer used for adiabatic
            residual computation but accepted for API compatibility.

    Returns:
        Dictionary containing:

        - ``"omega_diabatic"``: Diabatic omega residual.
        - ``"chi_diabatic"``: Diabatic velocity potential.
        - ``"u_div_diabatic"``: Diabatic divergent u-component.
        - ``"v_div_diabatic"``: Diabatic divergent v-component.
        - ``"chi_adiabatic"``: Adiabatic velocity potential.
        - ``"u_div_adiabatic"``: Adiabatic divergent u-component.
        - ``"v_div_adiabatic"``: Adiabatic divergent v-component.
    """
    omega_diabatic = omega_total - omega_adiabatic

    chi_m, u_div_m, v_div_m = solve_chi_from_omega(
        omega_diabatic, lat, lon, plevs_pa
    )
    chi_d, u_div_d, v_div_d = solve_chi_from_omega(
        omega_adiabatic, lat, lon, plevs_pa
    )

    return {
        "omega_diabatic": omega_diabatic,
        "chi_diabatic": chi_m,
        "u_div_diabatic": u_div_m,
        "v_div_diabatic": v_div_m,
        "chi_adiabatic": chi_d,
        "u_div_adiabatic": u_div_d,
        "v_div_adiabatic": v_div_d,
    }


def verify_div_additivity(
    u_div: np.ndarray,
    u_div_adiabatic: np.ndarray,
    u_div_diabatic: np.ndarray,
) -> float:
    """Verify additive consistency of divergent wind decomposition.

    Checks that the independently Poisson-inverted adiabatic and diabatic
    divergent winds sum to the total divergent wind to machine
    precision:

        max |u_div_adiabatic + u_div_diabatic − u_div|

    Args:
        u_div: Total divergent component, any shape.
        u_div_adiabatic: Adiabatic divergent component, same shape.
        u_div_diabatic: Diabatic divergent component, same shape.

    Returns:
        Maximum absolute error of the additive split.
    """
    return float(np.max(np.abs(u_div_adiabatic + u_div_diabatic - u_div)))
