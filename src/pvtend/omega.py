"""Hoskins (1978) Q-vector QG omega equation solver.

Solves the quasi-geostrophic omega equation:

    ∇²_p ω + (f²₀/σ) ∂²ω/∂p² = -2∇·Q - β(R_d/σp)(∂T/∂x)

Q-vector:
    Q = -(R_d/σp) * [(∂Vg/∂x · ∇T), (∂Vg/∂y · ∇T)]

Solver: FFT in longitude (periodic) + Thomas tridiagonal in pressure.
BCs: ω = 0 at top and bottom pressure levels.
Latitude taper: QG invalid below ~15°N.

References:
    Hoskins B J, Draghici I, Davies H C (1978). A new look at the
    ω-equation. Q.J.R. Meteorol. Soc., 104, 31-38.
    Bluestein H B (1992). Synoptic-Dynamic Meteorology in Midlatitudes,
    Vol. II, eq. 5.7.54.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .constants import R_EARTH, OMEGA_E, R_DRY, SIGMA0_CONST

# Constants for geostrophic wind
GEO_SMOOTH_SIGMA: float = 1.5   # Gaussian sigma in grid points
F_MIN_LAT: float = 5.0          # degrees — regularize |f| >= f(F_MIN_LAT)

# Latitude taper constants for QG validity
LAT_QG_LO: float = 15.0    # below this: omega_dry ≡ 0
LAT_QG_HI: float = 25.0    # above this: full weight
LAT_QG_POLAR: float = 80.0  # above this: taper to 0


def gaussian_smooth_2d(
    field_2d: np.ndarray,
    sigma: float = GEO_SMOOTH_SIGMA,
) -> np.ndarray:
    """Gaussian-smooth a 2-D field, handling NaNs by normalised convolution.

    Args:
        field_2d: Input 2D field, shape ``(nlat, nlon)``.
        sigma: Gaussian kernel sigma in grid points.

    Returns:
        Smoothed field with the same shape as *field_2d*.
    """
    mask = np.isnan(field_2d)
    filled = field_2d.copy()
    filled[mask] = 0.0
    weights = np.ones_like(field_2d)
    weights[mask] = 0.0
    s_field = gaussian_filter(filled, sigma=sigma, mode="wrap")
    s_weight = gaussian_filter(weights, sigma=sigma, mode="wrap")
    s_weight[s_weight < 1e-10] = np.nan
    return s_field / s_weight


def compute_geostrophic_wind(
    phi_3d: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    sigma_smooth: float = GEO_SMOOTH_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute geostrophic wind (u_g, v_g) from geopotential Φ.

    .. math::

        u_g = -(1/f) \\partial\\Phi/\\partial y, \\quad
        v_g =  (1/f) \\partial\\Phi/\\partial x

    Args:
        phi_3d: Geopotential [m² s⁻²], shape ``(nlev, nlat, nlon)``.
        lat: Latitude in degrees, ascending, shape ``(nlat,)``.
        lon: Longitude in degrees, shape ``(nlon,)``.
        sigma_smooth: Gaussian smoothing sigma in grid points (0 = none).

    Returns:
        ``(u_g, v_g)`` — geostrophic wind [m s⁻¹], each ``(nlev, nlat, nlon)``.
    """
    from .derivatives import gradient_periodic

    nlev, nlat, nlon = phi_3d.shape
    lat_rad = np.deg2rad(lat)
    f_arr = 2 * OMEGA_E * np.sin(lat_rad)
    f_min = 2 * OMEGA_E * np.sin(np.deg2rad(F_MIN_LAT))
    f_arr = np.where(
        np.abs(f_arr) < f_min,
        np.sign(f_arr + 1e-30) * f_min,
        f_arr,
    )

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
            phi_k = gaussian_smooth_2d(phi_k, sigma=sigma_smooth)
        dphi_dx, dphi_dy = gradient_periodic(phi_k, dx_arr, dy)
        for j in range(nlat):
            u_g[k, j, :] = -dphi_dy[j, :] / f_arr[j]
            v_g[k, j, :] = dphi_dx[j, :] / f_arr[j]

    return u_g, v_g


def _thomas_batch(
    a_1d: np.ndarray,
    b_2d: np.ndarray,
    c_1d: np.ndarray,
    d_2d: np.ndarray,
) -> np.ndarray:
    """Batched Thomas tridiagonal solver over wavenumber dimension.

    Solves the tridiagonal system ``A x = d`` for each wavenumber
    simultaneously.

    Args:
        a_1d: Sub-diagonal, shape ``(n,)``.
        b_2d: Main diagonal, shape ``(n, M)`` — varies with wavenumber.
        c_1d: Super-diagonal, shape ``(n,)``.
        d_2d: RHS, shape ``(n, M)``.

    Returns:
        Solution ``x``, shape ``(n, M)``.
    """
    n = b_2d.shape[0]
    cp = np.zeros_like(b_2d)
    dp = np.zeros_like(d_2d)

    denom = np.where(np.abs(b_2d[0]) < 1e-30, 1e-30, b_2d[0])
    cp[0] = c_1d[0] / denom
    dp[0] = d_2d[0] / denom

    for i in range(1, n):
        denom = b_2d[i] - a_1d[i] * cp[i - 1]
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        if i < n - 1:
            cp[i] = c_1d[i] / denom
        dp[i] = (d_2d[i] - a_1d[i] * dp[i - 1]) / denom

    x = np.zeros_like(d_2d)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def lat_taper(lat: np.ndarray) -> np.ndarray:
    """Compute combined latitude taper for QG validity.

    Linearly ramps from 0 at ``LAT_QG_LO`` to 1 at ``LAT_QG_HI``.
    Tapers back to 0 above ``LAT_QG_POLAR`` (grid singularity).

    Args:
        lat: Latitude array in degrees.

    Returns:
        Taper weights in [0, 1], same shape as *lat*.
    """
    taper_lo = np.clip(
        (lat - LAT_QG_LO) / (LAT_QG_HI - LAT_QG_LO), 0.0, 1.0
    )
    taper_hi = np.clip((LAT_QG_POLAR - lat) / 5.0, 0.0, 1.0)
    return taper_lo * taper_hi


def solve_qg_omega(
    u: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    plevs_pa: np.ndarray,
    *,
    use_constant_sigma: bool = True,
    sigma0_const: float = SIGMA0_CONST,
) -> np.ndarray:
    """Solve the QG omega equation for omega_dry on the full NH grid.

    Full implementation of the Hoskins Q-vector formulation using FFT in
    longitude (periodic BC) and a Thomas tridiagonal solver in pressure.
    Boundary conditions: ω = 0 at top and bottom pressure levels.

    Args:
        u: Geostrophic zonal wind [m s⁻¹], shape ``(nlev, nlat, nlon)``.
        v: Geostrophic meridional wind [m s⁻¹], shape ``(nlev, nlat, nlon)``.
        t: Temperature [K], shape ``(nlev, nlat, nlon)``.
        lat: Ascending latitude [degrees], shape ``(nlat,)``.
        lon: Longitude [degrees], shape ``(nlon,)``.
        plevs_pa: Pressure [Pa], ascending, shape ``(nlev,)``.
        use_constant_sigma: If True, use constant σ₀ = *sigma0_const*.
        sigma0_const: Constant static stability [m² Pa⁻² s⁻²].

    Returns:
        ``omega_dry`` — QG vertical velocity [Pa s⁻¹],
        shape ``(nlev, nlat, nlon)``.
    """
    from .derivatives import gradient_periodic

    nlev, nlat, nlon = u.shape
    lat_rad = np.deg2rad(lat)
    dlat = np.abs(lat[1] - lat[0]) if nlat > 1 else 1.5
    dlon = np.abs(lon[1] - lon[0]) if nlon > 1 else 1.5
    dy = np.deg2rad(dlat) * R_EARTH
    dx_arr = np.deg2rad(dlon) * R_EARTH * np.cos(lat_rad)
    dx_arr = np.maximum(dx_arr, dy * 0.1)

    # Coriolis and beta at each latitude
    f_min = 2 * OMEGA_E * np.sin(np.deg2rad(F_MIN_LAT))
    f_arr = 2 * OMEGA_E * np.sin(lat_rad)
    f_arr = np.sign(f_arr) * np.maximum(np.abs(f_arr), f_min)
    beta_arr = 2 * OMEGA_E * np.cos(lat_rad) / R_EARTH

    # Static stability profile
    kappa = R_DRY / 1004.0
    if use_constant_sigma:
        sigma0 = np.full(nlev, sigma0_const)
    else:
        T_mean = np.nanmean(t, axis=(1, 2))
        theta_m = T_mean * (1e5 / plevs_pa) ** kappa
        sigma0 = np.zeros(nlev)
        for k in range(1, nlev - 1):
            dp_s = plevs_pa[k + 1] - plevs_pa[k - 1]
            dlnt = np.log(theta_m[k + 1]) - np.log(theta_m[k - 1])
            sigma0[k] = -(R_DRY * T_mean[k] / plevs_pa[k]) * (dlnt / dp_s)
        sigma0[0] = sigma0[1]
        sigma0[-1] = sigma0[-2]
        sigma0 = np.maximum(sigma0, 1e-7)

    # Latitude taper for QG validity
    lat_taper_full = lat_taper(lat)
    u_tapered = u * lat_taper_full[None, :, None]
    v_tapered = v * lat_taper_full[None, :, None]

    # Temperature gradients
    dT_dx = np.zeros_like(t)
    dT_dy = np.zeros_like(t)
    for k in range(nlev):
        dT_dx[k], dT_dy[k] = gradient_periodic(t[k], dx_arr, dy)

    # Wind gradients
    dug_dx = np.zeros_like(u)
    dug_dy = np.zeros_like(u)
    dvg_dx = np.zeros_like(v)
    dvg_dy = np.zeros_like(v)
    for k in range(nlev):
        dug_dx[k], dug_dy[k] = gradient_periodic(u_tapered[k], dx_arr, dy)
        dvg_dx[k], dvg_dy[k] = gradient_periodic(v_tapered[k], dx_arr, dy)

    # Q-vector divergence and beta term → RHS
    rhs_qv = np.zeros_like(u)
    for k in range(nlev):
        coef = -R_DRY / (sigma0[k] * plevs_pa[k])
        Q1 = coef * (dug_dx[k] * dT_dx[k] + dvg_dx[k] * dT_dy[k])
        Q2 = coef * (dug_dy[k] * dT_dx[k] + dvg_dy[k] * dT_dy[k])
        Q1 *= lat_taper_full[:, None]
        Q2 *= lat_taper_full[:, None]
        dQ1_dx, _ = gradient_periodic(Q1, dx_arr, dy)
        _, dQ2_dy = gradient_periodic(Q2, dx_arr, dy)
        div_Q = dQ1_dx + dQ2_dy
        beta_term = (
            beta_arr[:, None]
            * (R_DRY / (sigma0[k] * plevs_pa[k]))
            * dT_dx[k]
        )
        beta_term *= lat_taper_full[:, None]
        rhs_qv[k] = -2.0 * div_Q - beta_term

    # --- FFT solve: spectral in longitude, Thomas in pressure ---
    rhs_hat = np.fft.rfft(rhs_qv, axis=-1)
    omega_hat = np.zeros_like(rhs_hat)
    n_int = nlev - 2  # interior pressure levels (exclude top/bottom BC)
    if n_int < 1:
        return np.zeros_like(u)

    dp_diff = np.diff(plevs_pa)
    f2 = f_arr ** 2
    nfreq = rhs_hat.shape[-1]
    m_arr = np.arange(nfreq, dtype=float)
    cos2 = np.maximum(np.cos(lat_rad) ** 2, 1e-6)
    kx2 = m_arr[None, :] ** 2 / (R_EARTH ** 2 * cos2[:, None])

    # Tridiagonal coefficients (vertical)
    hm_k = dp_diff[:n_int]
    hp_k = dp_diff[1 : n_int + 1]
    coef_k = 2.0 / (hm_k + hp_k)
    sigma0_int = sigma0[1 : nlev - 1]

    for j in range(nlat):
        if lat[j] < LAT_QG_LO:
            continue
        f2j = f2[j]
        f2_over_sigma = f2j / sigma0_int

        a_1d = f2_over_sigma * coef_k / hm_k
        c_1d = f2_over_sigma * coef_k / hp_k
        b_base = -f2_over_sigma * coef_k * (1.0 / hm_k + 1.0 / hp_k)
        b_2d = b_base[:, None] - kx2[j][None, :]

        d_2d = rhs_hat[1 : nlev - 1, j, :]
        omega_hat[1 : nlev - 1, j, :] = _thomas_batch(
            a_1d, b_2d, c_1d, d_2d
        )

    # Zero the zonal-mean mode (k=0) to remove indeterminate constant
    omega_hat[:, :, 0] = 0.0

    # Back to physical space
    omega_dry = np.fft.irfft(omega_hat, n=nlon, axis=-1)
    omega_dry *= lat_taper_full[None, :, None]
    return omega_dry
