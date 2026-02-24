"""Hoskins (1978) Q-vector QG omega equation solver.

Solves the quasi-geostrophic omega equation:

    ∇²_p ω + (f²₀/σ) ∂²ω/∂p² = -2∇·Q - β(R_d/σp)(∂T/∂x)

Q-vector:
    Q = -(R_d/σp) * [(∂Vg/∂x · ∇T), (∂Vg/∂y · ∇T)]

Three solver methods are available:

- **sp19** (default): Empirical scaling ω_dry = 1/3 ω_total following
  Steinfeld & Pfahl (2019).  Zero cost, structural consistency with
  ω_total.  Requires *omega_total* as input.

- **fft**: FFT in longitude (periodic) + Thomas tridiagonal in
  pressure.  Drops ∂²ω/∂y² but retains ∂²ω/∂x² via spectral
  representation.  Fast (~2 s) and captures >90 % of spatial variance.

- **log20**: Strongly Implicit Procedure (SIP, Stone 1968), full 3-D
  spherical stencil with tan(φ) metric term.  Closest analogue to
  Li & O'Gorman (2020).  Numba-accelerated, ~3–6 s per event pair.

BCs: ω = 0 at top and bottom pressure levels.
Latitude taper: QG invalid below ~15°N.

References:
    Hoskins B J, Draghici I, Davies H C (1978). A new look at the
    ω-equation. Q.J.R. Meteorol. Soc., 104, 31-38.
    Li L, O'Gorman P A (2020). Response of vertical velocities in
    extratropical precipitation extremes to climate change. J. Climate.
    Bluestein H B (1992). Synoptic-Dynamic Meteorology in Midlatitudes,
    Vol. II, eq. 5.7.54.
    Steinfeld D, Pfahl S (2019). The role of latent heating in
    atmospheric blocking dynamics. Climate Dyn., 53, 6159-6180.
    Stone H L (1968). Iterative solution of implicit approximations of
    multidimensional partial differential equations. SIAM J. Numer.
    Anal., 5, 530-558.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .constants import R_EARTH, OMEGA_E, R_DRY, SP19_DRY_FRACTION

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
    method: str = "sp19",
    center_lat: float | None = None,
    omega_total: np.ndarray | None = None,
    dry_fraction: float = SP19_DRY_FRACTION,
    t_full: np.ndarray | None = None,
    sip_maxit: int = 300,
    sip_alpha: float = 0.93,
    sip_resmax: float = 1e-14,
) -> np.ndarray:
    """Solve the QG omega equation for omega_dry.

    Args:
        u: Geostrophic zonal wind [m s⁻¹], shape ``(nlev, nlat, nlon)``.
        v: Geostrophic meridional wind [m s⁻¹], shape ``(nlev, nlat, nlon)``.
        t: Temperature [K], shape ``(nlev, nlat, nlon)``.  For anomaly
            solves pass anomaly here and full physical T via *t_full*.
        lat: Ascending latitude [degrees], shape ``(nlat,)``.
        lon: Longitude [degrees], shape ``(nlon,)``.
        plevs_pa: Pressure [Pa], ascending, shape ``(nlev,)``.
        method: Solver method — ``"sp19"`` (Steinfeld & Pfahl 2019
            empirical scaling, default), ``"fft"`` (FFT + Thomas
            tridiagonal), or ``"log20"`` (SIP, Li & O'Gorman 2020).
        center_lat: If given, use constant f₀ = 2Ω sin(center_lat) and
            β₀ = 2Ω cos(center_lat)/a instead of latitude-varying values.
            Recommended for event-centred patches (Li & O'Gorman 2020).
        omega_total: Total vertical velocity [Pa s⁻¹], required for
            ``method="sp19"``.  Shape ``(nlev, nlat, nlon)``.
        dry_fraction: Fraction ω_dry/ω_total for SP19 (default 1/3).
        t_full: Full physical temperature [K] for σ₀ computation.
            Required when *t* is an anomaly field (σ needs T ≈ 200–300 K).
        sip_maxit: Max SIP iterations for LOG20 (default 300).
        sip_alpha: SIP relaxation parameter for LOG20 (default 0.93).
        sip_resmax: SIP convergence threshold for LOG20 (default 1e-14).

    Returns:
        ``omega_dry`` — QG vertical velocity [Pa s⁻¹],
        shape ``(nlev, nlat, nlon)``.

    Raises:
        ValueError: If *method* is not ``"sp19"``, ``"fft"``, or ``"log20"``.
        ValueError: If *method* is ``"sp19"`` and *omega_total* is None.
    """
    _valid_methods = ("sp19", "fft", "log20")
    if method not in _valid_methods:
        raise ValueError(
            f"method must be one of {_valid_methods}, got {method!r}"
        )

    # ---- SP19: empirical scaling (no elliptic solve) ----
    if method == "sp19":
        if omega_total is None:
            raise ValueError(
                "omega_total is required for method='sp19' "
                "(Steinfeld & Pfahl 2019 scaling)"
            )
        return dry_fraction * omega_total

    # ---- LOG20: SIP iterative solver ----
    if method == "log20":
        omega_dry, _info = solve_qg_omega_sip(
            u, v, t, lat, lon, plevs_pa,
            center_lat=center_lat,
            t_full=t_full,
            maxit=sip_maxit,
            alpha=sip_alpha,
            resmax=sip_resmax,
        )
        return omega_dry

    # ---- FFT: compute Q-vector RHS and solve ----
    from .derivatives import gradient_periodic

    nlev, nlat, nlon = u.shape
    lat_rad = np.deg2rad(lat)
    dlat = np.abs(lat[1] - lat[0]) if nlat > 1 else 1.5
    dlon = np.abs(lon[1] - lon[0]) if nlon > 1 else 1.5
    dy = np.deg2rad(dlat) * R_EARTH
    dx_arr = np.deg2rad(dlon) * R_EARTH * np.cos(lat_rad)
    dx_arr = np.maximum(dx_arr, dy * 0.1)

    # Coriolis and beta — constant or latitude-varying
    f_min = 2 * OMEGA_E * np.sin(np.deg2rad(F_MIN_LAT))
    if center_lat is not None:
        f0_val = 2 * OMEGA_E * np.sin(np.deg2rad(center_lat))
        beta0_val = 2 * OMEGA_E * np.cos(np.deg2rad(center_lat)) / R_EARTH
        f_arr = np.full(nlat, f0_val)
        beta_arr = np.full(nlat, beta0_val)
    else:
        f_arr = 2 * OMEGA_E * np.sin(lat_rad)
        f_arr = np.sign(f_arr) * np.maximum(np.abs(f_arr), f_min)
        beta_arr = 2 * OMEGA_E * np.cos(lat_rad) / R_EARTH

    # Static stability profile σ₀(p) — domain-mean T (Bluestein eq 4.3.6)
    # Use t_full (physical temperature) when available; otherwise t itself.
    t_for_sigma = t_full if t_full is not None else t
    kappa = R_DRY / 1004.0
    T_mean = np.nanmean(t_for_sigma, axis=(1, 2))
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

    # Smooth T before computing gradients (Li & O'Gorman 2020 approach)
    dT_dx = np.zeros_like(t)
    dT_dy = np.zeros_like(t)
    for k in range(nlev):
        t_k = gaussian_smooth_2d(t[k], sigma=GEO_SMOOTH_SIGMA)
        dT_dx[k], dT_dy[k] = gradient_periodic(t_k, dx_arr, dy)

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

    # ---- FFT + Thomas tridiagonal solver ----
    omega_dry = _solve_fft_thomas(
        rhs_qv, lat, lat_rad, plevs_pa, sigma0, f_arr,
        nlev, nlat, nlon,
    )

    omega_dry *= lat_taper_full[None, :, None]
    return omega_dry


# =========================================================================
#  Internal solver: FFT in longitude + Thomas in pressure
# =========================================================================

def _solve_fft_thomas(
    rhs: np.ndarray,
    lat: np.ndarray,
    lat_rad: np.ndarray,
    plevs_pa: np.ndarray,
    sigma0: np.ndarray,
    f_arr: np.ndarray,
    nlev: int,
    nlat: int,
    nlon: int,
) -> np.ndarray:
    """FFT + Thomas tridiagonal solver (drops ∂²ω/∂y²)."""
    rhs_hat = np.fft.rfft(rhs, axis=-1)
    omega_hat = np.zeros_like(rhs_hat)
    n_int = nlev - 2
    if n_int < 1:
        return np.zeros((nlev, nlat, nlon))

    dp_diff = np.diff(plevs_pa)
    f2 = f_arr ** 2
    nfreq = rhs_hat.shape[-1]
    m_arr = np.arange(nfreq, dtype=float)
    cos2 = np.maximum(np.cos(lat_rad) ** 2, 1e-6)
    kx2 = m_arr[None, :] ** 2 / (R_EARTH ** 2 * cos2[:, None])

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

    omega_hat[:, :, 0] = 0.0
    return np.fft.irfft(omega_hat, n=nlon, axis=-1)


# =========================================================================
#  SIP (Strongly Implicit Procedure) QG omega solver — Li & O'Gorman (2020)
# =========================================================================

# --- numba availability check (once at import) ---
try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _compute_sigma_3d(
    t: np.ndarray,
    plevs_pa: np.ndarray,
) -> np.ndarray:
    """Compute 3-D static stability σ(x, y, p) from local temperature.

    .. math::

        \\sigma(x,y,p) = -\\frac{R_d\\,T(x,y,p)}{p}
                          \\frac{\\partial\\ln\\theta}{\\partial p}

    (Bluestein 1992, eq. 4.3.6; MetPy static_stability).

    Parameters
    ----------
    t : (nlev, nlat, nlon)
        Temperature [K].
    plevs_pa : (nlev,)
        Pressure [Pa], ascending.

    Returns
    -------
    sigma_3d : (nlev, nlat, nlon)
        Static stability [m² Pa⁻² s⁻²], clipped ≥ 1e-7.
    """
    kappa = R_DRY / 1004.0
    theta = t * (1e5 / plevs_pa[:, None, None]) ** kappa
    nlev = t.shape[0]
    sigma_3d = np.zeros_like(t)
    for k in range(1, nlev - 1):
        dp = plevs_pa[k + 1] - plevs_pa[k - 1]
        dlntheta = np.log(theta[k + 1]) - np.log(theta[k - 1])
        sigma_3d[k] = -(R_DRY * t[k] / plevs_pa[k]) * (dlntheta / dp)
    sigma_3d[0] = sigma_3d[1]
    sigma_3d[-1] = sigma_3d[-2]
    return np.maximum(sigma_3d, 1e-7)


def _compute_qg_rhs(
    u: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    plevs_pa: np.ndarray,
    *,
    center_lat: float | None = None,
    t_full: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Compute the Q-vector + β-term RHS shared by FFT and SIP solvers.

    The RHS is in σ₀-divided form:  ``Q = -R_d/(σ₀ p)·(∂Vg/∂·)·∇T``,
    where σ₀(p) is computed from **domain-mean** T.

    When solving for anomaly fields (t = T − T̄), pass the **full
    physical temperature** via *t_full* so that σ₀ is computed
    correctly (σ requires T ≈ 200–300 K, not anomaly ≈ 0 K).

    Zonal gradients use periodic wrapping when the longitude span ≥ 350°
    (full NH ring) and one-sided differences otherwise (local patch).

    Returns
    -------
    rhs : (nlev, nlat, nlon)
        Q-vector divergence + β-term forcing (σ₀-divided).
    sigma0 : (nlev,)
        Domain-mean static stability profile.
    f0 : float
        Constant Coriolis parameter at *center_lat*.
    beta0 : float
        β₀ = 2Ω cos(φ₀)/a.
    taper : (nlat,)
        Latitude taper mask.
    """
    from .derivatives import ddx, ddy

    nlev, nlat, nlon = u.shape
    lat_rad = np.deg2rad(lat)
    dlat = np.abs(lat[1] - lat[0]) if nlat > 1 else 1.5
    dlon = np.abs(lon[1] - lon[0]) if nlon > 1 else 1.5
    dy = np.deg2rad(dlat) * R_EARTH
    dx_arr = np.deg2rad(dlon) * R_EARTH * np.cos(lat_rad)
    dx_arr = np.maximum(dx_arr, dy * 0.1)

    # Detect full ring vs local patch
    lon_span = dlon * nlon
    periodic = lon_span > 350.0

    def _grad(field_2d):
        """Horizontal gradient respecting periodic/non-periodic lon."""
        return ddx(field_2d, dx_arr, periodic=periodic), ddy(field_2d, dy)

    # Coriolis and beta
    f_min = 2 * OMEGA_E * np.sin(np.deg2rad(F_MIN_LAT))
    if center_lat is not None:
        f0_val = 2 * OMEGA_E * np.sin(np.deg2rad(center_lat))
        beta0_val = 2 * OMEGA_E * np.cos(np.deg2rad(center_lat)) / R_EARTH
    else:
        f0_val = 2 * OMEGA_E * np.sin(np.deg2rad(45.0))
        beta0_val = 2 * OMEGA_E * np.cos(np.deg2rad(45.0)) / R_EARTH

    f_arr = np.full(nlat, f0_val) if center_lat is not None else (
        np.sign(2 * OMEGA_E * np.sin(lat_rad))
        * np.maximum(np.abs(2 * OMEGA_E * np.sin(lat_rad)), f_min)
    )
    beta_arr = np.full(nlat, beta0_val) if center_lat is not None else (
        2 * OMEGA_E * np.cos(lat_rad) / R_EARTH
    )

    # Static stability profile σ₀(p) — domain-mean T (Bluestein eq 4.3.6)
    # Use t_full (physical temperature) when available; otherwise t itself.
    t_for_sigma = t_full if t_full is not None else t
    kappa = R_DRY / 1004.0
    T_mean = np.nanmean(t_for_sigma, axis=(1, 2))
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

    # Smooth T before computing gradients (Li & O'Gorman 2020 approach)
    dT_dx = np.zeros_like(t)
    dT_dy = np.zeros_like(t)
    for k in range(nlev):
        t_k = gaussian_smooth_2d(t[k], sigma=GEO_SMOOTH_SIGMA)
        dT_dx[k], dT_dy[k] = _grad(t_k)

    # Wind gradients
    dug_dx = np.zeros_like(u)
    dug_dy = np.zeros_like(u)
    dvg_dx = np.zeros_like(v)
    dvg_dy = np.zeros_like(v)
    for k in range(nlev):
        dug_dx[k], dug_dy[k] = _grad(u_tapered[k])
        dvg_dx[k], dvg_dy[k] = _grad(v_tapered[k])

    # Q-vector divergence and beta term → RHS
    rhs = np.zeros_like(u)
    for k in range(nlev):
        coef = -R_DRY / (sigma0[k] * plevs_pa[k])
        Q1 = coef * (dug_dx[k] * dT_dx[k] + dvg_dx[k] * dT_dy[k])
        Q2 = coef * (dug_dy[k] * dT_dx[k] + dvg_dy[k] * dT_dy[k])
        Q1 *= lat_taper_full[:, None]
        Q2 *= lat_taper_full[:, None]
        dQ1_dx, _ = _grad(Q1)
        _, dQ2_dy = _grad(Q2)
        div_Q = dQ1_dx + dQ2_dy
        beta_term = (
            beta_arr[:, None]
            * (R_DRY / (sigma0[k] * plevs_pa[k]))
            * dT_dx[k]
        )
        beta_term *= lat_taper_full[:, None]
        rhs[k] = -2.0 * div_Q - beta_term

    return rhs, sigma0, f0_val, beta0_val, lat_taper_full


# ---------------------------------------------------------------------------
#  SIP core iteration (pure-Python fallback + optional numba)
# ---------------------------------------------------------------------------

def _sip_core_python(
    AP: np.ndarray,
    AE: np.ndarray,
    AW: np.ndarray,
    AN: np.ndarray,
    AS: np.ndarray,
    AT: np.ndarray,
    AB: np.ndarray,
    Q: np.ndarray,
    T: np.ndarray,
    Nk: int,
    Nj: int,
    Ni: int,
    alpha: float,
    maxit: int,
    resmax: float,
    periodic_lon: int = 1,
) -> tuple[int, float]:
    """Pure-Python SIP core (Stone 1968) — fallback when numba is absent.

    All arrays are ``(Nk, Nj, Ni)`` — (pressure, lat, lon).
    *T* is modified **in-place** (initial guess + boundary values pre-set).
    Lat/pressure boundaries are always Dirichlet.
    Longitude (i-axis) is periodic when *periodic_lon* == 1;
    otherwise Dirichlet (for local patches).

    Returns
    -------
    n_iters : int
        Number of SIP iterations performed.
    rsm : float
        Final relative residual.
    """
    LB  = np.zeros((Nk, Nj, Ni))
    LW  = np.zeros((Nk, Nj, Ni))
    LS  = np.zeros((Nk, Nj, Ni))
    LPR = np.zeros((Nk, Nj, Ni))
    UN  = np.zeros((Nk, Nj, Ni))
    UE  = np.zeros((Nk, Nj, Ni))
    UT  = np.zeros((Nk, Nj, Ni))
    RES = np.zeros((Nk, Nj, Ni))

    # i-loop bounds: periodic → 0..Ni-1; Dirichlet → 1..Ni-2
    i_lo = 0 if periodic_lon else 1
    i_hi = Ni if periodic_lon else Ni - 1

    # ── Step A — ILU-like factorisation (once) ──
    for k in range(1, Nk - 1):
        for j in range(1, Nj - 1):
            for i in range(i_lo, i_hi):
                if periodic_lon:
                    iw = (i - 1) % Ni
                else:
                    iw = i - 1

                lb_val = AB[k, j, i] / (1.0 + alpha * (UN[k - 1, j, i]
                                                         + UE[k - 1, j, i]))
                lw_val = AW[k, j, i] / (1.0 + alpha * (UN[k, j, iw]
                                                         + UT[k, j, iw]))
                ls_val = AS[k, j, i] / (1.0 + alpha * (UE[k, j - 1, i]
                                                         + UT[k, j - 1, i]))

                p1 = alpha * (lb_val * UN[k - 1, j, i]
                              + lw_val * UN[k, j, iw])
                p2 = alpha * (lb_val * UE[k - 1, j, i]
                              + ls_val * UE[k, j - 1, i])
                p3 = alpha * (lw_val * UT[k, j, iw]
                              + ls_val * UT[k, j - 1, i])

                lpr_val = 1.0 / (AP[k, j, i] + p1 + p2 + p3
                                  - lb_val * UT[k - 1, j, i]
                                  - lw_val * UE[k, j, iw]
                                  - ls_val * UN[k, j - 1, i]
                                  + 1e-20)

                un_val = (AN[k, j, i] - p1) * lpr_val
                ue_val = (AE[k, j, i] - p2) * lpr_val
                ut_val = (AT[k, j, i] - p3) * lpr_val

                LB[k, j, i]  = lb_val
                LW[k, j, i]  = lw_val
                LS[k, j, i]  = ls_val
                LPR[k, j, i] = lpr_val
                UN[k, j, i]  = un_val
                UE[k, j, i]  = ue_val
                UT[k, j, i]  = ut_val

    # ── Step B — SIP iteration loop ──
    res0 = 0.0
    n_iters = 0
    rsm = 1.0

    for it in range(1, maxit + 1):
        resl = 0.0

        # — forward sweep —
        for k in range(1, Nk - 1):
            for j in range(1, Nj - 1):
                for i in range(i_lo, i_hi):
                    if periodic_lon:
                        iw = (i - 1) % Ni
                        ie = (i + 1) % Ni
                    else:
                        iw = i - 1
                        ie = i + 1

                    res_val = (Q[k, j, i]
                               - AE[k, j, i] * T[k, j, ie]
                               - AW[k, j, i] * T[k, j, iw]
                               - AN[k, j, i] * T[k, j + 1, i]
                               - AS[k, j, i] * T[k, j - 1, i]
                               - AT[k, j, i] * T[k + 1, j, i]
                               - AB[k, j, i] * T[k - 1, j, i]
                               - AP[k, j, i] * T[k, j, i])
                    resl += abs(res_val)

                    res_val = ((res_val
                                - LB[k, j, i] * RES[k - 1, j, i]
                                - LW[k, j, i] * RES[k, j, iw]
                                - LS[k, j, i] * RES[k, j - 1, i])
                               * LPR[k, j, i])
                    RES[k, j, i] = res_val

        # — backward sweep —
        for k in range(Nk - 2, 0, -1):
            for j in range(Nj - 2, 0, -1):
                for i in range(i_hi - 1, i_lo - 1, -1):
                    if periodic_lon:
                        ie = (i + 1) % Ni
                    else:
                        ie = i + 1
                    RES[k, j, i] = (RES[k, j, i]
                                    - UN[k, j, i] * RES[k, j + 1, i]
                                    - UE[k, j, i] * RES[k, j, ie]
                                    - UT[k, j, i] * RES[k + 1, j, i])
                    T[k, j, i] += RES[k, j, i]

        # — convergence check —
        n_iters = it
        if it == 1:
            res0 = resl + 1e-30
        rsm = resl / res0
        if rsm < resmax:
            break

    return n_iters, rsm


if _HAS_NUMBA:
    @_njit(cache=True)
    def _sip_core(
        AP: np.ndarray,
        AE: np.ndarray,
        AW: np.ndarray,
        AN: np.ndarray,
        AS: np.ndarray,
        AT: np.ndarray,
        AB: np.ndarray,
        Q: np.ndarray,
        T: np.ndarray,
        Nk: int,
        Nj: int,
        Ni: int,
        alpha: float,
        maxit: int,
        resmax: float,
        periodic_lon: int = 1,
    ) -> tuple[int, float]:
        """Numba-accelerated SIP core (Stone 1968).

        All arrays are ``(Nk, Nj, Ni)``.  *T* modified **in-place**.
        Lat/pressure boundaries Dirichlet.
        Longitude: periodic when *periodic_lon* == 1, Dirichlet otherwise.
        """
        LB  = np.zeros((Nk, Nj, Ni))
        LW  = np.zeros((Nk, Nj, Ni))
        LS  = np.zeros((Nk, Nj, Ni))
        LPR = np.zeros((Nk, Nj, Ni))
        UN  = np.zeros((Nk, Nj, Ni))
        UE  = np.zeros((Nk, Nj, Ni))
        UT  = np.zeros((Nk, Nj, Ni))
        RES = np.zeros((Nk, Nj, Ni))

        # i-loop bounds: periodic → 0..Ni-1; Dirichlet → 1..Ni-2
        i_lo = 0 if periodic_lon else 1
        i_hi = Ni if periodic_lon else Ni - 1

        # ── Step A — ILU-like factorisation ──
        for k in range(1, Nk - 1):
            for j in range(1, Nj - 1):
                for i in range(i_lo, i_hi):
                    if periodic_lon:
                        iw = (i - 1) % Ni
                    else:
                        iw = i - 1

                    lb_val = AB[k, j, i] / (1.0 + alpha * (UN[k - 1, j, i]
                                                             + UE[k - 1, j, i]))
                    lw_val = AW[k, j, i] / (1.0 + alpha * (UN[k, j, iw]
                                                             + UT[k, j, iw]))
                    ls_val = AS[k, j, i] / (1.0 + alpha * (UE[k, j - 1, i]
                                                             + UT[k, j - 1, i]))

                    p1 = alpha * (lb_val * UN[k - 1, j, i]
                                  + lw_val * UN[k, j, iw])
                    p2 = alpha * (lb_val * UE[k - 1, j, i]
                                  + ls_val * UE[k, j - 1, i])
                    p3 = alpha * (lw_val * UT[k, j, iw]
                                  + ls_val * UT[k, j - 1, i])

                    lpr_val = 1.0 / (AP[k, j, i] + p1 + p2 + p3
                                      - lb_val * UT[k - 1, j, i]
                                      - lw_val * UE[k, j, iw]
                                      - ls_val * UN[k, j - 1, i]
                                      + 1e-20)

                    un_val = (AN[k, j, i] - p1) * lpr_val
                    ue_val = (AE[k, j, i] - p2) * lpr_val
                    ut_val = (AT[k, j, i] - p3) * lpr_val

                    LB[k, j, i]  = lb_val
                    LW[k, j, i]  = lw_val
                    LS[k, j, i]  = ls_val
                    LPR[k, j, i] = lpr_val
                    UN[k, j, i]  = un_val
                    UE[k, j, i]  = ue_val
                    UT[k, j, i]  = ut_val

        # ── Step B — SIP iteration loop ──
        res0 = 0.0
        n_iters = 0
        rsm = 1.0

        for it in range(1, maxit + 1):
            resl = 0.0

            # — forward sweep —
            for k in range(1, Nk - 1):
                for j in range(1, Nj - 1):
                    for i in range(i_lo, i_hi):
                        if periodic_lon:
                            iw = (i - 1) % Ni
                            ie = (i + 1) % Ni
                        else:
                            iw = i - 1
                            ie = i + 1

                        res_val = (Q[k, j, i]
                                   - AE[k, j, i] * T[k, j, ie]
                                   - AW[k, j, i] * T[k, j, iw]
                                   - AN[k, j, i] * T[k, j + 1, i]
                                   - AS[k, j, i] * T[k, j - 1, i]
                                   - AT[k, j, i] * T[k + 1, j, i]
                                   - AB[k, j, i] * T[k - 1, j, i]
                                   - AP[k, j, i] * T[k, j, i])
                        resl += abs(res_val)

                        res_val = ((res_val
                                    - LB[k, j, i] * RES[k - 1, j, i]
                                    - LW[k, j, i] * RES[k, j, iw]
                                    - LS[k, j, i] * RES[k, j - 1, i])
                                   * LPR[k, j, i])
                        RES[k, j, i] = res_val

            # — backward sweep —
            for k in range(Nk - 2, 0, -1):
                for j in range(Nj - 2, 0, -1):
                    for i in range(i_hi - 1, i_lo - 1, -1):
                        if periodic_lon:
                            ie = (i + 1) % Ni
                        else:
                            ie = i + 1
                        RES[k, j, i] = (RES[k, j, i]
                                        - UN[k, j, i] * RES[k, j + 1, i]
                                        - UE[k, j, i] * RES[k, j, ie]
                                        - UT[k, j, i] * RES[k + 1, j, i])
                        T[k, j, i] += RES[k, j, i]

            # — convergence —
            n_iters = it
            if it == 1:
                res0 = resl + 1e-30
            rsm = resl / res0
            if rsm < resmax:
                break

        return n_iters, rsm
else:
    _sip_core = _sip_core_python


def solve_qg_omega_sip(
    u: np.ndarray,
    v: np.ndarray,
    t: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    plevs_pa: np.ndarray,
    *,
    center_lat: float | None = None,
    omega_b: np.ndarray | None = None,
    t_full: np.ndarray | None = None,
    maxit: int = 300,
    alpha: float = 0.93,
    resmax: float = 1e-14,
) -> tuple[np.ndarray, dict]:
    """Solve the QG omega equation using the SIP (Strongly Implicit Procedure).

    Matches the Li & O'Gorman (2020) solver from
    ``dante831/QG-omega/source/SIP_inversion.m``.

    Same RHS computation as :func:`solve_qg_omega` (Q-vector + β-term),
    but retains **both** ∂²ω/∂x² and ∂²ω/∂y² on the LHS using the
    full 7-point spherical stencil (no FFT in longitude).

    **Static stability**: uses domain-mean σ₀(p) throughout, matching
    the original Li & O'Gorman MATLAB reference.

    **Boundary conditions**: Dirichlet ω = 0 on all faces.  When the
    longitude span covers a full ring (≥ 350°), periodic wrapping is
    used in longitude instead.

    Args:
        u: Geostrophic zonal wind [m s⁻¹], ``(nlev, nlat, nlon)``.
        v: Geostrophic meridional wind [m s⁻¹], ``(nlev, nlat, nlon)``.
        t: Temperature [K], ``(nlev, nlat, nlon)``.  For anomaly solves,
            pass the anomaly field here and the **full** physical
            temperature via *t_full*.
        lat: Ascending latitude [degrees], ``(nlat,)``.
        lon: Longitude [degrees], ``(nlon,)``.
        plevs_pa: Pressure [Pa], ascending, ``(nlev,)``.
        center_lat: Latitude for constant f₀ and β₀.
        omega_b: Boundary omega ``(nlev, nlat, nlon)`` (None → 0).
        t_full: Full physical temperature [K] for σ₀ and σ_3d
            computation.  When ``None`` (default), *t* itself is used.
            **Must be supplied when *t* is an anomaly field**,
            otherwise σ ≈ 0 → NaN.
        maxit: Max SIP iterations (default 300).
        alpha: SIP relaxation parameter (default 0.93).
        resmax: Convergence threshold (default 1e-14).

    Returns:
        ``(omega_dry, info)`` where *omega_dry* is ``(nlev, nlat, nlon)``
        QG vertical velocity [Pa s⁻¹] and *info* is a dict with keys
        ``'iters'``, ``'final_residual'``, ``'numba'``, ``'solve_time'``.

    References:
        Stone H L (1968). Iterative solution of implicit approximations
        of multidimensional partial differential equations. SIAM J. Numer.
        Anal., 5, 530–558.

        Li L, O'Gorman P A (2020). Response of vertical velocities in
        extratropical precipitation extremes to climate change. J. Climate.
    """
    import time as _time
    t0_wall = _time.perf_counter()

    nlev, nlat, nlon = u.shape

    # --- 1. Compute Q-vector RHS (σ₀-divided, shared with FFT solver) ---
    rhs, sigma0, f0, beta0, lat_taper_full = _compute_qg_rhs(
        u, v, t, lat, lon, plevs_pa,
        center_lat=center_lat,
        t_full=t_full,
    )

    # Detect full ring vs local patch — controls BCs in SIP core
    dlon_deg = np.abs(lon[1] - lon[0]) if len(lon) > 1 else 1.5
    lon_span = dlon_deg * len(lon)
    periodic_lon = 1 if lon_span > 350.0 else 0

    # --- 2. Grid parameters in spherical coordinates ---
    r = R_EARTH
    phi = np.deg2rad(lat)                       # (nlat,)
    dlat_deg = np.abs(lat[1] - lat[0]) if nlat > 1 else 1.5
    dphi = np.deg2rad(dlat_deg)
    dlambda = np.deg2rad(dlon_deg)
    f2_0 = f0 ** 2

    # --- 3. Build 7-point stencil coefficients ---
    #   σ₀-divided form (domain-mean σ₀ throughout, matching
    #   Li & O'Gorman 2020 and the original MATLAB code):
    #     AW = AE = 1/(r²cos²φ Δλ²)
    #     AS = 1/(r²Δφ²) + tanφ/(2r²Δφ)
    #     AN = 1/(r²Δφ²) - tanφ/(2r²Δφ)
    #     AB = 2 f₀²/(σ₀ Δp₁ (Δp₁+Δp₂))
    #     AT = 2 f₀²/(σ₀ Δp₂ (Δp₁+Δp₂))
    #     AP = -2/(r²cos²φΔλ²) - 2/(r²Δφ²) - 2f₀²/(σ₀ Δp₁ Δp₂)
    AP = np.zeros((nlev, nlat, nlon))
    AE = np.zeros((nlev, nlat, nlon))
    AW = np.zeros((nlev, nlat, nlon))
    AN = np.zeros((nlev, nlat, nlon))
    AS = np.zeros((nlev, nlat, nlon))
    AT = np.zeros((nlev, nlat, nlon))
    AB = np.zeros((nlev, nlat, nlon))

    cos2_phi = np.cos(phi) ** 2
    tan_phi = np.tan(phi)

    for k in range(1, nlev - 1):
        dp1 = plevs_pa[k] - plevs_pa[k - 1]
        dp2 = plevs_pa[k + 1] - plevs_pa[k]
        s0k = sigma0[k]

        ab_val = 2.0 * f2_0 / (s0k * dp1 * (dp1 + dp2))
        at_val = 2.0 * f2_0 / (s0k * dp2 * (dp1 + dp2))
        vert_diag = -2.0 * f2_0 / (s0k * dp1 * dp2)

        for j in range(1, nlat - 1):
            c2 = cos2_phi[j]
            tp = tan_phi[j]

            ew = 1.0 / (r ** 2 * c2 * dlambda ** 2)
            ns_base = 1.0 / (r ** 2 * dphi ** 2)
            ns_tan  = tp / (2.0 * r ** 2 * dphi)

            # Horizontal coefficients — uniform σ₀ (no local σ_ratio)
            AW[k, j, :] = ew
            AE[k, j, :] = ew
            AS[k, j, :] = ns_base + ns_tan
            AN[k, j, :] = ns_base - ns_tan

            # Vertical coefficients
            AB[k, j, :] = ab_val
            AT[k, j, :] = at_val

            # Diagonal
            AP[k, j, :] = -2.0 * ew - 2.0 * ns_base + vert_diag

    # --- 4. Prepare solution array with BCs ---
    T_sol = np.zeros((nlev, nlat, nlon))
    if omega_b is not None:
        T_sol[0, :, :]  = omega_b[0, :, :]
        T_sol[-1, :, :] = omega_b[-1, :, :]
        T_sol[:, 0, :]  = omega_b[:, 0, :]
        T_sol[:, -1, :] = omega_b[:, -1, :]
    # For Dirichlet lon BCs, also fix i=0 and i=-1
    if not periodic_lon and omega_b is not None:
        T_sol[:, :, 0]  = omega_b[:, :, 0]
        T_sol[:, :, -1] = omega_b[:, :, -1]

    # --- 5. Solve with SIP core ---
    n_iters, final_res = _sip_core(
        AP, AE, AW, AN, AS, AT, AB, rhs, T_sol,
        nlev, nlat, nlon, alpha, maxit, resmax,
        periodic_lon,
    )

    omega_dry = T_sol

    # --- 6. Apply latitude taper ---
    omega_dry *= lat_taper_full[None, :, None]

    # --- 7. Remove zonal mean (match FFT where m=0 is excluded) ---
    # Only remove zonal mean when longitude covers a full ring (~360°);
    # on a local patch the "zonal mean" is NOT the true m=0 wavenumber
    # and removing it would destroy the signal.
    if periodic_lon:
        zmean = np.mean(omega_dry, axis=-1, keepdims=True)
        omega_dry -= zmean

    # --- 8. Clean NaN / inf ---
    omega_dry = np.nan_to_num(omega_dry, nan=0.0, posinf=0.0, neginf=0.0)

    elapsed = _time.perf_counter() - t0_wall
    info = {
        "iters": n_iters,
        "final_residual": float(final_res),
        "numba": _HAS_NUMBA,
        "solve_time": elapsed,
    }
    return omega_dry, info
