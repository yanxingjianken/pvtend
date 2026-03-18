"""Hoskins (1978) Q-vector QG omega equation solver.

Solves the quasi-geostrophic omega equation in σ-in-Laplacian form
(Li & O'Gorman 2020):

    ∇²(σω) + f₀² ∂²ω/∂p² = A + B + C

where:
    A = 2 R_d/p · ∇·Q   (Dostalek et al. 2017 spherical Q-vector)
    B = f₀ β ∂v_g/∂p     (direct β-term)
    C = -(κ/p) ∇²J       (diabatic heating)

Q-vector (Dostalek, Schubert & DeMaria 2017, Eq. 19):
    Qλ = (1/a²)[∂T/∂φ(1/cosφ ∂v_g/∂λ + u_g tanφ) - 1/cosφ ∂T/∂λ ∂v_g/∂φ]
    Qφ = (1/a²)[∂T/∂φ(-1/cosφ ∂u_g/∂λ + v_g tanφ) + 1/cosφ ∂T/∂λ ∂u_g/∂φ]

Divergence (Lynch 1989):
    ∇·Q = 1/(a cosφ) ∂Qλ/∂λ + 1/(a cosφ) ∂(Qφ cosφ)/∂φ

Solver: Strongly Implicit Procedure (SIP, Stone 1968), full 3-D
spherical stencil with tan(φ) metric term.  σ enters the horizontal
stencil at **neighbor** grid points (σ-in-Laplacian), matching LOG20
``SIP_inversion.m``.  Numba-accelerated.

BCs: Real ERA5 ω at top and bottom pressure levels (Dirichlet),
lateral N/S faces may also use observed ω.
Latitude taper: QG invalid below ~15°N.

References:
    Hoskins B J, Draghici I, Davies H C (1978). A new look at the
    ω-equation. Q.J.R. Meteorol. Soc., 104, 31-38.
    Li L, O'Gorman P A (2020). Response of vertical velocities in
    extratropical precipitation extremes to climate change. J. Climate.
    Dostalek J F, Schubert W H, DeMaria M (2017). Derivation of the
    equations for Q vectors in spherical coordinates. MWR, 145, 2285-2289.
    Lynch P (1989). Partitioning the wind in a limited domain. MWR, 117.
    Stone H L (1968). Iterative solution of implicit approximations of
    multidimensional partial differential equations. SIAM J. Numer.
    Anal., 5, 530-558.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .constants import R_EARTH, OMEGA_E, R_DRY

# Constants for geostrophic wind
GEO_SMOOTH_SIGMA: float = 1.5   # Gaussian sigma in grid points
F_MIN_LAT: float = 5.0          # degrees — regularize |f| >= f(F_MIN_LAT)

# Latitude taper constants for QG validity
LAT_QG_LO: float = 15.0    # below this: omega_dry ≡ 0
LAT_QG_HI: float = 25.0    # above this: full weight
LAT_QG_POLAR: float = 85.0  # above this: taper to 0


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
    sigma_smooth: float = 0,
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
            Following Li & O'Gorman (2020), the geostrophic wind is
            computed from **unsmoothed** geopotential by default.

    Returns:
        ``(u_g, v_g)`` — geostrophic wind [m s⁻¹], each ``(nlev, nlat, nlon)``.
    """
    from .derivatives import ddx, ddy

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

    # Detect full ring vs local patch
    lon_span = dlon * nlon
    periodic = lon_span > 350.0

    u_g = np.zeros_like(phi_3d)
    v_g = np.zeros_like(phi_3d)

    for k in range(nlev):
        phi_k = phi_3d[k]
        if sigma_smooth > 0:
            phi_k = gaussian_smooth_2d(phi_k, sigma=sigma_smooth)
        dphi_dx = ddx(phi_k, dx_arr, periodic=periodic)
        dphi_dy = ddy(phi_k, dy)
        for j in range(nlat):
            u_g[k, j, :] = -dphi_dy[j, :] / f_arr[j]
            v_g[k, j, :] = dphi_dx[j, :] / f_arr[j]

    return u_g, v_g


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
    phi_3d: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Compute the Q-vector + β-term RHS (Dostalek 2017 spherical form).

    Uses the Dostalek, Schubert & DeMaria (2017) spherical Q-vector
    with curvature/metric terms (tanφ, 1/cosφ), matching LOG20
    ``Q_vector.m``.  The divergence uses the full spherical form
    (LOG20 ``div.m``).

    The RHS is **not** divided by σ (the σ-in-Laplacian form absorbs σ
    into the LHS stencil).

    Returns
    -------
    rhs : (nlev, nlat, nlon)
        Q-vector divergence + β-term forcing (**not** σ-divided).
    sigma0 : (nlev,)
        Domain-mean static stability profile (for reference).
    f0 : float
        Constant Coriolis parameter at *center_lat*.
    beta0 : float
        β₀ = 2Ω cos(φ₀)/a.
    taper : (nlat,)
        Latitude taper mask.
    """
    from .derivatives import d_dlambda, d_dphi, div_spherical, ddp

    nlev, nlat, nlon = u.shape
    lat_rad = np.deg2rad(lat)
    dlat = np.abs(lat[1] - lat[0]) if nlat > 1 else 1.5
    dlon = np.abs(lon[1] - lon[0]) if nlon > 1 else 1.5
    dphi_rad = np.deg2rad(dlat)
    dlambda = np.deg2rad(dlon)

    # Detect full ring vs local patch
    lon_span = dlon * nlon
    periodic = lon_span > 350.0

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

    # Pre-compute trigonometric arrays (nlat, 1) for broadcasting
    cos_phi = np.cos(lat_rad)[:, None]           # (nlat, 1)
    tan_phi = np.tan(lat_rad)[:, None]           # (nlat, 1)
    inv_cos_phi = 1.0 / np.maximum(cos_phi, 1e-10)  # avoid pole singularity
    R2 = R_EARTH ** 2

    # === Angular derivatives of T (or T_geo via thermal wind) ===
    dT_dlam = np.zeros_like(t)   # ∂T/∂λ [K/rad]
    dT_dphi_arr = np.zeros_like(t)   # ∂T/∂φ [K/rad]

    if phi_3d is not None:
        # GEO_T: thermal-wind form (matching LOG20 Q_vector.m lines 34-35)
        dug_dp_all = ddp(u_tapered, plevs_pa)
        dvg_dp_all = ddp(v_tapered, plevs_pa)
        for k in range(nlev):
            dT_dlam[k] = (
                f0_val * plevs_pa[k] / R_DRY
                * (-dvg_dp_all[k]) * R_EARTH * cos_phi
            )
            dT_dphi_arr[k] = (
                f0_val * plevs_pa[k] / R_DRY
                * dug_dp_all[k] * R_EARTH
            )
    else:
        # Direct angular derivatives of smoothed T
        for k in range(nlev):
            t_k = gaussian_smooth_2d(t[k], sigma=GEO_SMOOTH_SIGMA)
            dT_dlam[k] = d_dlambda(t_k, dlambda, periodic)
            dT_dphi_arr[k] = d_dphi(t_k, dphi_rad)

    # === Angular derivatives of geostrophic wind ===
    dug_dlam = np.zeros_like(u)
    dug_dphi_a = np.zeros_like(u)
    dvg_dlam = np.zeros_like(v)
    dvg_dphi_a = np.zeros_like(v)
    for k in range(nlev):
        dug_dlam[k] = d_dlambda(u_tapered[k], dlambda, periodic)
        dug_dphi_a[k] = d_dphi(u_tapered[k], dphi_rad)
        dvg_dlam[k] = d_dlambda(v_tapered[k], dlambda, periodic)
        dvg_dphi_a[k] = d_dphi(v_tapered[k], dphi_rad)

    # === β-term: f·β·∂vg/∂p (LOG20 traditional_A.m) ===
    dvg_dp = ddp(v_tapered, plevs_pa)

    # === Dostalek (2017) spherical Q-vector + divergence → RHS ===
    rhs = np.zeros_like(u)
    for k in range(nlev):
        # Q-vector components (LOG20 Q_vector.m lines 39-45)
        Qx = (1.0 / R2) * (
            dT_dphi_arr[k] * (
                inv_cos_phi * dvg_dlam[k]
                + u_tapered[k] * tan_phi
            )
            - inv_cos_phi * dT_dlam[k] * dvg_dphi_a[k]
        )
        Qy = (1.0 / R2) * (
            dT_dphi_arr[k] * (
                -inv_cos_phi * dug_dlam[k]
                + v_tapered[k] * tan_phi
            )
            + inv_cos_phi * dT_dlam[k] * dug_dphi_a[k]
        )

        Qx *= lat_taper_full[:, None]
        Qy *= lat_taper_full[:, None]

        # Spherical divergence (LOG20 div.m)
        div_Q = div_spherical(Qx, Qy, lat_rad, dphi_rad, dlambda, periodic)

        # A = 2·Rd/p · div(Q)  (LOG20 Q_vector.m line 47)
        A_k = 2.0 * R_DRY / plevs_pa[k] * div_Q

        # B = f·β·∂vg/∂p  (LOG20 traditional_A.m line 28)
        beta_term = f_arr[:, None] * beta_arr[:, None] * dvg_dp[k]
        beta_term *= lat_taper_full[:, None]

        rhs[k] = A_k + beta_term

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
    @_njit(cache=True, nogil=True)
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


# ---------------------------------------------------------------------------
#  Diabatic forcing functions (Term C)
# ---------------------------------------------------------------------------

def _compute_diabatic_rhs_log20(
    t: np.ndarray,
    dT_dt: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    omega_era5: np.ndarray,
    sigma_3d: np.ndarray,
    plevs_pa: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Full LOG20 diabatic forcing: C = -(κ/p) ∇²_spherical(J).

    Computes J = J₁ + J₂ where:
        J₁ = c_p (∂T/∂t + v·∇T)   (local + horizontal advection)
        J₂ = -σ_local p / R_d c_p ω  (adiabatic warming/cooling)

    Uses the conservative spherical Laplacian for ∇².

    Returns C as (nlev, nlat, nlon), **non** σ₀-divided.
    """
    from .derivatives import ddx, ddy
    from .helmholtz import laplacian_spherical_fft

    nlev, nlat, nlon = t.shape
    lat_rad = np.deg2rad(lat)
    dlat = float(np.abs(lat[1] - lat[0])) if nlat > 1 else 1.5
    dlon = float(np.abs(lon[1] - lon[0])) if nlon > 1 else 1.5
    dy_m = np.deg2rad(dlat) * R_EARTH
    dx_arr = np.deg2rad(dlon) * R_EARTH * np.cos(lat_rad)
    dx_arr = np.maximum(dx_arr, dy_m * 0.1)
    dlon_rad = np.deg2rad(dlon)
    lon_span = dlon * nlon
    periodic = lon_span > 350.0

    kappa = R_DRY / 1004.0
    c_p = 1004.0

    dT_dt_safe = np.nan_to_num(dT_dt, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.zeros_like(t)

    for k in range(nlev):
        # Smooth T *before* taking derivatives (matches LOG20 Matlab;
        # smoothing J *after* kills small-scale features that the
        # Laplacian amplifies, underestimating C by ~36×).
        t_smooth_k = gaussian_smooth_2d(t[k], sigma=GEO_SMOOTH_SIGMA)
        dT_dt_smooth_k = gaussian_smooth_2d(dT_dt_safe[k],
                                            sigma=GEO_SMOOTH_SIGMA)

        # T gradients for advection (smoothed T, not T_geo)
        dT_dx_k = ddx(t_smooth_k, dx_arr, periodic=periodic)
        dT_dy_k = ddy(t_smooth_k, dy_m)

        # J₁ = c_p (∂T_smooth/∂t + u ∂T_smooth/∂x + v ∂T_smooth/∂y)
        J1_k = c_p * (dT_dt_smooth_k + u[k] * dT_dx_k + v[k] * dT_dy_k)
        # J₂ = -σ_local × (p/R_d) × c_p × ω
        J2_k = -sigma_3d[k] * (plevs_pa[k] / R_DRY) * c_p * omega_era5[k]
        J_k = J1_k + J2_k

        # C = -(κ/p) ∇²_spherical(J)
        lap_J = laplacian_spherical_fft(J_k, lat, dy_m, dlon_rad,
                                        R_earth=R_EARTH)
        C[k] = -(kappa / plevs_pa[k]) * lap_J

    return C


def _compute_diabatic_rhs_emanuel(
    theta_dot_lhr: np.ndarray,
    t: np.ndarray,
    theta: np.ndarray,
    plevs_pa: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Emanuel LHR-based diabatic forcing: C_em = -(κ/p) ∇²_spherical(J_em).

    J_em = c_p θ̇_LHR T/θ  [W/kg]

    Returns C_em as (nlev, nlat, nlon), **non** σ₀-divided.
    """
    from .helmholtz import laplacian_spherical_fft

    nlev, nlat, nlon = t.shape
    dlat = float(np.abs(lat[1] - lat[0])) if nlat > 1 else 1.5
    dlon = float(np.abs(lon[1] - lon[0])) if nlon > 1 else 1.5
    dy_m = np.deg2rad(dlat) * R_EARTH
    dlon_rad = np.deg2rad(dlon)

    kappa = R_DRY / 1004.0
    c_p = 1004.0

    theta_dot_safe = np.nan_to_num(theta_dot_lhr, nan=0.0, posinf=0.0,
                                   neginf=0.0)
    C_em = np.zeros_like(t)

    for k in range(nlev):
        # J_em = c_p θ̇_LHR T/θ
        with np.errstate(divide='ignore', invalid='ignore'):
            J_em_k = c_p * theta_dot_safe[k] * t[k] / np.maximum(theta[k], 1.0)

        # C_em = -(κ/p) ∇²_spherical(J_em)
        lap_J = laplacian_spherical_fft(J_em_k, lat, dy_m, dlon_rad,
                                        R_earth=R_EARTH)
        C_em[k] = -(kappa / plevs_pa[k]) * lap_J

    return C_em


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
    rhs_c: np.ndarray | None = None,
    phi_3d: np.ndarray | None = None,
    bc_top: np.ndarray | float | None = None,
    bc_bot: np.ndarray | float | None = None,
    bc_lateral: np.ndarray | None = None,
    maxit: int = 300,
    alpha: float = 0.93,
    resmax: float = 1e-14,
) -> tuple[np.ndarray, dict]:
    """Solve the QG omega equation using the SIP (Strongly Implicit Procedure).

    Matches the Li & O'Gorman (2020) solver from
    ``dante831/QG-omega/source/SIP_inversion.m``.

    Solves the σ-in-Laplacian form:

    .. math::

        \\nabla^2(\\sigma\\omega) + f_0^2 \\frac{\\partial^2\\omega}{\\partial p^2}
        = A + B + C

    where A is the Dostalek (2017) spherical Q-vector divergence,
    B = f₀β∂vg/∂p (direct form), and C = -(κ/p)∇²J (diabatic).

    The SIP stencil uses local σ(k,j,i) at **neighbor** points in the
    horizontal coefficients (AW, AE, AS, AN), and no σ in the vertical
    coefficients (AB, AT).

    **Boundary conditions**: Faces can be individually controlled:

    - *omega_b*: prescribe ERA5 ω on all Dirichlet faces (default mode).
    - *bc_top*/*bc_bot*: override top/bottom faces (e.g. 0.0 for dry solve).
    - *bc_lateral*: override N/S lateral faces (e.g. ω_bar climatology).

    **Diabatic heating (term C)**: When *rhs_c* is provided (pre-computed
    via :func:`_compute_diabatic_rhs_log20` or
    :func:`_compute_diabatic_rhs_emanuel`), it is added directly to
    the A+B RHS before solving.

    **Geostrophic temperature**: When *phi_3d* (geopotential) is
    provided, thermal-wind angular gradients of T_geo are used for
    the Q-vector, matching LOG20 ``GEO_T=true``.

    Args:
        u: Geostrophic zonal wind [m s⁻¹], ``(nlev, nlat, nlon)``.
        v: Geostrophic meridional wind [m s⁻¹], ``(nlev, nlat, nlon)``.
        t: Temperature [K], ``(nlev, nlat, nlon)``.
        lat: Ascending latitude [degrees], ``(nlat,)``.
        lon: Longitude [degrees], ``(nlon,)``.
        plevs_pa: Pressure [Pa], ascending, ``(nlev,)``.
        center_lat: Latitude for constant f₀ and β₀.
        omega_b: Boundary omega ``(nlev, nlat, nlon)`` (None → 0).
        t_full: Full physical temperature [K] for σ₀ and σ_3d.
        rhs_c: Pre-computed diabatic forcing C ``(nlev, nlat, nlon)``,
            -(κ/p)∇²J.  Added directly to A+B RHS (σ-in-Laplacian).
        phi_3d: Geopotential Φ [m² s⁻²], ``(nlev, nlat, nlon)``.
            When provided, T_geo from hydrostatic relation replaces
            actual T for Q-vector ∇T gradients.
        bc_top: Top boundary (k=0).  2D array or scalar.
        bc_bot: Bottom boundary (k=-1).  2D array or scalar.
        bc_lateral: Lateral boundary omega ``(nlev, nlat, nlon)``.
            Overrides N/S faces from omega_b.
        maxit: Max SIP iterations (default 300).
        alpha: SIP relaxation parameter (default 0.93).
        resmax: Convergence threshold (default 1e-14).

    Returns:
        ``(omega, info)`` where *omega* is ``(nlev, nlat, nlon)``
        QG vertical velocity [Pa s⁻¹] and *info* is a dict with
        ``'iters'``, ``'final_residual'``, ``'numba'``,
        ``'solve_time'``, ``'terms'``.

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

    # --- 1. Compute Q-vector RHS (non σ-divided, Dostalek spherical) ---
    rhs, sigma0, f0, beta0, lat_taper_full = _compute_qg_rhs(
        u, v, t, lat, lon, plevs_pa,
        center_lat=center_lat,
        t_full=t_full,
        phi_3d=phi_3d,
    )
    terms_used = "AB"

    # --- 1a. Add pre-computed diabatic forcing C ---
    # rhs_c is already -(κ/p)∇²_sph(J) — added directly (no σ division)
    # because the σ-in-Laplacian form keeps σ on the LHS.
    # Apply the same lat_taper as A+B to suppress polar singularities.
    if rhs_c is not None:
        rhs += rhs_c * lat_taper_full[None, :, None]
        terms_used = "ABC"

    # --- 1b. Local 3-D static stability σ(k,j,i) ---
    t_for_sigma = t_full if t_full is not None else t
    kappa = R_DRY / 1004.0
    sigma_3d = np.zeros((nlev, nlat, nlon))
    for k in range(1, nlev - 1):
        dp_s = plevs_pa[k + 1] - plevs_pa[k - 1]
        theta_kp1 = t_for_sigma[k + 1] * (1e5 / plevs_pa[k + 1]) ** kappa
        theta_km1 = t_for_sigma[k - 1] * (1e5 / plevs_pa[k - 1]) ** kappa
        dlnt = np.log(theta_kp1) - np.log(theta_km1)
        sigma_3d[k] = -(R_DRY * t_for_sigma[k] / plevs_pa[k]) * (dlnt / dp_s)
    sigma_3d[0] = sigma_3d[1]
    sigma_3d[-1] = sigma_3d[-2]
    sigma_3d = np.maximum(sigma_3d, 1e-7)

    # 1-2-1 horizontal smoothing (matches dante831 reference)
    for k in range(nlev):
        tmp = sigma_3d[k].copy()
        if nlat > 2:
            tmp[1:-1, :] = (
                0.25 * sigma_3d[k, :-2, :]
                + 0.5 * sigma_3d[k, 1:-1, :]
                + 0.25 * sigma_3d[k, 2:, :]
            )
        if nlon > 2:
            tmp2 = tmp.copy()
            tmp2[:, 1:-1] = (
                0.25 * tmp[:, :-2] + 0.5 * tmp[:, 1:-1] + 0.25 * tmp[:, 2:]
            )
            tmp = tmp2
        sigma_3d[k] = tmp

    # RHS is used directly — no σ₀→σ₃ᵈ rescaling needed (σ is on the LHS)

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
    #   σ-in-Laplacian form (matching LOG20 SIP_inversion.m):
    #     AW = σ(k,j,i-1) / (r²cos²φ Δλ²)     ← σ at WEST neighbor
    #     AE = σ(k,j,i+1) / (r²cos²φ Δλ²)     ← σ at EAST neighbor
    #     AS = (1/(r²Δφ²) + tanφ/(2r²Δφ)) σ(k,j-1,i)  ← σ at SOUTH
    #     AN = (1/(r²Δφ²) - tanφ/(2r²Δφ)) σ(k,j+1,i)  ← σ at NORTH
    #     AB = 2 f₀² / (Δp₁ (Δp₁+Δp₂))        ← NO σ
    #     AT = 2 f₀² / (Δp₂ (Δp₁+Δp₂))        ← NO σ
    #     AP = -2σ(k,j,i)/(r²cos²φΔλ²) - 2σ(k,j,i)/(r²Δφ²)
    #          - 2f₀²/(Δp₁ Δp₂)
    AP = np.zeros((nlev, nlat, nlon))
    AE = np.zeros((nlev, nlat, nlon))
    AW = np.zeros((nlev, nlat, nlon))
    AN = np.zeros((nlev, nlat, nlon))
    AS = np.zeros((nlev, nlat, nlon))
    AT = np.zeros((nlev, nlat, nlon))
    AB = np.zeros((nlev, nlat, nlon))

    cos2_phi = np.cos(phi) ** 2
    tan_phi = np.tan(phi)

    # Pre-compute shifted σ arrays for neighbor access
    sigma_west = np.roll(sigma_3d, 1, axis=2)   # σ(k, j, i-1)
    sigma_east = np.roll(sigma_3d, -1, axis=2)  # σ(k, j, i+1)

    for k in range(1, nlev - 1):
        dp1 = plevs_pa[k] - plevs_pa[k - 1]
        dp2 = plevs_pa[k + 1] - plevs_pa[k]

        for j in range(1, nlat - 1):
            c2 = cos2_phi[j]
            tp = tan_phi[j]

            ew = 1.0 / (r ** 2 * c2 * dlambda ** 2)
            ns_base = 1.0 / (r ** 2 * dphi ** 2)
            ns_tan  = tp / (2.0 * r ** 2 * dphi)

            # Horizontal coefficients — σ at NEIGHBOR (σ-in-Laplacian)
            AW[k, j, :] = ew * sigma_west[k, j, :]
            AE[k, j, :] = ew * sigma_east[k, j, :]
            AS[k, j, :] = (ns_base + ns_tan) * sigma_3d[k, j - 1, :]
            AN[k, j, :] = (ns_base - ns_tan) * sigma_3d[k, j + 1, :]

            # Vertical coefficients — NO σ (LOG20 SIP_inversion.m)
            AB[k, j, :] = 2.0 * f2_0 / (dp1 * (dp1 + dp2))
            AT[k, j, :] = 2.0 * f2_0 / (dp2 * (dp1 + dp2))

            # Diagonal: horizontal σ(center) + vertical (no σ)
            s_c = sigma_3d[k, j, :]
            AP[k, j, :] = (
                -2.0 * ew * s_c
                - 2.0 * ns_base * s_c
                - 2.0 * f2_0 / (dp1 * dp2)
            )

    # --- 4. Prepare solution array with BCs ---
    T_sol = np.zeros((nlev, nlat, nlon))
    # Top / bottom boundaries
    if bc_top is not None:
        T_sol[0, :, :] = bc_top
    elif omega_b is not None:
        T_sol[0, :, :] = omega_b[0, :, :]
    if bc_bot is not None:
        T_sol[-1, :, :] = bc_bot
    elif omega_b is not None:
        T_sol[-1, :, :] = omega_b[-1, :, :]
    # Lateral (N/S faces)
    lat_src = bc_lateral if bc_lateral is not None else omega_b
    if lat_src is not None:
        T_sol[:, 0, :]  = lat_src[:, 0, :]
        T_sol[:, -1, :] = lat_src[:, -1, :]
    # For Dirichlet lon BCs, also fix i=0 and i=-1
    if not periodic_lon and lat_src is not None:
        T_sol[:, :, 0]  = lat_src[:, :, 0]
        T_sol[:, :, -1] = lat_src[:, :, -1]

    # --- 5. Solve with SIP core ---
    n_iters, final_res = _sip_core(
        AP, AE, AW, AN, AS, AT, AB, rhs, T_sol,
        nlev, nlat, nlon, alpha, maxit, resmax,
        periodic_lon,
    )

    omega_out = T_sol

    # --- 6. Apply latitude taper ---
    omega_out *= lat_taper_full[None, :, None]

    # --- 7. Remove zonal mean (match FFT where m=0 is excluded) ---
    # Only remove zonal mean when longitude covers a full ring (~360°);
    # on a local patch the "zonal mean" is NOT the true m=0 wavenumber
    # and removing it would destroy the signal.
    if periodic_lon:
        zmean = np.nanmean(omega_out, axis=-1, keepdims=True)
        omega_out -= zmean

    # --- 8. Clean NaN / inf ---
    omega_out = np.nan_to_num(omega_out, nan=0.0, posinf=0.0, neginf=0.0)

    elapsed = _time.perf_counter() - t0_wall
    info = {
        "iters": n_iters,
        "final_residual": float(final_res),
        "numba": _HAS_NUMBA,
        "solve_time": elapsed,
        "terms": terms_used,
    }
    return omega_out, info
