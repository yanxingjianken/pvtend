"""Helmholtz decomposition for wind fields on a latitude-longitude grid.

Decomposes a 2-D wind field (u, v) into three orthogonal parts:

    u = u_rot + u_div + u_har
    v = v_rot + v_div + v_har

where
    rotational   (non-divergent):  u_rot = −∂ψ/∂y,  v_rot =  ∂ψ/∂x
    divergent    (irrotational):   u_div =  ∂χ/∂x,  v_div =  ∂χ/∂y
    harmonic     (residual):       ∇·u_har = 0  AND  ζ(u_har) = 0

Poisson solver:
    ``'spherical'``: Full spherical Laplacian (conservative form,
    xinvert/MiniUFO), periodic in λ, Dirichlet in φ.  All other
    backends (direct, fft, dct, sor) are deprecated.

All horizontal derivatives use the canonical operators from
:mod:`pvtend.derivatives` (``ddx`` / ``ddy``) with periodic zonal
boundary conditions matching the full NH ring.

References:
    Lynch P (1989) MWR 117, 1492-1500.
    Schumann U & Sweet R (1988) J. Comput. Phys. 75, 123-137.
    MiniUFO/xinvert — conservative-form spherical Laplacian.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# sparse, splinalg, scipy_fft — no longer needed (deprecated solvers)
# from scipy import sparse
# from scipy.sparse import linalg as splinalg
# from scipy import fft as scipy_fft

from .constants import R_EARTH
from .derivatives import ddx, ddy


# ═══════════════════════════════════════════════════════════════
#  Differential operators — thin wrappers around derivatives.py
# ═══════════════════════════════════════════════════════════════


def compute_vorticity_divergence(
    u: np.ndarray,
    v: np.ndarray,
    dx: np.ndarray,
    dy: float,
    lat: np.ndarray | None = None,
    R_earth: float = R_EARTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical vorticity and divergence.

    .. math::

        \\zeta = \\partial v/\\partial x - \\partial u/\\partial y
                + u \\tan\\varphi / a

        \\delta = \\partial u/\\partial x + \\partial v/\\partial y
                - v \\tan\\varphi / a

    Uses periodic zonal BCs (matching the full-NH ring) and one-sided
    differences at the meridional boundaries, via :func:`derivatives.ddx`
    and :func:`derivatives.ddy`.

    Args:
        u: Zonal wind, shape ``(nlat, nlon)``.
        v: Meridional wind, shape ``(nlat, nlon)``.
        dx: Zonal grid spacing [m]. Scalar or array of shape ``(nlat,)``.
        dy: Meridional grid spacing [m].
        lat: Latitude in **degrees**, shape ``(nlat,)``.  When *None*
            the spherical metric terms (tan φ / a) are omitted
            (flat-earth fallback, deprecated).
        R_earth: Earth radius [m].

    Returns:
        ``(vorticity, divergence)`` — each ``(nlat, nlon)``.
    """
    nlat = u.shape[0]
    dx_arr = np.full(nlat, float(dx)) if np.isscalar(dx) else np.asarray(dx, float).ravel()

    du_dx = ddx(u, dx_arr, periodic=True)
    dv_dx = ddx(v, dx_arr, periodic=True)
    du_dy = ddy(u, dy)
    dv_dy = ddy(v, dy)

    vort = dv_dx - du_dy
    div = du_dx + dv_dy

    if lat is not None:
        tan_over_a = (np.tan(np.deg2rad(lat)) / R_earth)[:, np.newaxis]
        vort = vort + u * tan_over_a
        div = div - v * tan_over_a

    return vort, div


def gradient(
    phi: np.ndarray,
    dx: np.ndarray,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Physical gradient (∂φ/∂x, ∂φ/∂y) — spectral zonal, FD meridional.

    The zonal derivative uses a spectral (FFT) method so that the
    composition div(grad(χ)) is consistent with the compact Laplacian
    stencil used in :func:`solve_poisson_spherical_fft`.  The
    meridional derivative keeps centred finite differences via
    :func:`derivatives.ddy`.

    .. note::

       The meridional derivative uses plain centred FD, which does NOT
       match the conservative half-grid stencil in the Poisson solver.
       Therefore ``div_FD(gradient(χ)) ≠ ∇²χ``.  For a consistent
       round-trip use :func:`laplacian_spherical_fft` instead.

    Args:
        phi: Scalar field, shape ``(nlat, nlon)``.
        dx: Zonal grid spacing [m]. Scalar or array of shape ``(nlat,)``.
            When an array, ``dx[j] = a · cos(φ_j) · Δλ``.
        dy: Meridional grid spacing [m].

    Returns:
        ``(dphi_dx, dphi_dy)`` — each ``(nlat, nlon)``.
    """
    nlat, nlon = phi.shape
    dx_arr = np.full(nlat, float(dx)) if np.isscalar(dx) else np.asarray(dx, float).ravel()

    # Zonal: spectral derivative via FFT (exact for the periodic ring)
    phi_hat = np.fft.rfft(phi, axis=1)
    m = np.arange(phi_hat.shape[1])            # wavenumber indices
    # Derivative of trig interpolant in grid-index space:  d/dn → i·2πm/N
    deriv_coeff = 1j * 2.0 * np.pi * m / nlon
    # Zero the Nyquist mode for even N to avoid sign ambiguity
    if nlon % 2 == 0:
        deriv_coeff[-1] = 0.0
    dphi_dn = np.fft.irfft(phi_hat * deriv_coeff[np.newaxis, :], n=nlon, axis=1)
    # Convert from per-grid-point to per-metre
    dphi_dx = dphi_dn / dx_arr[:, np.newaxis]

    # Meridional: centred FD (non-periodic direction)
    dphi_dy = ddy(phi, dy)

    return dphi_dx, dphi_dy


def laplacian_spherical_fft(
    phi: np.ndarray,
    lat: np.ndarray,
    dy: float,
    dlon_rad: float,
    R_earth: float = R_EARTH,
) -> np.ndarray:
    """Apply the spherical Laplacian using the SAME stencil as the Poisson solver.

    Computes :math:`\\nabla^2 \\phi` using the conservative-form spherical
    Laplacian identical to the one in :func:`solve_poisson_spherical_fft`:

    .. math::

        \\nabla^2 \\phi = \\frac{1}{a^2 \\cos^2\\varphi}
        \\frac{\\partial^2 \\phi}{\\partial \\lambda^2}
        + \\frac{1}{a^2 \\cos\\varphi}
        \\frac{\\partial}{\\partial \\varphi}
        \\left(\\cos\\varphi
        \\frac{\\partial \\phi}{\\partial \\varphi}\\right)

    This function is the *analysis* (forward) operator conjugate to the
    Poisson solver, so ``laplacian_spherical_fft(solve_poisson(..., f), ...)
    == f`` to machine precision (interior points).

    Args:
        phi: Scalar field, shape ``(nlat, nlon)``.
        lat: Latitude in degrees, shape ``(nlat,)``.
        dy: Meridional grid spacing :math:`a \\Delta\\varphi` [m].
        dlon_rad: Longitudinal grid spacing :math:`\\Delta\\lambda` [radians].
        R_earth: Earth radius [m].

    Returns:
        :math:`\\nabla^2 \\phi`, shape ``(nlat, nlon)``.
        Boundary rows (j=0 and j=-1) are set to zero (Dirichlet).
    """
    nlat, nlon = phi.shape
    lat_rad = np.deg2rad(lat)
    cos_phi = np.cos(lat_rad)
    dphi = dy / R_earth
    dphi2 = dphi * dphi
    dlam2 = dlon_rad * dlon_rad

    cos_half = np.cos(0.5 * (lat_rad[:-1] + lat_rad[1:]))

    # Zonal second derivative via FFT (same discrete form as solver)
    phi_hat = np.fft.rfft(phi, axis=1)
    m = np.arange(phi_hat.shape[1])
    lam_k_full = 2.0 * (np.cos(2.0 * np.pi * m / nlon) - 1.0)  # eigenvalues
    d2phi_dlam2 = np.fft.irfft(phi_hat * lam_k_full[None, :], n=nlon, axis=1)

    # Meridional conservative form with half-grid cosines
    dphi_dphi_half = np.diff(phi, axis=0) / dphi
    flux = cos_half[:, None] * dphi_dphi_half
    merid = np.zeros_like(phi)
    merid[1:-1] = np.diff(flux, axis=0) / dphi

    lap = (d2phi_dlam2 / (R_earth**2 * cos_phi[:, None]**2 * dlam2)
           + merid / (R_earth**2 * cos_phi[:, None]))

    # Boundary rows: zero (consistent with Dirichlet BCs)
    lap[0, :] = 0.0
    lap[-1, :] = 0.0
    return lap


# ═══════════════════════════════════════════════════════════════
#  Poisson solvers
# ═══════════════════════════════════════════════════════════════


# ── DEPRECATED: solve_poisson_direct ────────────────────────────────
# Flat-earth sparse LU solver. Replaced by solve_poisson_spherical_fft.
# Kept commented-out for reference.
#
# def solve_poisson_direct(
#     rhs: np.ndarray, dx: np.ndarray, dy: float,
# ) -> np.ndarray:
#     """Solve ∇²φ = rhs with Dirichlet φ = 0 (flat-earth, Lynch 1989)."""
#     nlat, nlon = rhs.shape
#     dx_arr = np.full(nlat, float(dx)) if np.isscalar(dx) else np.asarray(dx, float).ravel()
#     dy2 = dy * dy; ni, nj = nlat - 2, nlon - 2; N = ni * nj
#     if N == 0: return np.zeros_like(rhs)
#     rows, cols, vals, b = [], [], [], np.zeros(N)
#     for ii in range(ni):
#         jg = ii + 1; dx2 = dx_arr[jg] ** 2; cx, cy = 1.0/dx2, 1.0/dy2
#         diag = -2.0 * (cx + cy)
#         for jj in range(nj):
#             k = ii*nj + jj; rows.append(k); cols.append(k); vals.append(diag)
#             if jj > 0: rows.append(k); cols.append(k-1); vals.append(cx)
#             if jj < nj-1: rows.append(k); cols.append(k+1); vals.append(cx)
#             if ii > 0: rows.append(k); cols.append(k-nj); vals.append(cy)
#             if ii < ni-1: rows.append(k); cols.append(k+nj); vals.append(cy)
#             b[k] = rhs[jg, jj + 1]
#     A = sparse.csc_matrix((vals, (rows, cols)), shape=(N, N))
#     x = splinalg.spsolve(A, b)
#     phi = np.zeros_like(rhs); phi[1:-1, 1:-1] = x.reshape(ni, nj)
#     return phi


# ── DEPRECATED: solve_poisson_fft ───────────────────────────────────
# Flat-earth FFT solver. Replaced by solve_poisson_spherical_fft.
# Kept commented-out for reference.
#
# def solve_poisson_fft(rhs, dx, dy):
#     """Flat-earth ∇²φ = rhs — periodic lon, Dirichlet lat (Schumann & Sweet 1988)."""
#     nlat, nlon = rhs.shape
#     dx_arr = np.full(nlat, float(dx)) if np.isscalar(dx) else np.asarray(dx, float).ravel()
#     dy2 = dy * dy; ni = nlat - 2
#     if ni == 0: return np.zeros_like(rhs)
#     rhs_hat = np.fft.fft(rhs, axis=1)
#     phi_hat = np.zeros_like(rhs_hat)
#     for kk in range(nlon):
#         lam_k = 2.0 * (np.cos(2.0 * np.pi * kk / nlon) - 1.0)
#         a = np.full(ni, 1.0/dy2); c = np.full(ni, 1.0/dy2)
#         b = np.empty(ni); f = np.empty(ni, dtype=complex)
#         for j in range(ni):
#             jg = j + 1
#             b[j] = lam_k / (dx_arr[jg]**2) - 2.0/dy2
#             f[j] = rhs_hat[jg, kk]
#         phi_hat[1:-1, kk] = _thomas(a, b, c, f)
#     return np.fft.ifft(phi_hat, axis=1).real


def solve_poisson_spherical_fft(
    rhs: np.ndarray,
    lat: np.ndarray,
    dy: float,
    dlon_rad: float,
    R_earth: float = R_EARTH,
) -> np.ndarray:
    """Solve the spherical Poisson equation — periodic in λ, Dirichlet in φ.

    Solves the full spherical Laplacian on a lat-lon grid:

        (1/(a²cos²φ)) ∂²χ/∂λ² + (1/a²cosφ) ∂/∂φ(cosφ ∂χ/∂φ) = f

    Uses the **conservative (divergence) form** for the meridional
    operator — following xinvert (MiniUFO) — which naturally includes
    the −tan(φ) curvature term without explicit first-derivative
    discretisation.  Multiply through by a²cosφ:

        (1/cosφ) ∂²χ/∂λ² + ∂/∂φ(cosφ ∂χ/∂φ) = f · a² · cosφ

    The meridional tridiagonal coefficients become:

        a_j = cos(φ_{j−½}) / Δφ²
        c_j = cos(φ_{j+½}) / Δφ²
        b_j = λ_k / (cos(φ_j) · Δλ²) − (a_j + c_j) / Δφ²

    (but note a_j, c_j already contain 1/Δφ², stored without it below).

    Args:
        rhs: Right-hand side of ∇²χ = rhs, shape ``(nlat, nlon)``.
        lat: Latitude in **degrees**, shape ``(nlat,)``.
        dy: Meridional grid spacing a·Δφ [m].
        dlon_rad: Longitudinal grid spacing Δλ in **radians**.
        R_earth: Earth radius [m].

    Returns:
        Solution χ with Dirichlet BCs at lat boundaries,
        shape ``(nlat, nlon)``.
    """
    nlat, nlon = rhs.shape
    ni = nlat - 2
    if ni == 0:
        return np.zeros_like(rhs)

    lat_rad = np.deg2rad(lat)
    cos_phi = np.cos(lat_rad)                      # (nlat,)
    dphi = dy / R_earth                             # Δφ in radians
    dphi2 = dphi * dphi
    dlam2 = dlon_rad * dlon_rad

    # Half-grid cosines: cos(φ_{j+½})
    cos_half = np.cos(0.5 * (lat_rad[:-1] + lat_rad[1:]))  # (nlat-1,)

    # ── Scale RHS: f_scaled = f · a² · cosφ  (but our rhs is already
    #    in physical units of 1/s, so scale by R_earth² · cosφ)
    rhs_scaled = rhs * (R_earth ** 2) * cos_phi[:, None]

    rhs_hat = np.fft.fft(rhs_scaled, axis=1)
    phi_hat = np.zeros_like(rhs_hat)

    for kk in range(nlon):
        lam_k = 2.0 * (np.cos(2.0 * np.pi * kk / nlon) - 1.0)

        a_vec = np.empty(ni)     # sub-diagonal
        c_vec = np.empty(ni)     # super-diagonal
        b_vec = np.empty(ni)     # main diagonal
        f_vec = np.empty(ni, dtype=complex)

        for j in range(ni):
            jg = j + 1                             # global index
            a_vec[j] = cos_half[jg - 1] / dphi2       # cos(φ_{j-½})
            c_vec[j] = cos_half[jg] / dphi2           # cos(φ_{j+½})
            b_vec[j] = lam_k / (cos_phi[jg] * dlam2) - (a_vec[j] + c_vec[j])
            f_vec[j] = rhs_hat[jg, kk]

        phi_hat[1:-1, kk] = _thomas(a_vec, b_vec, c_vec, f_vec)

    return np.fft.ifft(phi_hat, axis=1).real


# ── DEPRECATED: solve_poisson_dct ───────────────────────────────────
# DST-I with constant dx. Replaced by solve_poisson_spherical_fft.
#
# def solve_poisson_dct(rhs, dx_mean, dy, **kw):
#     nlat, nlon = rhs.shape
#     kx = np.arange(1, nlon + 1); ky = np.arange(1, nlat + 1)
#     KX, KY = np.meshgrid(kx, ky)
#     lam = (2*(np.cos(np.pi*KX/(nlon+1))-1)/dx_mean**2
#            + 2*(np.cos(np.pi*KY/(nlat+1))-1)/dy**2)
#     rhs_hat = scipy_fft.dstn(rhs, type=1)
#     phi_hat = np.zeros_like(rhs_hat)
#     mask = np.abs(lam) > 1e-14
#     phi_hat[mask] = rhs_hat[mask] / lam[mask]
#     return scipy_fft.idstn(phi_hat, type=1)


# ── DEPRECATED: solve_poisson_sor ───────────────────────────────────
# SOR iterative solver. Replaced by solve_poisson_spherical_fft.
#
# def solve_poisson_sor(rhs, dx, dy, omega=1.8, tol=1e-6, max_iter=10000):
#     nlat, nlon = rhs.shape
#     dx_arr = np.full(nlat, float(dx)) if np.isscalar(dx) else np.asarray(dx, float).ravel()
#     dy2 = dy * dy; phi = np.zeros_like(rhs)
#     for _ in range(max_iter):
#         max_diff = 0.0
#         for j in range(1, nlat - 1):
#             dx2 = dx_arr[j] ** 2; coef = 2.0 * (1.0/dx2 + 1.0/dy2)
#             for i in range(1, nlon - 1):
#                 old = phi[j, i]
#                 phi_new = ((phi[j,i+1]+phi[j,i-1])/dx2
#                            + (phi[j+1,i]+phi[j-1,i])/dy2 - rhs[j,i]) / coef
#                 phi[j, i] = old + omega * (phi_new - old)
#                 max_diff = max(max_diff, abs(phi[j, i] - old))
#         if max_diff < tol: break
#     return phi


def _thomas(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system.

    Solves the system where *a*, *b*, *c* are sub-, main-, and
    super-diagonal and *d* is the RHS.

    Args:
        a: Sub-diagonal, shape ``(n,)``.
        b: Main diagonal, shape ``(n,)``.
        c: Super-diagonal, shape ``(n,)``.
        d: RHS vector, shape ``(n,)``.

    Returns:
        Solution ``x``, shape ``(n,)``.
    """
    n = len(d)
    cp = np.zeros(n, dtype=complex)
    dp = np.zeros(n, dtype=complex)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x = np.zeros(n, dtype=complex)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ═══════════════════════════════════════════════════════════════
#  Main decomposition routines
# ═══════════════════════════════════════════════════════════════


def helmholtz_decomposition(
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    R_earth: float = R_EARTH,
    method: str = "spherical",
    solver_kw: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """Helmholtz decomposition on a lat-lon grid (spherical Laplacian).

    Computes vorticity ζ and divergence δ from (u, v), removes the
    area-weighted (cosφ) mean from each before solving the Poisson
    equations ∇²ψ = ζ and ∇²χ = δ using the full spherical Laplacian
    (conservative form, following xinvert/MiniUFO).

    Args:
        u: Zonal wind [m s⁻¹], shape ``(nlat, nlon)``.
        v: Meridional wind [m s⁻¹], shape ``(nlat, nlon)``.
        lat: Latitude in degrees (ascending), shape ``(nlat,)``.
        lon: Longitude in degrees, shape ``(nlon,)``.
        R_earth: Earth radius [m].
        method: Ignored (kept for API compat). Always uses spherical.
        solver_kw: Ignored (kept for API compat).

    Returns:
        Dictionary with keys ``u_rot``, ``v_rot``, ``u_div``, ``v_div``,
        ``u_har``, ``v_har``, ``chi``, ``psi``, ``vorticity``,
        ``divergence`` — each ``(nlat, nlon)``.

    References:
        MiniUFO/xinvert — conservative-form spherical Laplacian.
    """
    nan_mask = np.isnan(u) | np.isnan(v)
    u_work = np.where(nan_mask, 0.0, u) if nan_mask.any() else u.copy()
    v_work = np.where(nan_mask, 0.0, v) if nan_mask.any() else v.copy()

    nlat, nlon = u.shape
    lat_rad = np.deg2rad(lat)
    dlat = np.abs(lat[1] - lat[0]) if nlat > 1 else 1.5
    dlon = np.abs(lon[1] - lon[0]) if nlon > 1 else 1.5

    dy = np.deg2rad(dlat) * R_earth
    dx = np.deg2rad(dlon) * R_earth * np.cos(lat_rad)
    dx = np.maximum(dx, dy * 0.1)
    dlon_rad = np.deg2rad(dlon)

    vort, div = compute_vorticity_divergence(u_work, v_work, dx, dy,
                                               lat=lat, R_earth=R_earth)

    # ── Area-weighted mean removal (Fredholm solvability) ──
    cos_phi = np.cos(lat_rad)
    area_weights = cos_phi / cos_phi.sum()        # (nlat,)
    div_mean = np.sum(area_weights[:, None] * div) / nlon
    vort_mean = np.sum(area_weights[:, None] * vort) / nlon
    div = div - div_mean
    vort = vort - vort_mean

    # ── Spherical Poisson solve ──
    chi = solve_poisson_spherical_fft(div, lat, dy, dlon_rad, R_earth=R_earth)
    psi = solve_poisson_spherical_fft(vort, lat, dy, dlon_rad, R_earth=R_earth)

    dchi_dx, dchi_dy = gradient(chi, dx, dy)
    dpsi_dx, dpsi_dy = gradient(psi, dx, dy)

    u_rot, v_rot = -dpsi_dy, dpsi_dx
    u_div, v_div = dchi_dx, dchi_dy
    u_har = u_work - u_rot - u_div
    v_har = v_work - v_rot - v_div

    if nan_mask.any():
        for arr in (u_rot, v_rot, u_div, v_div, u_har, v_har, chi, psi, vort, div):
            arr[nan_mask] = np.nan

    return dict(
        u_rot=u_rot,
        v_rot=v_rot,
        u_div=u_div,
        v_div=v_div,
        u_har=u_har,
        v_har=v_har,
        chi=chi,
        psi=psi,
        vorticity=vort,
        divergence=div,
    )


def helmholtz_decomposition_3d(
    u_3d: np.ndarray,
    v_3d: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    R_earth: float = R_EARTH,
    method: str = "spherical",
) -> dict[str, np.ndarray]:
    """Helmholtz decomposition for each vertical level.

    Applies :func:`helmholtz_decomposition` independently to every
    level along axis 0, using the spherical Laplacian solver.

    Args:
        u_3d: Zonal wind [m s⁻¹], shape ``(nlevels, nlat, nlon)``.
        v_3d: Meridional wind [m s⁻¹], shape ``(nlevels, nlat, nlon)``.
        lat: Latitude in degrees (ascending), shape ``(nlat,)``.
        lon: Longitude in degrees, shape ``(nlon,)``.
        R_earth: Earth radius [m].
        method: Ignored (always spherical).

    Returns:
        Dictionary with the same keys as :func:`helmholtz_decomposition`,
        each ``(nlevels, nlat, nlon)``.
    """
    nlevels = u_3d.shape[0]
    keys = [
        "u_rot",
        "v_rot",
        "u_div",
        "v_div",
        "u_har",
        "v_har",
        "chi",
        "psi",
        "vorticity",
        "divergence",
    ]
    out: dict[str, np.ndarray] = {k: np.zeros_like(u_3d) for k in keys}
    for lev in range(nlevels):
        res = helmholtz_decomposition(
            u_3d[lev],
            v_3d[lev],
            lat,
            lon,
            R_earth=R_earth,
        )
        for k in keys:
            out[k][lev] = res[k]
    return out
