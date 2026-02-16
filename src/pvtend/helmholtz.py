"""Helmholtz decomposition for wind fields on a latitude-longitude grid.

Decomposes a 2-D wind field (u, v) into three orthogonal parts:

    u = u_rot + u_div + u_har
    v = v_rot + v_div + v_har

where
    rotational   (non-divergent):  u_rot = −∂ψ/∂y,  v_rot =  ∂ψ/∂x
    divergent    (irrotational):   u_div =  ∂χ/∂x,  v_div =  ∂χ/∂y
    harmonic     (residual):       ∇·u_har = 0  AND  ζ(u_har) = 0

Four Poisson solver backends:
    - ``'direct'``: Sparse LU (Lynch 1989), Dirichlet BCs, exact variable dx
    - ``'fft'``:    FFT in longitude + Thomas in latitude (periodic + Dirichlet)
    - ``'dct'``:    DST-I with mean(dx), fast but approximate at high latitudes
    - ``'sor'``:    SOR iteration, slow reference

References:
    Lynch P (1989) MWR 117, 1492-1500.
    Schumann U & Sweet R (1988) J. Comput. Phys. 75, 123-137.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy import fft as scipy_fft

from .constants import R_EARTH


# ═══════════════════════════════════════════════════════════════
#  Basic differential operators (self-contained for this module)
# ═══════════════════════════════════════════════════════════════


def compute_vorticity_divergence(
    u: np.ndarray,
    v: np.ndarray,
    dx: np.ndarray,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative vorticity ζ = ∂v/∂x − ∂u/∂y and divergence δ = ∂u/∂x + ∂v/∂y.

    Uses centred differences in the interior and one-sided differences
    at the edges.

    Args:
        u: Zonal wind, shape ``(nlat, nlon)``.
        v: Meridional wind, shape ``(nlat, nlon)``.
        dx: Zonal grid spacing [m]. Scalar or array of shape ``(nlat,)``.
        dy: Meridional grid spacing [m].

    Returns:
        ``(vorticity, divergence)`` — each ``(nlat, nlon)``.
    """
    nlat, nlon = u.shape
    dx_arr = np.full(nlat, float(dx)) if np.isscalar(dx) else np.asarray(dx, float).ravel()

    du_dx = np.zeros_like(u)
    du_dy = np.zeros_like(u)
    dv_dx = np.zeros_like(v)
    dv_dy = np.zeros_like(v)

    for j in range(nlat):
        du_dx[j, 1:-1] = (u[j, 2:] - u[j, :-2]) / (2 * dx_arr[j])
        dv_dx[j, 1:-1] = (v[j, 2:] - v[j, :-2]) / (2 * dx_arr[j])
        du_dx[j, 0] = (u[j, 1] - u[j, 0]) / dx_arr[j]
        du_dx[j, -1] = (u[j, -1] - u[j, -2]) / dx_arr[j]
        dv_dx[j, 0] = (v[j, 1] - v[j, 0]) / dx_arr[j]
        dv_dx[j, -1] = (v[j, -1] - v[j, -2]) / dx_arr[j]

    du_dy[1:-1] = (u[2:] - u[:-2]) / (2 * dy)
    dv_dy[1:-1] = (v[2:] - v[:-2]) / (2 * dy)
    du_dy[0] = (u[1] - u[0]) / dy
    du_dy[-1] = (u[-1] - u[-2]) / dy
    dv_dy[0] = (v[1] - v[0]) / dy
    dv_dy[-1] = (v[-1] - v[-2]) / dy

    return dv_dx - du_dy, du_dx + dv_dy


def gradient(
    phi: np.ndarray,
    dx: np.ndarray,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(∂φ/∂x, ∂φ/∂y) with centred differences, one-sided at edges.

    Args:
        phi: Scalar field, shape ``(nlat, nlon)``.
        dx: Zonal grid spacing [m]. Scalar or array of shape ``(nlat,)``.
        dy: Meridional grid spacing [m].

    Returns:
        ``(dphi_dx, dphi_dy)`` — each ``(nlat, nlon)``.
    """
    nlat, nlon = phi.shape
    dx_arr = np.full(nlat, float(dx)) if np.isscalar(dx) else np.asarray(dx, float).ravel()

    dphi_dx = np.zeros_like(phi)
    dphi_dy = np.zeros_like(phi)

    for j in range(nlat):
        dphi_dx[j, 1:-1] = (phi[j, 2:] - phi[j, :-2]) / (2 * dx_arr[j])
        dphi_dx[j, 0] = (phi[j, 1] - phi[j, 0]) / dx_arr[j]
        dphi_dx[j, -1] = (phi[j, -1] - phi[j, -2]) / dx_arr[j]

    dphi_dy[1:-1] = (phi[2:] - phi[:-2]) / (2 * dy)
    dphi_dy[0] = (phi[1] - phi[0]) / dy
    dphi_dy[-1] = (phi[-1] - phi[-2]) / dy

    return dphi_dx, dphi_dy


# ═══════════════════════════════════════════════════════════════
#  Poisson solvers
# ═══════════════════════════════════════════════════════════════


def solve_poisson_direct(
    rhs: np.ndarray,
    dx: np.ndarray,
    dy: float,
) -> np.ndarray:
    """Solve ∇²φ = rhs with Dirichlet φ = 0 on all four boundaries.

    Builds the 5-point finite-difference Laplacian as a sparse matrix
    and solves with ``scipy.sparse.linalg.spsolve`` (direct LU).  Variable
    ``dx(lat)`` is handled exactly.

    Args:
        rhs: Right-hand side, shape ``(nlat, nlon)``.
        dx: Zonal grid spacing [m]. Scalar or array of shape ``(nlat,)``.
        dy: Meridional grid spacing [m].

    Returns:
        Solution φ with Dirichlet BCs, shape ``(nlat, nlon)``.

    References:
        Lynch P (1989) MWR 117, 1492-1500.
    """
    nlat, nlon = rhs.shape
    dx_arr = (
        np.full(nlat, float(dx)) if np.isscalar(dx) else np.asarray(dx, float).ravel()
    )
    dy2 = dy * dy
    ni, nj = nlat - 2, nlon - 2
    N = ni * nj
    if N == 0:
        return np.zeros_like(rhs)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(N)

    for ii in range(ni):
        jg = ii + 1
        dx2 = dx_arr[jg] ** 2
        cx, cy = 1.0 / dx2, 1.0 / dy2
        diag = -2.0 * (cx + cy)
        for jj in range(nj):
            k = ii * nj + jj
            rows.append(k)
            cols.append(k)
            vals.append(diag)
            if jj > 0:
                rows.append(k)
                cols.append(k - 1)
                vals.append(cx)
            if jj < nj - 1:
                rows.append(k)
                cols.append(k + 1)
                vals.append(cx)
            if ii > 0:
                rows.append(k)
                cols.append(k - nj)
                vals.append(cy)
            if ii < ni - 1:
                rows.append(k)
                cols.append(k + nj)
                vals.append(cy)
            b[k] = rhs[jg, jj + 1]

    A = sparse.csc_matrix((vals, (rows, cols)), shape=(N, N))
    x = splinalg.spsolve(A, b)

    phi = np.zeros_like(rhs)
    phi[1:-1, 1:-1] = x.reshape(ni, nj)
    return phi


def solve_poisson_fft(
    rhs: np.ndarray,
    dx: np.ndarray,
    dy: float,
) -> np.ndarray:
    """Solve ∇²φ = rhs — periodic in longitude, Dirichlet in latitude.

    1. FFT each row → eigenvalues for the x-derivative.
    2. For each wavenumber *k*, solve a tridiagonal system in latitude
       with variable ``dx(lat)``.
    3. Inverse FFT back to physical space.

    Args:
        rhs: Right-hand side, shape ``(nlat, nlon)``.
        dx: Zonal grid spacing [m]. Scalar or array of shape ``(nlat,)``.
        dy: Meridional grid spacing [m].

    Returns:
        Solution φ, shape ``(nlat, nlon)``.

    References:
        Schumann U & Sweet R (1988) J. Comput. Phys. 75, 123-137.
    """
    nlat, nlon = rhs.shape
    dx_arr = (
        np.full(nlat, float(dx)) if np.isscalar(dx) else np.asarray(dx, float).ravel()
    )
    dy2 = dy * dy
    ni = nlat - 2
    if ni == 0:
        return np.zeros_like(rhs)

    rhs_hat = np.fft.fft(rhs, axis=1)
    phi_hat = np.zeros_like(rhs_hat)

    for kk in range(nlon):
        lam_k = 2.0 * (np.cos(2.0 * np.pi * kk / nlon) - 1.0)

        # Tridiagonal: a_j φ_{j-1} + b_j φ_j + c_j φ_{j+1} = f_j
        a = np.full(ni, 1.0 / dy2)
        c = np.full(ni, 1.0 / dy2)
        b = np.empty(ni)
        f = np.empty(ni, dtype=complex)

        for j in range(ni):
            jg = j + 1
            b[j] = lam_k / (dx_arr[jg] ** 2) - 2.0 / dy2
            f[j] = rhs_hat[jg, kk]

        phi_hat[1:-1, kk] = _thomas(a, b, c, f)

    return np.fft.ifft(phi_hat, axis=1).real


def solve_poisson_dct(
    rhs: np.ndarray,
    dx_mean: float,
    dy: float,
) -> np.ndarray:
    """DST-I Poisson solver with constant dx.

    Solve ∇²φ = rhs with Dirichlet φ = 0 on all boundaries using a
    Discrete Sine Transform (type I).  Uses **constant** ``dx_mean``
    — fast O(N log N) but loses accuracy at high latitudes where dx
    varies with cos(lat).

    Args:
        rhs: Right-hand side, shape ``(nlat, nlon)``.
        dx_mean: Mean zonal grid spacing [m].
        dy: Meridional grid spacing [m].

    Returns:
        Solution φ, shape ``(nlat, nlon)``.
    """
    nlat, nlon = rhs.shape
    kx = np.arange(1, nlon + 1)
    ky = np.arange(1, nlat + 1)
    KX, KY = np.meshgrid(kx, ky)
    lam = (
        2 * (np.cos(np.pi * KX / (nlon + 1)) - 1) / (dx_mean ** 2)
        + 2 * (np.cos(np.pi * KY / (nlat + 1)) - 1) / (dy ** 2)
    )
    rhs_hat = scipy_fft.dstn(rhs, type=1)
    phi_hat = np.zeros_like(rhs_hat)
    mask = np.abs(lam) > 1e-14
    phi_hat[mask] = rhs_hat[mask] / lam[mask]
    return scipy_fft.idstn(phi_hat, type=1)


def solve_poisson_sor(
    rhs: np.ndarray,
    dx: np.ndarray,
    dy: float,
    omega: float = 1.8,
    tol: float = 1e-6,
    max_iter: int = 10000,
) -> np.ndarray:
    """SOR iterative Poisson solver.

    Solve ∇²φ = rhs with Dirichlet φ = 0 using Successive
    Over-Relaxation.  Same discrete 5-point Laplacian as
    :func:`solve_poisson_direct`, but solved iteratively.
    ~1000× slower; kept only for verification.

    Args:
        rhs: Right-hand side, shape ``(nlat, nlon)``.
        dx: Zonal grid spacing [m]. Scalar or array of shape ``(nlat,)``.
        dy: Meridional grid spacing [m].
        omega: SOR relaxation parameter.
        tol: Convergence tolerance on max absolute update.
        max_iter: Maximum number of SOR iterations.

    Returns:
        Solution φ, shape ``(nlat, nlon)``.
    """
    nlat, nlon = rhs.shape
    dx_arr = (
        np.full(nlat, float(dx)) if np.isscalar(dx) else np.asarray(dx, float).ravel()
    )
    dy2 = dy * dy
    phi = np.zeros_like(rhs)
    for _ in range(max_iter):
        max_diff = 0.0
        for j in range(1, nlat - 1):
            dx2 = dx_arr[j] ** 2
            coef = 2.0 * (1.0 / dx2 + 1.0 / dy2)
            for i in range(1, nlon - 1):
                old = phi[j, i]
                phi_new = (
                    (phi[j, i + 1] + phi[j, i - 1]) / dx2
                    + (phi[j + 1, i] + phi[j - 1, i]) / dy2
                    - rhs[j, i]
                ) / coef
                phi[j, i] = old + omega * (phi_new - old)
                max_diff = max(max_diff, abs(phi[j, i] - old))
        if max_diff < tol:
            break
    return phi


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
    method: str = "direct",
    solver_kw: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """Helmholtz decomposition on a lat-lon grid.

    Args:
        u: Zonal wind [m s⁻¹], shape ``(nlat, nlon)``.
        v: Meridional wind [m s⁻¹], shape ``(nlat, nlon)``.
        lat: Latitude in degrees (ascending), shape ``(nlat,)``.
        lon: Longitude in degrees, shape ``(nlon,)``.
        R_earth: Earth radius [m].
        method: Poisson solver backend — ``'direct'``, ``'fft'``,
            ``'dct'``, or ``'sor'``.
        solver_kw: Extra keyword arguments forwarded to the solver
            (e.g. ``{'max_iter': 3000, 'tol': 1e-5}`` for SOR).

    Returns:
        Dictionary with keys ``u_rot``, ``v_rot``, ``u_div``, ``v_div``,
        ``u_har``, ``v_har``, ``chi``, ``psi``, ``vorticity``,
        ``divergence`` — each ``(nlat, nlon)``.
    """
    if solver_kw is None:
        solver_kw = {}

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

    vort, div = compute_vorticity_divergence(u_work, v_work, dx, dy)

    if method == "fft":
        chi = solve_poisson_fft(div, dx, dy, **solver_kw)
        psi = solve_poisson_fft(vort, dx, dy, **solver_kw)
    elif method == "dct":
        dx_mean = float(np.mean(dx))
        chi = solve_poisson_dct(div, dx_mean, dy, **solver_kw)
        psi = solve_poisson_dct(vort, dx_mean, dy, **solver_kw)
    elif method == "sor":
        chi = solve_poisson_sor(div, dx, dy, **solver_kw)
        psi = solve_poisson_sor(vort, dx, dy, **solver_kw)
    else:  # 'direct'
        chi = solve_poisson_direct(div, dx, dy, **solver_kw)
        psi = solve_poisson_direct(vort, dx, dy, **solver_kw)

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
    method: str = "direct",
) -> dict[str, np.ndarray]:
    """Helmholtz decomposition for each vertical level.

    Applies :func:`helmholtz_decomposition` independently to every
    level along axis 0.

    Args:
        u_3d: Zonal wind [m s⁻¹], shape ``(nlevels, nlat, nlon)``.
        v_3d: Meridional wind [m s⁻¹], shape ``(nlevels, nlat, nlon)``.
        lat: Latitude in degrees (ascending), shape ``(nlat,)``.
        lon: Longitude in degrees, shape ``(nlon,)``.
        R_earth: Earth radius [m].
        method: Poisson solver backend (see :func:`helmholtz_decomposition`).

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
            method=method,
        )
        for k in keys:
            out[k][lev] = res[k]
    return out
