"""Spherical-harmonic primitive operators (windspharm / pyspharm backend).

This module provides the *spectral* (default) backend used by
:mod:`pvtend.helmholtz`, :mod:`pvtend.moist_dry`, and
:mod:`pvtend.derivatives.second_derivs`. All operators close cleanly
across both poles and (when an NH-only field is supplied) across the
equator via parity mirroring, so the legacy 90 °N artefacts that
plagued the finite-difference paths disappear analytically.

Conventions
-----------
* Input grids use **ascending** latitude (south → north) and longitude
  in [0, 360). The wrappers internally flip to the (north → south)
  layout that pyspharm / spharm expect, run the transform, and flip
  back.
* When the input lat range is the Northern Hemisphere only (lat
  monotonically ascending and ``lat[0] >= 0``, ``abs(lat[-1] - 90) <
  1e-6``), the field is automatically parity-mirrored to the full
  globe before transforming. Output is sliced back to NH.
* Parity rules (Phase 2 plan): scalars and zonal wind ``u`` are
  *even* across the equator; meridional wind ``v`` is *odd*. Mixed
  second-order derivative ``dxy`` of a scalar therefore inherits an
  *odd* parity.

Functions
---------
``gradient_sh``         physical (∂/∂x, ∂/∂y) of a scalar.
``laplacian_sh``        ∇² of a scalar.
``invert_laplacian_sh`` solve ∇²χ = rhs (mean-zero gauge).
``vortdiv_sh``          (ζ, δ) from (u, v).
``helmholtz_sh``        full Helmholtz decomposition.
``second_derivs_sh``    (∂²/∂x², ∂²/∂x∂y, ∂²/∂y²) of a scalar.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import R_EARTH

try:
    from spharm import Spharmt
    _SPHARM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPHARM_AVAILABLE = False
    Spharmt = None  # type: ignore[assignment]


__all__ = [
    "gradient_sh",
    "laplacian_sh",
    "invert_laplacian_sh",
    "vortdiv_sh",
    "helmholtz_sh",
    "second_derivs_sh",
    "is_nh_grid",
    "parity_mirror",
    "restrict_to_nh",
    "require_spharm",
]


# ═══════════════════════════════════════════════════════════════
#  Helpers — domain detection and parity mirroring
# ═══════════════════════════════════════════════════════════════


def require_spharm() -> None:
    """Raise a clear error if pyspharm is unavailable."""
    if not _SPHARM_AVAILABLE:
        raise ImportError(
            "pvtend.sh_ops requires `pyspharm` and `windspharm`. "
            "Install via `micromamba install -c conda-forge windspharm pyspharm`."
        )


def is_nh_grid(lat: np.ndarray, tol: float = 1e-3) -> bool:
    """Detect whether *lat* covers only the Northern Hemisphere.

    Args:
        lat: Latitudes in degrees, monotonically ascending or descending.
        tol: Tolerance on the equator endpoint (degrees).

    Returns:
        ``True`` when the latitudes start at or near the equator and end
        at the north pole; ``False`` for a full-globe grid.
    """
    lat = np.asarray(lat, dtype=float).ravel()
    lo, hi = float(min(lat[0], lat[-1])), float(max(lat[0], lat[-1]))
    return (lo > -tol) and (abs(hi - 90.0) < tol)


def parity_mirror(
    field: np.ndarray,
    lat: np.ndarray,
    kind: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror an NH-only field onto the full globe with the correct parity.

    Constructs the full-globe field by reflecting the NH portion across
    the equator. Parity rules (per Phase 2 plan):

    * ``"scalar"`` / ``"u"``: field is **even** across the equator,
      ``F(-φ, λ) =  F(φ, λ)``.
    * ``"v"`` / ``"odd"``: field is **odd**, ``F(-φ, λ) = -F(φ, λ)``;
      the equator row is set to zero.

    Args:
        field: Array with latitude on axis ``-2`` and longitude on
            axis ``-1``. Latitudes must run from the equator (≈0) to
            the north pole (90).
        lat: Latitudes in degrees, ascending, ``lat[0] ≈ 0``,
            ``lat[-1] ≈ 90``.
        kind: Parity tag; one of ``"scalar"``, ``"u"``, ``"v"``,
            ``"odd"``.

    Returns:
        Tuple ``(field_global, lat_global)`` where ``field_global``
        has latitude axis length ``2 * nlat_nh - 1`` and
        ``lat_global`` runs from −lat[-1] to +lat[-1].
    """
    lat = np.asarray(lat, dtype=float).ravel()
    if abs(lat[0]) > 1e-3:
        raise ValueError(
            f"parity_mirror expects lat[0] ≈ 0; got {lat[0]:.4f}"
        )
    nlat_nh = lat.size
    # Mirror: drop equator row when concatenating (kept once, in NH).
    south_lat = -lat[1:][::-1]                      # (nlat_nh-1,)
    lat_global = np.concatenate([south_lat, lat])   # (2*nlat_nh-1,)

    if kind in ("scalar", "u"):
        sign = 1.0
    elif kind in ("v", "odd"):
        sign = -1.0
    else:
        raise ValueError(f"Unknown parity kind: {kind!r}")

    south = sign * field[..., 1:, :][..., ::-1, :]   # mirror across equator
    field_global = np.concatenate([south, field], axis=-2)

    if sign == -1.0:
        # Force exact zero at the equator for odd fields (continuity).
        eq_idx = nlat_nh - 1                         # equator row in global
        field_global[..., eq_idx, :] = 0.0

    return field_global, lat_global


def restrict_to_nh(
    field_global: np.ndarray,
    lat_global: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice a full-globe field to the Northern-Hemisphere portion.

    Args:
        field_global: Array with latitude on axis ``-2`` covering both
            hemispheres, lat ascending.
        lat_global: Latitudes in degrees, ascending.

    Returns:
        Tuple ``(field_nh, lat_nh)`` containing the equator and points
        north of it.
    """
    lat_global = np.asarray(lat_global, dtype=float).ravel()
    eq_idx = int(np.argmin(np.abs(lat_global)))
    return field_global[..., eq_idx:, :], lat_global[eq_idx:]


def _orient_for_spharm(
    field: np.ndarray,
    lat: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Flip the latitude axis so the array is ordered north→south.

    pyspharm expects latitude descending. Most pvtend callers store
    latitude ascending, so we flip on the way in and revert on the way
    out via :func:`_revert_orientation`.

    Returns:
        Tuple ``(field_ns, was_ascending)``.
    """
    lat = np.asarray(lat, dtype=float).ravel()
    ascending = bool(lat[1] > lat[0]) if lat.size > 1 else True
    if ascending:
        return field[..., ::-1, :].copy(), True
    return np.ascontiguousarray(field), False


def _revert_orientation(field_ns: np.ndarray, was_ascending: bool) -> np.ndarray:
    """Undo :func:`_orient_for_spharm`."""
    if was_ascending:
        return field_ns[..., ::-1, :].copy()
    return field_ns


# ═══════════════════════════════════════════════════════════════
#  Spharmt instance cache (legendre tables are O(nlat³))
# ═══════════════════════════════════════════════════════════════


_SPHARMT_CACHE: dict = {}


def _get_spharmt(nlon: int, nlat: int, rsphere: float) -> "Spharmt":
    """Return a cached :class:`spharm.Spharmt` instance for this grid."""
    require_spharm()
    key = (int(nlon), int(nlat), float(rsphere))
    sx = _SPHARMT_CACHE.get(key)
    if sx is None:
        sx = Spharmt(int(nlon), int(nlat), rsphere=float(rsphere),
                     gridtype="regular", legfunc="stored")
        _SPHARMT_CACHE[key] = sx
    return sx


# ═══════════════════════════════════════════════════════════════
#  Internal helpers — eigenvalues for inverse Laplacian
# ═══════════════════════════════════════════════════════════════


def _laplacian_eigenvalues(ntrunc: int, rsphere: float) -> np.ndarray:
    """Eigenvalues -l(l+1)/a² for the spherical Laplacian, packed in
    pyspharm's triangular order."""
    n_coeffs = (ntrunc + 1) * (ntrunc + 2) // 2
    eigs = np.zeros(n_coeffs, dtype=np.float64)
    idx = 0
    for m in range(ntrunc + 1):
        for l in range(m, ntrunc + 1):
            eigs[idx] = -l * (l + 1) / (rsphere * rsphere)
            idx += 1
    return eigs


# ═══════════════════════════════════════════════════════════════
#  Public operators
# ═══════════════════════════════════════════════════════════════


def _to_global(field: np.ndarray, lat: np.ndarray, kind: str):
    """Helper: parity-mirror to global if input is NH-only."""
    if is_nh_grid(lat):
        return (*parity_mirror(field, lat, kind), True)
    return field, np.asarray(lat, dtype=float).ravel(), False


def _from_global(
    field_global: np.ndarray,
    lat_global: np.ndarray,
    was_nh: bool,
):
    """Restore NH portion if we mirrored on input."""
    if was_nh:
        f, _ = restrict_to_nh(field_global, lat_global)
        return f
    return field_global


def gradient_sh(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    R_earth: float = R_EARTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Physical gradient of a scalar via spherical harmonics.

    Computes ``(∂f/∂x, ∂f/∂y)`` in m⁻¹ × ``[field]`` units. Closes
    cleanly at both poles. NH-only inputs are auto-mirrored
    (scalar parity) and the result is sliced back.

    Args:
        field: Scalar field, shape ``(nlat, nlon)``.
        lat: Latitudes in degrees, ascending.
        lon: Longitudes in degrees in [0, 360).
        R_earth: Earth radius in metres.

    Returns:
        Tuple ``(df_dx, df_dy)`` each with the same shape as *field*.
    """
    require_spharm()
    f_g, lat_g, was_nh = _to_global(field, lat, kind="scalar")
    f_ns, asc = _orient_for_spharm(f_g, lat_g)

    nlat, nlon = f_ns.shape
    sx = _get_spharmt(nlon, nlat, R_earth)
    spec = sx.grdtospec(f_ns.astype(np.float32))
    uchi, vchi = sx.getgrad(spec)
    # pyspharm's getgrad returns the components of ∇χ with v pointing
    # north (positive towards north). Our convention is the same in
    # the *south→north* orientation, so flip back consistently.
    df_dx_ns = np.asarray(uchi)
    df_dy_ns = np.asarray(vchi)

    df_dx_g = _revert_orientation(df_dx_ns, asc)
    df_dy_g = _revert_orientation(df_dy_ns, asc)

    return _from_global(df_dx_g, lat_g, was_nh), _from_global(df_dy_g, lat_g, was_nh)


def laplacian_sh(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    R_earth: float = R_EARTH,
) -> np.ndarray:
    """Spherical Laplacian of a scalar field via spherical harmonics.

    Args:
        field: Scalar, shape ``(nlat, nlon)``.
        lat: Latitudes in degrees, ascending.
        lon: Longitudes in degrees.
        R_earth: Earth radius in metres.

    Returns:
        ``∇²f`` with the same shape as *field*.
    """
    require_spharm()
    f_g, lat_g, was_nh = _to_global(field, lat, kind="scalar")
    f_ns, asc = _orient_for_spharm(f_g, lat_g)

    nlat, nlon = f_ns.shape
    sx = _get_spharmt(nlon, nlat, R_earth)
    ntrunc = nlat - 1
    spec = sx.grdtospec(f_ns.astype(np.float32), ntrunc=ntrunc)
    eigs = _laplacian_eigenvalues(ntrunc, R_earth).astype(spec.dtype)
    lap_ns = sx.spectogrd(spec * eigs)
    lap_g = _revert_orientation(np.asarray(lap_ns), asc)
    return _from_global(lap_g, lat_g, was_nh)


def invert_laplacian_sh(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    R_earth: float = R_EARTH,
    parity: str = "scalar",
) -> np.ndarray:
    """Solve ``∇²χ = rhs`` on the sphere with the global-mean zero gauge.

    Args:
        rhs: Right-hand side, shape ``(nlat, nlon)``.
        lat: Latitudes in degrees, ascending.
        lon: Longitudes in degrees.
        R_earth: Earth radius in metres.
        parity: Mirror parity used when *rhs* is NH-only. Defaults to
            ``"scalar"`` (even). Set to ``"v"`` for a divergence-like
            RHS that is odd across the equator.

    Returns:
        ``χ`` of the same shape as *rhs*, with the global area-weighted
        mean removed (Fredholm-compatible gauge fix).
    """
    require_spharm()
    f_g, lat_g, was_nh = _to_global(rhs, lat, kind=parity)
    f_ns, asc = _orient_for_spharm(f_g, lat_g)

    nlat, nlon = f_ns.shape
    sx = _get_spharmt(nlon, nlat, R_earth)
    ntrunc = nlat - 1
    spec = sx.grdtospec(f_ns.astype(np.float32), ntrunc=ntrunc)
    eigs = _laplacian_eigenvalues(ntrunc, R_earth)
    inv = np.zeros_like(eigs)
    inv[1:] = 1.0 / eigs[1:]                       # zero global mean
    chi_ns = sx.spectogrd((spec * inv.astype(spec.dtype)))
    chi_g = _revert_orientation(np.asarray(chi_ns), asc)
    return _from_global(chi_g, lat_g, was_nh)


def filter_low_modes_sh(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    nmin: int = 2,
    R_earth: float = R_EARTH,
    parity: str = "scalar",
) -> np.ndarray:
    """Return *field* with all total-wavenumber modes n < *nmin* zeroed.

    .. deprecated::
        Only used by the archived SH-PPVI method in
        ``pv_inversion/archive/sh/sh_ppvi/``.  Not exported via ``__all__``.
        Will be removed in a future release.

    Uses the identical parity-mirror and north→south orientation pipeline
    as :func:`invert_laplacian_sh`, so results are self-consistent.

    The primary use case is ``nmin=2``: removes the global-mean (n=0) and
    dipole (n=1) modes before a spectral inverse-Laplacian.  On the full
    sphere ∇⁻² amplifies n=1 by R²/[n(n+1)·LL²] ≈ 730× (for LL=1.667e5 m),
    causing Picard divergence in the BALNC ψ-solve when those modes are
    unconstrained by lateral Dirichlet walls (cf. Wu's finite-difference
    solver qinvert21_94.f, which pins rows I=1,I=NY,J=1,J=NX).

    **pyspharm m-major spectral ordering** (triangular truncation T=ntrunc):
      - block m=0: l = 0, 1, …, ntrunc   (ntrunc+1 entries)
      - block m=1: l = 1, 2, …, ntrunc   (ntrunc   entries)
      - block m=k: l = k, k+1, …, ntrunc  (ntrunc+1-k entries)
      Start index of block m: ``m*(ntrunc+1) - m*(m-1)//2``
    For ``nmin=2`` this zeros:
      - (n=0, m=0)  → spectral index 0
      - (n=1, m=0)  → spectral index 1          (already zero for even-parity NH)
      - (n=1, m=1)  → spectral index ntrunc+1   (the primary culprit)

    Args:
        field:   Scalar field, shape ``(nlat, nlon)``.
        lat:     Latitudes in degrees, ascending.
        lon:     Longitudes in degrees.
        nmin:    First total wavenumber to **keep**. Default ``2``.
        R_earth: Earth radius in metres (used for the Spharmt cache key).
        parity:  Mirror parity for NH-only inputs.  Defaults to ``"scalar"``.

    Returns:
        Filtered field of the same shape as *field*.
    """
    require_spharm()
    f_g, lat_g, was_nh = _to_global(field, lat, kind=parity)
    f_ns, asc = _orient_for_spharm(f_g, lat_g)

    nlat, nlon = f_ns.shape
    sx = _get_spharmt(nlon, nlat, R_earth)
    ntrunc = nlat - 1
    spec = sx.grdtospec(f_ns.astype(np.float32), ntrunc=ntrunc)

    # Zero all spectral modes with total wavenumber n (= l in pyspharm) < nmin.
    # m-major packing: block for wavenumber m starts at index m*(ntrunc+1) - m*(m-1)//2.
    # Within that block l runs m, m+1, ..., ntrunc; so l < nmin means the first
    # (nmin - m) entries of the block need to be zeroed.
    for m in range(min(nmin, ntrunc + 1)):
        m_start = m * (ntrunc + 1) - m * (m - 1) // 2
        n_zero = nmin - m
        spec[m_start : m_start + n_zero] = 0.0

    filtered_ns = sx.spectogrd(spec)
    filtered_g = _revert_orientation(np.asarray(filtered_ns), asc)
    return _from_global(filtered_g, lat_g, was_nh)


def invert_helmholtz_sh(
    rhs: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    c: float,
    R_earth: float = R_EARTH,
    parity: str = "scalar",
) -> np.ndarray:
    """Solve ``(∇² + c)χ = rhs`` on the sphere for a scalar constant ``c``.

    .. deprecated::
        Only used by the archived SH-PPVI method in
        ``pv_inversion/archive/sh/sh_ppvi/``.  Not exported via ``__all__``.
        Will be removed in a future release.

    This is the modified-Helmholtz (or screened-Poisson) equation.
    When ``c = 0`` the result is identical to ``invert_laplacian_sh`` (with
    the exception that the n=0 mean mode is also solved here rather than
    pinned to zero, so the caller controls the mean).

    Args:
        rhs: Right-hand side, shape ``(nlat, nlon)``.
        lat: Latitudes in degrees, ascending.
        lon: Longitudes in degrees.
        c: Constant Helmholtz coefficient.  Negative values are allowed
           (as in the PPVI H-equation where ``c ≈ ASI·BB < 0``).
        R_earth: Earth radius in metres.
        parity: Mirror parity for NH-only inputs.  Defaults to ``"scalar"``.

    Returns:
        ``χ`` with the same shape as *rhs*.
    """
    require_spharm()
    f_g, lat_g, was_nh = _to_global(rhs, lat, kind=parity)
    f_ns, asc = _orient_for_spharm(f_g, lat_g)

    nlat, nlon = f_ns.shape
    sx = _get_spharmt(nlon, nlat, R_earth)
    ntrunc = nlat - 1
    spec = sx.grdtospec(f_ns.astype(np.float32), ntrunc=ntrunc)
    eigs = _laplacian_eigenvalues(ntrunc, R_earth)          # -n(n+1)/R²
    helm_eigs = eigs + float(c)                              # -n(n+1)/R² + c
    # Guard against accidental zero eigenvalues
    safe = np.abs(helm_eigs) > 1e-30
    inv = np.where(safe, 1.0 / np.where(safe, helm_eigs, 1.0), 0.0)
    chi_ns = sx.spectogrd((spec * inv.astype(spec.dtype)))
    chi_g = _revert_orientation(np.asarray(chi_ns), asc)
    return _from_global(chi_g, lat_g, was_nh)


def vortdiv_sh(
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    R_earth: float = R_EARTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vorticity ζ and divergence δ from a 2-D wind field.

    NH-only inputs are mirrored with the correct parity (u even, v odd).

    Args:
        u: Zonal wind, shape ``(nlat, nlon)``.
        v: Meridional wind, shape ``(nlat, nlon)``.
        lat: Latitudes in degrees, ascending.
        lon: Longitudes in degrees.
        R_earth: Earth radius in metres.

    Returns:
        Tuple ``(vort, div)`` each shaped ``(nlat, nlon)``.
    """
    require_spharm()
    nh = is_nh_grid(lat)
    if nh:
        u_g, lat_g = parity_mirror(u, lat, kind="u")
        v_g, _ = parity_mirror(v, lat, kind="v")
    else:
        u_g, v_g, lat_g = u, v, np.asarray(lat, dtype=float).ravel()

    u_ns, asc = _orient_for_spharm(u_g, lat_g)
    v_ns, _ = _orient_for_spharm(v_g, lat_g)
    nlat, nlon = u_ns.shape
    sx = _get_spharmt(nlon, nlat, R_earth)
    ntrunc = nlat - 1
    vrtspec, divspec = sx.getvrtdivspec(
        u_ns.astype(np.float32), v_ns.astype(np.float32), ntrunc=ntrunc
    )
    vort_ns = sx.spectogrd(vrtspec)
    div_ns = sx.spectogrd(divspec)
    vort_g = _revert_orientation(np.asarray(vort_ns), asc)
    div_g = _revert_orientation(np.asarray(div_ns), asc)
    if nh:
        vort_nh, _ = restrict_to_nh(vort_g, lat_g)
        div_nh, _ = restrict_to_nh(div_g, lat_g)
        return vort_nh, div_nh
    return vort_g, div_g


def helmholtz_sh(
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    R_earth: float = R_EARTH,
    return_harmonic: bool = True,
    sanity_tol: float = 1e-3,
) -> dict:
    """Spherical-harmonic Helmholtz decomposition.

    On the full sphere a Helmholtz decomposition into rotational and
    divergent parts is *exact*, so the harmonic residual is zero up to
    floating-point error. We still report ``u_har`` / ``v_har`` for API
    parity with the FD path. NH-only inputs are mirrored with parity
    (u even, v odd) so the equator becomes a hard wall and the harmonic
    residual remains negligible inside the NH.

    Args:
        u: Zonal wind, shape ``(nlat, nlon)``.
        v: Meridional wind, shape ``(nlat, nlon)``.
        lat: Latitudes in degrees, ascending.
        lon: Longitudes in degrees in [0, 360).
        R_earth: Earth radius in metres.
        return_harmonic: If True, also return the (numerical-residual)
            harmonic part.
        sanity_tol: Warn if ``‖u_har‖ / ‖u‖ > sanity_tol`` on a global
            input. Disabled for NH-only inputs (the equator wall
            generates an expected boundary residual).

    Returns:
        Dict with the same keys as :func:`pvtend.helmholtz.helmholtz_decomposition`.
    """
    require_spharm()
    import warnings

    nh = is_nh_grid(lat)
    if nh:
        u_g, lat_g = parity_mirror(u, lat, kind="u")
        v_g, _ = parity_mirror(v, lat, kind="v")
    else:
        u_g, v_g, lat_g = u, v, np.asarray(lat, dtype=float).ravel()

    u_ns, asc = _orient_for_spharm(u_g, lat_g)
    v_ns, _ = _orient_for_spharm(v_g, lat_g)
    nlat, nlon = u_ns.shape
    sx = _get_spharmt(nlon, nlat, R_earth)
    ntrunc = nlat - 1

    psi_ns, chi_ns = sx.getpsichi(
        u_ns.astype(np.float32), v_ns.astype(np.float32), ntrunc=ntrunc
    )
    vrtspec, divspec = sx.getvrtdivspec(
        u_ns.astype(np.float32), v_ns.astype(np.float32), ntrunc=ntrunc
    )
    vort_ns = sx.spectogrd(vrtspec)
    div_ns = sx.spectogrd(divspec)

    # Rotational wind: from vorticity only (zero divergence).
    zero_spec = np.zeros_like(vrtspec)
    u_rot_ns, v_rot_ns = sx.getuv(vrtspec, zero_spec)
    # Divergent wind: from divergence only (zero vorticity).
    u_div_ns, v_div_ns = sx.getuv(zero_spec, divspec)

    # Flip back to ascending lat.
    psi_g = _revert_orientation(np.asarray(psi_ns), asc)
    chi_g = _revert_orientation(np.asarray(chi_ns), asc)
    vort_g = _revert_orientation(np.asarray(vort_ns), asc)
    div_g = _revert_orientation(np.asarray(div_ns), asc)
    u_rot_g = _revert_orientation(np.asarray(u_rot_ns), asc)
    v_rot_g = _revert_orientation(np.asarray(v_rot_ns), asc)
    u_div_g = _revert_orientation(np.asarray(u_div_ns), asc)
    v_div_g = _revert_orientation(np.asarray(v_div_ns), asc)

    u_har_g = u_g - u_rot_g - u_div_g
    v_har_g = v_g - v_rot_g - v_div_g

    # Sanity check: on a global field, ‖u_har‖ / ‖u‖ must be tiny when
    # the pole rows (where u, v are coordinate-singular) are excluded.
    if (not nh) and return_harmonic:
        # Identify pole rows in the ascending-lat global frame.
        lat_global = np.asarray(lat_g, dtype=float).ravel()
        non_pole = np.abs(np.abs(lat_global) - 90.0) > 1e-6
        if non_pole.any():
            u_norm = float(np.linalg.norm(u_g[non_pole])) + 1e-30
            v_norm = float(np.linalg.norm(v_g[non_pole])) + 1e-30
            rel = max(
                float(np.linalg.norm(u_har_g[non_pole])) / u_norm,
                float(np.linalg.norm(v_har_g[non_pole])) / v_norm,
            )
            if rel > sanity_tol:
                warnings.warn(
                    f"helmholtz_sh: harmonic residual {rel:.2e} exceeds tol "
                    f"{sanity_tol:.2e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if nh:
        psi = restrict_to_nh(psi_g, lat_g)[0]
        chi = restrict_to_nh(chi_g, lat_g)[0]
        vort = restrict_to_nh(vort_g, lat_g)[0]
        div = restrict_to_nh(div_g, lat_g)[0]
        u_rot = restrict_to_nh(u_rot_g, lat_g)[0]
        v_rot = restrict_to_nh(v_rot_g, lat_g)[0]
        u_div = restrict_to_nh(u_div_g, lat_g)[0]
        v_div = restrict_to_nh(v_div_g, lat_g)[0]
        u_har = restrict_to_nh(u_har_g, lat_g)[0]
        v_har = restrict_to_nh(v_har_g, lat_g)[0]
    else:
        psi, chi, vort, div = psi_g, chi_g, vort_g, div_g
        u_rot, v_rot, u_div, v_div = u_rot_g, v_rot_g, u_div_g, v_div_g
        u_har, v_har = u_har_g, v_har_g

    return dict(
        u_rot=u_rot, v_rot=v_rot,
        u_div=u_div, v_div=v_div,
        u_har=u_har, v_har=v_har,
        chi=chi, psi=psi,
        vorticity=vort, divergence=div,
    )


def second_derivs_sh(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    R_earth: float = R_EARTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second-order horizontal derivatives ``(f_xx, f_xy, f_yy)``.

    Uses two successive applications of :func:`gradient_sh`, so the
    operator is the spectral analogue of ``∇(∇f)`` and inherits the
    pole-closure of the SH transform. Mixed derivatives ``f_xy`` are
    *not* automatically symmetric on a finite truncation, but the
    discrepancy is at the truncation noise level (~1e-12 for ntrunc =
    nlat − 1).

    Args:
        field: Scalar field, shape ``(nlat, nlon)``.
        lat: Latitudes in degrees, ascending.
        lon: Longitudes in degrees.
        R_earth: Earth radius in metres.

    Returns:
        Tuple ``(f_xx, f_xy, f_yy)`` each shaped like *field*.
    """
    require_spharm()
    fx, fy = gradient_sh(field, lat, lon, R_earth=R_earth)
    fxx, fxy = gradient_sh(fx, lat, lon, R_earth=R_earth)
    fyx, fyy = gradient_sh(fy, lat, lon, R_earth=R_earth)
    # Symmetrise the mixed derivative to suppress truncation asymmetry.
    fxy_sym = 0.5 * (fxy + fyx)
    return fxx, fxy_sym, fyy
