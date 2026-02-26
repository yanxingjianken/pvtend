"""Isentropic interpolation aligned with MetPy's algorithm.

Interpolates 3-D isobaric fields onto isentropic (constant-θ) surfaces
using the method of Ziv & Alpert (1994), as implemented in MetPy v1.7
(`metpy.calc.isentropic_interpolation`).

Algorithm — Pressure solve (Newton-Raphson on Poisson's equation)
-----------------------------------------------------------------
1. Sort pressure levels in **descending** order (surface → top).
2. Compute potential temperature at every grid point:

       θ = T · (P₀ / p)^κ

3. For each target θ*, find the **bounding pressure levels** (above/below).
4. Assume temperature varies **linearly with ln(p)** between the
   bounding levels:

       T(ln p) = a · ln(p) + b

   where  a = (T_above − T_below) / (ln p_above − ln p_below),
          b = T_above − a · ln(p_above).

5. Solve for p* on the θ* surface using **Newton-Raphson** (fixed-point
   iteration on ln(p)):

       θ* = T(ln p) · (P₀ / p)^κ
          = (a · ln p + b) · P₀^κ · exp(−κ · ln p)

       f(ln p)  = θ* − (a · ln p + b) · P₀^κ · exp(−κ · ln p)
       f'(ln p) = P₀^κ · exp(−κ · ln p) · (κ T − a)

       ln p_{n+1} = ln p_n − f / f'

Algorithm — Field interpolation (vectorized linear-in-θ)
--------------------------------------------------------
6. Once p* is known, additional fields φ are **linearly interpolated** vs θ.
   Vectorized implementation (no Python column loops):

   a) Sort each grid column by ascending θ (``np.argsort`` on axis 0).
   b) For each target θ*, find the bracketing index via ``np.searchsorted``
      on the sorted θ columns.
   c) Compute linear weight  w = (θ* − θ_lo) / (θ_hi − θ_lo).
   d) Result = φ_lo + w · (φ_hi − φ_lo).
   e) Mask out-of-range (θ* outside column min/max) → NaN.

Reference
---------
Ziv, B., & Alpert, P. (1994). Isentropic cross-sections across the
Middle East and their relationship to synoptic patterns.
*Beitr. Phys. Atmosph.*, 67, 221–230.

MetPy implementation:
https://github.com/Unidata/MetPy/blob/v1.7.1/src/metpy/calc/thermo.py#L3023-L3190
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import fixed_point

from pvtend.constants import KAPPA, R_DRY, CP_DRY

# Reference surface pressure [hPa] — matches MetPy's P0 = 1000 hPa
_P0_HPA: float = 1000.0


# ── helpers ──────────────────────────────────────────────────────────
def _potential_temperature(pressure_hpa: np.ndarray,
                           temperature_k: np.ndarray) -> np.ndarray:
    """θ = T · (P₀/p)^κ  (all in hPa / K)."""
    return temperature_k * (_P0_HPA / pressure_hpa) ** KAPPA


def _find_bounding_indices(
    theta_sorted: np.ndarray,
    target_levels: np.ndarray,
    vertical_axis: int = 0,
    from_below: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find indices that bracket each target level along the vertical axis.

    Parameters
    ----------
    theta_sorted : (..., nlev, ...) — potential temperature sorted so that
        values increase along *vertical_axis*.
    target_levels : (n_theta,) — 1-D sorted target θ values.
    vertical_axis : int
    from_below : bool — search direction.

    Returns
    -------
    above, below : index arrays (n_theta, *spatial_shape)
    good : bool mask of same shape — True where both bounds are valid.
    """
    nlev = theta_sorted.shape[vertical_axis]
    n_theta = target_levels.size

    # Move vertical to axis-0 for easier indexing
    theta_t = np.moveaxis(theta_sorted, vertical_axis, 0)  # (nlev, ...)
    spatial_shape = theta_t.shape[1:]

    above = np.empty((n_theta, *spatial_shape), dtype=int)
    below = np.empty_like(above)
    good = np.zeros((n_theta, *spatial_shape), dtype=bool)

    for ti, th_target in enumerate(target_levels):
        mask_ge = theta_t >= th_target  # (nlev, ...)
        if from_below:
            idx = np.argmax(mask_ge, axis=0)
        else:
            idx = nlev - 1 - np.argmax(mask_ge[::-1], axis=0)

        any_ge = np.any(mask_ge, axis=0)
        idx_below = idx - 1

        valid = any_ge & (idx_below >= 0)
        above[ti] = idx
        below[ti] = np.clip(idx_below, 0, nlev - 1)
        good[ti] = valid

    return above, below, good


def _isen_iter(iter_log_p, theta_target_nd, ka, a, b, pok):
    """One Newton-Raphson step — identical to MetPy's ``_isen_iter``."""
    exner = pok * np.exp(-ka * iter_log_p)
    t = a * iter_log_p + b
    f = theta_target_nd - t * exner
    fp = exner * (ka * t - a)
    return iter_log_p - (f / fp)


# ── public API ───────────────────────────────────────────────────────
def isentropic_interpolation_pressure(
    theta_levels: np.ndarray,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    *,
    vertical_axis: int = 0,
    max_iters: int = 50,
    eps: float = 1e-6,
    bottom_up_search: bool = True,
) -> np.ndarray:
    """Compute pressure on isentropic surfaces (MetPy algorithm).

    Parameters
    ----------
    theta_levels : (n_theta,)
        Target potential-temperature levels [K].
    pressure_hpa : (nlev,) or same shape as *temperature_k*
        Pressure [hPa].
    temperature_k : (nlev, ...) — temperature [K] on isobaric levels.
    vertical_axis : int
        Axis of *temperature_k* that is the vertical.
    max_iters, eps : Newton-Raphson controls.
    bottom_up_search : bool
        Search direction for bounding indices.

    Returns
    -------
    isen_prs : (n_theta, ...) — pressure [hPa] on each θ surface.
        NaN where the θ level is outside the data range.
    """
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    pressure_hpa = np.asarray(pressure_hpa, dtype=np.float64)
    theta_levels = np.asarray(theta_levels, dtype=np.float64)

    ndim = temperature_k.ndim
    slices = [np.newaxis] * ndim
    slices[vertical_axis] = slice(None)
    slices = tuple(slices)

    if pressure_hpa.ndim == 1:
        pressure_hpa = pressure_hpa[slices]
    pressure_hpa = np.broadcast_to(pressure_hpa, temperature_k.shape).copy()

    # ── Sort descending pressure (surface first) ──
    sort_idx = np.argsort(pressure_hpa, axis=vertical_axis)
    sort_idx = np.flip(sort_idx, axis=vertical_axis)
    sorter = _broadcast_indices(sort_idx, temperature_k.shape, vertical_axis)
    levs = pressure_hpa[sorter]
    tmpk = temperature_k[sorter]

    # ── Sort target θ levels ascending ──
    theta_levels = np.sort(theta_levels)
    n_theta = theta_levels.size

    # ── Potential temperature on sorted isobaric grid ──
    pres_theta = _potential_temperature(levs, tmpk)

    # ── Build broadcast shape for isentropic levels ──
    shape = list(temperature_k.shape)
    shape[vertical_axis] = n_theta
    isentlevs_nd = np.broadcast_to(
        theta_levels.reshape(
            [1] * vertical_axis + [n_theta]
            + [1] * (ndim - vertical_axis - 1)
        ),
        shape,
    )

    ka = KAPPA
    log_p = np.log(levs)
    pok = _P0_HPA ** ka

    # ── Bounding indices ──
    above, below, good = _find_bounding_indices(
        pres_theta, theta_levels, vertical_axis, from_below=bottom_up_search)

    # Move vertical axis to 0 for easier advanced indexing
    log_p_0 = np.moveaxis(log_p, vertical_axis, 0)
    tmpk_0 = np.moveaxis(tmpk, vertical_axis, 0)

    # ── Compute coefficients a, b (vectorized) ──
    log_p_above = _gather(log_p_0, above)
    log_p_below = _gather(log_p_0, below)
    t_above = _gather(tmpk_0, above)
    t_below = _gather(tmpk_0, below)

    dlog = log_p_above - log_p_below
    dlog[dlog == 0] = np.nan
    a = (t_above - t_below) / dlog
    b = t_above - a * log_p_above

    # ── First guess: midpoint in log-p space ──
    isentprs = 0.5 * (log_p_above + log_p_below)

    # Ignore NaNs in a (they propagate from NaN temperature)
    good = good & np.isfinite(a)

    # ── Newton-Raphson via scipy.optimize.fixed_point ──
    log_p_solved = fixed_point(
        _isen_iter,
        isentprs[good],
        args=(isentlevs_nd.reshape(isentprs.shape)[good], ka, a[good], b[good], pok),
        xtol=eps,
        maxiter=max_iters,
    )
    isentprs[good] = np.exp(log_p_solved)
    isentprs[~good] = np.nan
    # Mask points beyond the max pressure in the data
    isentprs[~(good & (isentprs <= np.nanmax(levs) * 1.001))] = np.nan

    return isentprs  # (n_theta, *spatial_shape)


def isentropic_interpolation(
    theta_levels: np.ndarray,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    *fields: np.ndarray,
    vertical_axis: int = 0,
    max_iters: int = 50,
    eps: float = 1e-6,
    bottom_up_search: bool = True,
) -> list[np.ndarray]:
    """Interpolate fields from isobaric to isentropic coordinates.

    This is a pure-NumPy/SciPy re-implementation of MetPy's
    ``isentropic_interpolation`` that does **not** require pint units,
    making it suitable for use inside pvtend's compositing pipeline.

    Parameters
    ----------
    theta_levels : (n_theta,) — target θ [K].
    pressure_hpa : (nlev,) or (nlev, ny, nx) — pressure [hPa].
    temperature_k : (nlev, ny, nx) or (nlev, ...) — temperature [K].
    *fields : additional arrays of same shape to interpolate.
    vertical_axis : int — the pressure/vertical axis (default 0).
    max_iters, eps, bottom_up_search : Newton-Raphson controls.

    Returns
    -------
    [isen_pressure, *interpolated_fields] — each (n_theta, ...).
        Pressure in [hPa]; fields in their original units.
    """
    temperature_k = np.asarray(temperature_k, dtype=np.float64)
    pressure_hpa = np.asarray(pressure_hpa, dtype=np.float64)
    theta_levels = np.asarray(theta_levels, dtype=np.float64).ravel()
    theta_levels = np.sort(theta_levels)

    ndim = temperature_k.ndim
    slices = [np.newaxis] * ndim
    slices[vertical_axis] = slice(None)
    slices = tuple(slices)

    if pressure_hpa.ndim == 1:
        pressure_hpa = pressure_hpa[slices]
    pressure_hpa = np.broadcast_to(pressure_hpa, temperature_k.shape).copy()

    # ── Sort descending pressure (surface first) ──
    sort_idx = np.argsort(pressure_hpa, axis=vertical_axis)
    sort_idx = np.flip(sort_idx, axis=vertical_axis)
    sorter = _broadcast_indices(sort_idx, temperature_k.shape, vertical_axis)
    levs = pressure_hpa[sorter]
    tmpk = temperature_k[sorter]

    # ── Compute θ on sorted grid ──
    pres_theta = _potential_temperature(levs, tmpk)

    # ── Get pressure on isentropic surfaces ──
    isen_prs = isentropic_interpolation_pressure(
        theta_levels, levs, tmpk,
        vertical_axis=vertical_axis,
        max_iters=max_iters,
        eps=eps,
        bottom_up_search=bottom_up_search,
    )

    ret = [isen_prs]

    # ── Interpolate additional fields linearly vs θ ──
    if fields:
        for fld in fields:
            fld = np.asarray(fld, dtype=np.float64)
            fld_sorted = fld[sorter]
            interp_fld = _interp_on_theta(
                theta_levels, pres_theta, fld_sorted, vertical_axis)
            ret.append(interp_fld)

    return ret


# ── Internal utilities ───────────────────────────────────────────────
def _broadcast_indices(
    indices: np.ndarray, shape: tuple, axis: int,
) -> tuple:
    """Build advanced-indexing tuple from argsort output (MetPy-style)."""
    ndim = len(shape)
    idx = []
    for d in range(ndim):
        if d == axis:
            idx.append(indices)
        else:
            s = [1] * ndim
            s[d] = shape[d]
            ar = np.arange(shape[d]).reshape(s)
            idx.append(np.broadcast_to(ar, shape))
    return tuple(idx)


def _gather(arr_axis0: np.ndarray, idx_nd: np.ndarray) -> np.ndarray:
    """Gather from arr_axis0[k, ...] using idx_nd of shape (n_theta, ...).

    Fully vectorized — no Python loops.

    Parameters
    ----------
    arr_axis0 : (nlev, *spatial_shape)
    idx_nd    : (n_theta, *spatial_shape) — level indices to gather

    Returns
    -------
    (n_theta, *spatial_shape)
    """
    spatial_shape = arr_axis0.shape[1:]
    # Build spatial index arrays that broadcast with idx_nd
    spatial_indices = np.indices(spatial_shape)  # (ndim_spatial, *spatial_shape)
    # Expand each to (1, *spatial_shape) then broadcast with idx_nd's leading dim
    expanded = tuple(
        np.broadcast_to(si[np.newaxis], idx_nd.shape)
        for si in spatial_indices
    )
    return arr_axis0[(idx_nd, *expanded)]


def _interp_on_theta(
    theta_levels: np.ndarray,
    pres_theta: np.ndarray,
    field: np.ndarray,
    vertical_axis: int = 0,
) -> np.ndarray:
    """Linearly interpolate *field* onto isentropic levels using θ as coordinate.

    **Vectorized** — no Python column loops.  Sort each column by ascending θ,
    then use fancy indexing to find the bracketing pair and compute the
    linear weight.

    Parameters
    ----------
    theta_levels : (n_theta,) — sorted ascending target θ values.
    pres_theta   : (nlev, ...) — θ on isobaric grid.
    field        : (nlev, ...) — variable to interpolate.
    vertical_axis : int

    Returns
    -------
    (n_theta, ...) interpolated array (NaN where out-of-range).
    """
    # Move vertical to axis-0
    pt = np.moveaxis(pres_theta, vertical_axis, 0)   # (nlev, ...)
    fld = np.moveaxis(field, vertical_axis, 0)        # (nlev, ...)
    nlev = pt.shape[0]
    spatial_shape = pt.shape[1:]
    n_theta = theta_levels.size

    # ── Sort each column by ascending θ ──
    sort_idx = np.argsort(pt, axis=0)                  # (nlev, ...)
    theta_sorted = np.take_along_axis(pt, sort_idx, axis=0)
    field_sorted = np.take_along_axis(fld, sort_idx, axis=0)

    # ── Mask invalid (NaN) values: set to inf / -inf so they don't bracket ──
    nan_mask = ~(np.isfinite(theta_sorted) & np.isfinite(field_sorted))
    theta_clean = theta_sorted.copy()
    theta_clean[nan_mask] = np.inf  # push NaNs to "top"

    # ── Compute min/max valid θ per column ──
    # Replace NaN with inf/−inf so nanmin/nanmax work even if all-NaN
    th_for_min = np.where(nan_mask, np.inf, theta_sorted)
    th_for_max = np.where(nan_mask, -np.inf, theta_sorted)
    col_min = np.min(th_for_min, axis=0)  # (...)
    col_max = np.max(th_for_max, axis=0)  # (...)

    out = np.full((n_theta, *spatial_shape), np.nan, dtype=np.float64)

    # Spatial index arrays for fancy indexing into (nlev, *spatial_shape)
    spatial_indices = np.indices(spatial_shape)  # (ndim_spatial, *spatial_shape)

    for ti, th_target in enumerate(theta_levels):
        # Number of levels below target (sorted ascending)
        n_below = np.sum(theta_clean <= th_target, axis=0)  # (...)
        idx_below = np.clip(n_below - 1, 0, nlev - 2)
        idx_above = idx_below + 1

        # Build fancy-index tuples
        idx_lo = (idx_below, *spatial_indices)
        idx_hi = (idx_above, *spatial_indices)

        th_lo = theta_sorted[idx_lo]
        th_hi = theta_sorted[idx_hi]
        f_lo = field_sorted[idx_lo]
        f_hi = field_sorted[idx_hi]

        # Linear interpolation weight
        dth = th_hi - th_lo
        dth = np.where(dth == 0, np.nan, dth)
        w = (th_target - th_lo) / dth
        result = f_lo + w * (f_hi - f_lo)

        # Mask out-of-range and invalid
        in_range = (th_target >= col_min) & (th_target <= col_max)
        valid = in_range & np.isfinite(th_lo) & np.isfinite(th_hi)
        out[ti] = np.where(valid, result, np.nan)

    return out


# ── Convenience wrappers for composite NPZ events ───────────────────
def interp_event_fields_to_theta(
    theta_3d: np.ndarray,
    pressure_hpa_1d: np.ndarray,
    field_3d: np.ndarray,
    theta_levels: np.ndarray,
) -> np.ndarray:
    """Interpolate one 3-D isobaric field to multiple θ surfaces (vectorized).

    This is the convenience wrapper for the composite-event use case
    (shape: ``(nlev, ny, nx)``).

    Parameters
    ----------
    theta_3d : (nlev, ny, nx) — potential temperature [K]
    pressure_hpa_1d : (nlev,) — isobaric levels [hPa] (unused in θ-path
        but kept for API compatibility with the full MetPy path).
    field_3d : (nlev, ny, nx) — field to interpolate
    theta_levels : (n_theta,) — target θ [K]

    Returns
    -------
    (n_theta, ny, nx) — field on isentropic surfaces.
    """
    theta_levels = np.sort(
        np.asarray(theta_levels, dtype=np.float64).ravel()
    )
    return _interp_on_theta(theta_levels, theta_3d, field_3d, vertical_axis=0)


def interp_event_field_to_single_theta(
    theta_3d: np.ndarray,
    field_3d: np.ndarray,
    theta_target: float,
) -> np.ndarray:
    """Interpolate one 3-D field onto a single θ surface → (ny, nx).

    Vectorized linear-in-θ interpolation.
    Equivalent to one slice of :func:`interp_event_fields_to_theta`.
    """
    result = _interp_on_theta(
        np.array([float(theta_target)]),
        theta_3d,
        field_3d,
        vertical_axis=0,
    )
    return result[0]  # squeeze the θ dimension
