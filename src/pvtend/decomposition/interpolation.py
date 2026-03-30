"""Temporal interpolation utilities for PV basis construction.

Provides linear interpolation between consecutive hourly snapshots so that
the orthogonal basis is built from fields representative of a sub-hourly
instant (e.g. 15 min before dh with alpha=0.75).
"""

from __future__ import annotations

import numpy as np


def lerp_fields(
    fields_prev: dict[str, np.ndarray],
    fields_curr: dict[str, np.ndarray],
    alpha: float = 0.75,
    keys: tuple[str, ...] = ("pv_anom", "pv_dx", "pv_dy"),
) -> dict[str, np.ndarray]:
    """Linearly interpolate selected 2-D fields between two time-steps.

    Args:
        fields_prev: NPZ dict at dh-1.
        fields_curr: NPZ dict at dh.
        alpha: Interpolation weight for *fields_curr*.
            ``alpha = 0.75`` yields the field 15 min before dh.
        keys: Field names to interpolate.

    Returns:
        Dict mapping each key to ``(1 - alpha) * prev + alpha * curr``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    out: dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = (1.0 - alpha) * np.asarray(fields_prev[k], dtype=np.float64) + \
                 alpha * np.asarray(fields_curr[k], dtype=np.float64)
    return out


def compute_pv_center(
    pv_anom: np.ndarray,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
) -> tuple[float, float]:
    """Compute the abs(pv_anom)-weighted centroid over the pv_anom < 0 region.

    Args:
        pv_anom: 2-D PV anomaly field (ny, nx).
        x_rel: 1-D or 2-D relative longitude array (degrees).
        y_rel: 1-D or 2-D relative latitude array (degrees).

    Returns:
        (cx, cy) centroid in the same units as *x_rel* / *y_rel*.
        Returns (0.0, 0.0) if no negative PV anomaly exists.
    """
    pv = np.asarray(pv_anom, dtype=np.float64)
    if x_rel.ndim == 1 and y_rel.ndim == 1:
        X, Y = np.meshgrid(x_rel, y_rel)
    else:
        X, Y = np.asarray(x_rel, dtype=np.float64), np.asarray(y_rel, dtype=np.float64)

    mask = pv < 0
    if not mask.any():
        return 0.0, 0.0

    w = np.abs(pv[mask])
    w_sum = w.sum()
    cx = float(np.sum(w * X[mask]) / w_sum)
    cy = float(np.sum(w * Y[mask]) / w_sum)
    return cx, cy
