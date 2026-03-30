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
