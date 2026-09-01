"""Potential-vorticity floors.

The inversion is elliptic only where the potential vorticity keeps one sign, so
negative values have to be raised before the solve.  Doing that adds potential
vorticity, and adding it changes the circulation the field describes, so the
amount added is taken back from the points that were left alone.  That is done
per level and weighted by area, never by point count: on a global grid the boxes
shrink towards the poles, so counting points would take a share of the correction
from every column regardless of how little atmosphere it holds, and the
redistribution would be biased poleward.

Two modes, matching :class:`~pvinv_sph.config.ClampConfig`:

``parity``
    One pass, taking the correction only from points that remain above the floor
    once it has been taken.  That guard leaves a small conservation slack, which
    is reported rather than hidden.
``clean``
    Iterates the redistribution until the area integral is conserved to rounding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .levels import LevelSet, pv_rhs_scale


@dataclass
class FloorReport:
    """What the floor did, per level."""

    fraction: np.ndarray
    added_pvu: np.ndarray
    conservation_slack_pvu: np.ndarray

    def any_active(self) -> bool:
        return bool(np.any(self.fraction > 0))


def floor_pv(
    q_hat: np.ndarray,
    levels: LevelSet,
    weights: np.ndarray,
    qmin_pvu: float = 0.01,
    mode: str = "parity",
    max_passes: int = 20,
) -> tuple[np.ndarray, FloorReport]:
    """Raise the potential vorticity to a floor, preserving its area integral.

    Args:
        q_hat: Right-hand-side potential vorticity, ``(nint, nlat, nlon)``.
        levels: Level set, used to convert the floor from PVU.
        weights: Gaussian quadrature weights of the grid's latitudes.
        qmin_pvu: Floor in PVU.
        mode: ``"parity"`` or ``"clean"``.
        max_passes: Iteration cap for ``"clean"``.

    Returns:
        The floored field and a :class:`FloorReport`.
    """
    q = np.array(q_hat, dtype=float, copy=True)
    nint, nlat, nlon = q.shape
    scale = pv_rhs_scale(levels.p_hpa[levels.interior])
    area = np.broadcast_to(weights[:, None], (nlat, nlon))

    fraction = np.zeros(nint)
    added = np.zeros(nint)
    slack = np.zeros(nint)

    for k in range(nint):
        floor = qmin_pvu * 1.0e-6 * scale[k]
        below = q[k] < floor
        fraction[k] = float(np.mean(below))
        if not below.any():
            continue
        deficit = float(np.sum(((floor - q[k]) * area)[below]))
        added[k] = deficit / scale[k] * 1.0e6
        q[k][below] = floor

        if mode == "parity":
            # One pass, taking the correction only from points that stay above the
            # floor afterwards; whatever that leaves unbalanced is the slack.
            donors = q[k] > floor
            if donors.any():
                q[k][donors] -= deficit / float(np.sum(area[donors]))
            fixed = q[k] < floor
            slack[k] = (
                float(np.sum(((floor - q[k]) * area)[fixed])) / scale[k] * 1.0e6
                if fixed.any()
                else 0.0
            )
            q[k][fixed] = floor
        else:
            remaining = deficit
            for _ in range(max_passes):
                donors = q[k] > floor
                if not donors.any() or remaining <= 0:
                    break
                q[k][donors] -= remaining / float(np.sum(area[donors]))
                pushed_under = q[k] < floor
                remaining = float(np.sum(((floor - q[k]) * area)[pushed_under]))
                q[k][pushed_under] = floor
            slack[k] = remaining / scale[k] * 1.0e6

    return q, FloorReport(
        fraction=fraction, added_pvu=added, conservation_slack_pvu=slack
    )
