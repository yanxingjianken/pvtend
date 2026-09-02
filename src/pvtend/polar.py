"""Continuation of an event patch across the North Pole.

An event patch is a box of ±30° latitude by ±60° longitude cut from the
archive around the tracked centre, column by column along fixed meridians.
North of the pole a meridian continues down the far side of the Earth at
longitude λ+180°, so the row of the box at nominal latitude 90+d is a real
place -- geographic latitude 90−d on the antimeridian -- with all of its
fields. This module fills those rows from the archive instead of leaving them
missing, so that a composite of high-latitude events averages every event on
every row.

The rule is exact. Following the meridian through the pole turns the local
frame by 180°: the box's eastward direction becomes geographic west and its
northward direction geographic south. A 180° turn is a proper rotation, so
scalars are unchanged, every horizontal vector component and every first
horizontal derivative changes sign, and anything built from two such factors
(the advection cross terms, the second derivatives, the vorticity) is unchanged.
Nothing is interpolated: each continued row is a copy of a grid row, and the
rows the box already had are untouched.

The continuation needs a grid row on the pole and an even number of longitudes
(so that λ+180° is a grid column); both archives this pipeline reads have them.
The rows' nominal latitudes are stored as ``lat_vec`` (monotone, above 90 past
the pole), their geographic latitudes as ``lat_geo_vec``, and ``far_side_rows``
marks the rows whose geographic longitude is ``lon_vec + 180``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_PRODUCT = re.compile(r"^[uvw](_[a-z0-9]+)*_pv_")
_SECOND_DERIVATIVE = ("_dx_dx", "_dy_dy", "_dx_dy")


@dataclass
class PolarContinuation:
    """Which patch rows lie past the pole and which grid rows supply them.

    Attributes:
        slots: Patch row indices (ascending) of the rows past the pole.
        src_rows: Grid row index supplying each slot: the row at geographic
            latitude 180 minus the slot's nominal latitude.
        nominal_lat: The slots' nominal latitudes, 90 + m·dlat.
        col_shift: Columns to add to a patch column to reach λ+180°.
    """

    slots: np.ndarray
    src_rows: np.ndarray
    nominal_lat: np.ndarray
    col_shift: int

    def far_columns(self, cols: np.ndarray, nlon: int) -> np.ndarray:
        return (np.asarray(cols) + self.col_shift) % nlon


def plan_continuation(
    lat_all: np.ndarray,
    nlon: int,
    lat_pad: int,
    eff_north: int,
    eff_south: int,
    dlat: float,
) -> PolarContinuation | None:
    """Plan the rows past the pole of one patch, or ``None`` if there are none.

    Args:
        lat_all: The archive's latitude axis, in its own order.
        nlon: Number of longitudes on the archive grid.
        lat_pad: Half-height of the patch in rows.
        eff_north, eff_south: Rows the archive supplies north and south of the
            centre row (``lat_pad`` when the box fits).
        dlat: Latitude spacing in degrees.
    """
    n_missing = int(lat_pad - eff_north)
    if n_missing <= 0:
        return None
    lat_all = np.asarray(lat_all, dtype=float)
    tol = 0.25 * float(dlat)
    if abs(float(lat_all.max()) - 90.0) > tol or nlon % 2:
        return None
    y0 = lat_pad - eff_south
    y_eff = eff_north + eff_south + 1
    slots, src, nominal = [], [], []
    for m in range(1, n_missing + 1):
        target = 90.0 - m * dlat
        r = int(np.argmin(np.abs(lat_all - target)))
        if abs(lat_all[r] - target) > tol:
            break
        slots.append(y0 + y_eff - 1 + m)
        src.append(r)
        nominal.append(90.0 + m * dlat)
    if not slots:
        return None
    return PolarContinuation(
        slots=np.array(slots, dtype=int),
        src_rows=np.array(src, dtype=int),
        nominal_lat=np.array(nominal, dtype=float),
        col_shift=nlon // 2,
    )


def negates_across_pole(key: str) -> bool:
    """Whether a stored field changes sign on the rows past the pole.

    Rank-1 horizontal quantities do: the wind components (``u``, ``v`` and every
    ``u_*``/``v_*`` component, rotational, divergent, harmonic, the inversion
    pieces) and the first horizontal derivatives (``*_dx``, ``*_dy``). Products
    of two of them (the ``u_*_pv_*_dx`` cross terms), second derivatives and
    scalars do not. Vertical (``_dp``) and time (``_dt``) derivatives are
    scalars here.
    """
    k = key[:-3] if key.endswith("_3d") else key
    if _PRODUCT.match(k):
        return False
    if k in ("u", "v") or k.startswith(("u_", "v_")):
        return True
    if k.endswith(("_dx", "_dy")) and not k.endswith(_SECOND_DERIVATIVE):
        return True
    return False


def geographic_latitude(lat_vec: np.ndarray) -> np.ndarray:
    """Geographic latitude of each patch row: 180 minus the nominal one past the pole."""
    lat_vec = np.asarray(lat_vec, dtype=float)
    return np.where(lat_vec > 90.0, 180.0 - lat_vec, lat_vec)


def far_side_rows(lat_vec: np.ndarray) -> np.ndarray:
    """Rows whose geographic longitude is the patch longitude plus 180°."""
    lat_vec = np.asarray(lat_vec, dtype=float)
    return np.isfinite(lat_vec) & (lat_vec > 90.0 + 1e-9)
