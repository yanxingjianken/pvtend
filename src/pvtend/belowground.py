"""Below-ground gap filling and the record of where it happened.

Pressure-level fields derived from terrain-following model levels (CESM f09) are
missing where a level lies below ground: ~52 % of the northern hemisphere at
1000 hPa, 4.6 % at 850, 1.1 % at 700. Nothing downstream tolerates a NaN -- one
destroys a whole spherical-harmonic transform and the QG-omega solve spreads it
across the domain -- so the window is filled before anything is computed:
persistence downward from the lowest valid level for every field, the height
continued hydrostatically. ERA5 is extrapolated below ground at the source and
passes through untouched.

The filled cells are also *recorded*. A persisted temperature and wind under
high terrain carry a step at the terrain seam, and the thermal-wind shear across
that step forces the QG-omega equation with line sources along plateau margins
(Greenland, Tibet, the Rockies) -- vertical motion of 4-18 Pa/s at 850 hPa that
is an artefact of the fill, not weather. The mask lets the omega solves drop
their forcing on those cells while the filled state itself stays available for
every other diagnostic.
"""
from __future__ import annotations

import numpy as np

RD = 287.05
G = 9.80665


def below_ground_mask(fields: dict[str, np.ndarray], height_key: str = "z") -> np.ndarray:
    """Cells the fill will invent: where the height (or any field) is missing."""
    mask = ~np.isfinite(np.asarray(fields[height_key]))
    for k, v in fields.items():
        if k != height_key:
            mask |= ~np.isfinite(np.asarray(v))
    return mask


def fill_below_ground_stack(
    fields: dict[str, np.ndarray],
    plev_hpa: np.ndarray,
    height_key: str = "z",
    temp_key: str = "t",
) -> dict[str, np.ndarray]:
    """Gap-fill an arbitrary set of ``(nlev, ny, nx)`` fields.

    Persistence downward from the lowest valid level for everything except the
    height, which is continued hydrostatically with the (persisted, so
    isothermal) layer temperature: ``H_k = H_{k+1} + (R_d T̄ / g) ln(p_{k+1}/p_k)``.

    Args:
        fields: ``{name: (nlev, ny, nx)}``; must contain *height_key* and
            *temp_key*. Modified copies are returned; the inputs are untouched.
        plev_hpa: ``(nlev,)`` pressure levels [hPa], descending in altitude
            (index 0 is the highest pressure).
        height_key: Field continued hydrostatically rather than by persistence.
        temp_key: Temperature [K], needed for the hypsometric thickness.

    Returns:
        A new dict with the same keys, gap-filled; the inputs themselves when
        nothing was missing.
    """
    if all(np.isfinite(v).all() for v in fields.values()):
        return dict(fields)
    for k in (height_key, temp_key):
        if k not in fields:
            raise KeyError(f"fill_below_ground_stack needs {k!r}; got {sorted(fields)}")

    out = {k: np.array(v, dtype=np.float64, copy=True) for k, v in fields.items()}
    p = np.asarray(plev_hpa, dtype=np.float64)
    if p[0] < p[-1]:
        raise ValueError(
            f"plev_hpa must descend in altitude (index 0 = highest pressure); "
            f"got {p[0]} .. {p[-1]}"
        )
    H, T = out[height_key], out[temp_key]
    persist = [v for k, v in out.items() if k != height_key]

    for k in range(len(p) - 2, -1, -1):      # downward: fill k from k+1
        for A in persist:
            m = ~np.isfinite(A[k])
            A[k][m] = A[k + 1][m]
        mz = ~np.isfinite(H[k])
        dz = (RD * 0.5 * (T[k] + T[k + 1]) / G) * np.log(p[k + 1] / p[k])
        H[k][mz] = (H[k + 1] + dz)[mz]
    return out
