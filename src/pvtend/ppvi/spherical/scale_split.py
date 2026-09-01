"""Splitting a piece's source by horizontal scale.

Pass D is linear in its source, so any *additive* split of a source gives pieces
that still sum exactly.  That is what makes a scale split a legitimate piecewise
inversion rather than a decomposition applied after the fact -- and it is why the
split goes through the source rather than through a level list.

Four things here are easy to get wrong, and each is easy to get wrong quietly.

**The filter has to be global.**  Zonal wavenumber only exists on a full circle.
A sixty-degree box has a fundamental of global wavenumber three, so filtering
inside a box would call the event itself planetary.  Here the domain *is* the
circle, so the filter is a selection of spherical-harmonic orders -- exact, with
nothing to crop and nothing to reassemble.

**The object is a connected body in three dimensions, not a stack of outlines.**
A blocking anticyclone is one thing through the depth of the troposphere; taking
a separate two-dimensional component at each level would let the object jump
between features from one level to the next.

**The object search is confined to the event's box.**  On a wavenumber-limited
field the zero line is a single smooth curve, so its negative side is one
component covering a large part of the hemisphere -- every anomaly of that sign,
not this event.  A seed alone is not enough on a global domain; the box is what
bounds it.

**The mask has to be filtered again.**  A hard mask edge puts power back into
every wavenumber, so a merely masked field is not wavenumber-limited whatever it
is called.  Filtering again costs some of the planetary amplitude, which leaks
outside the mask; that is the price of the piece being what its name says.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .sphere import SphereOps

#: Zonal orders kept in the planetary part.  Order zero is excluded: it is the
#: zonal mean, and putting it here would give the planetary piece a
#: meridional-only offset carrying no wave structure at all.
KMIN, KMAX = 1, 4

#: Contour defining the object, as a fraction of the event's own extremum.  The
#: form matters more than the value: anchoring on the event's own amplitude is
#: scale-free, where an absolute threshold cannot serve both ends of a life cycle
#: and a percentile pins the object's area by construction.
OBJ_FRAC = 0.35

#: Half-width of the event box, in degrees, within which the object is sought.
BOX_LAT_HALF, BOX_LON_HALF = 30.0, 60.0


def zonal_filter(
    ops: SphereOps, field: np.ndarray, kmin: int = KMIN, kmax: int = KMAX
) -> np.ndarray:
    """Keep only zonal orders ``kmin`` to ``kmax`` of a field on the solver grid.

    Exact: on the sphere the zonal order is the spherical-harmonic order, so this
    is a selection of coefficients rather than a windowed transform.  Accepts a
    single field or a stack of them.
    """
    if not 0 <= kmin <= kmax:
        raise ValueError(f"need 0 <= kmin <= kmax, got {kmin}, {kmax}")
    single = field.ndim == 2
    stack = field[None] if single else field
    out = np.stack(
        [
            ops.synth(_band(ops.analyze(stack[k]), kmin, kmax))
            for k in range(stack.shape[0])
        ]
    )
    return out[0] if single else out


def _band(spec: np.ndarray, kmin: int, kmax: int) -> np.ndarray:
    keep = np.zeros_like(spec)
    upper = min(kmax, spec.shape[-2] - 1)
    keep[kmin : upper + 1, :] = spec[kmin : upper + 1, :]
    return keep


def great_circle_degrees(ops: SphereOps, lat0: float, lon0: float) -> np.ndarray:
    """Great-circle distance from a centre to every point of the solver grid."""
    lon2d, lat2d = np.meshgrid(ops.grid.lon, ops.grid.lat)
    cos_c = np.sin(np.radians(lat2d)) * np.sin(np.radians(lat0)) + np.cos(
        np.radians(lat2d)
    ) * np.cos(np.radians(lat0)) * np.cos(np.radians(lon2d - lon0))
    return np.degrees(np.arccos(np.clip(cos_c, -1.0, 1.0)))


def event_box(
    ops: SphereOps,
    lat0: float,
    lon0: float,
    lat_half: float = BOX_LAT_HALF,
    lon_half: float = BOX_LON_HALF,
) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the box the object is sought in.

    Longitude is taken the short way round, so a box across the seam is one box
    and not two.  Latitude is clipped at the pole: a box reaching past it simply
    stops, because there is nothing beyond.
    """
    lat, lon = ops.grid.lat, ops.grid.lon
    rows = np.flatnonzero((lat >= lat0 - lat_half) & (lat <= lat0 + lat_half))
    offset = (lon - lon0 + 180.0) % 360.0 - 180.0
    cols = np.flatnonzero(np.abs(offset) <= lon_half)
    if rows.size == 0 or cols.size == 0:
        raise ValueError(f"the box around ({lat0}, {lon0}) is empty on this grid")
    return rows, cols


def component_containing(
    mask: np.ndarray, seed: tuple[int, int, int]
) -> np.ndarray:
    """Three-dimensional six-connected component of ``mask`` holding the seed.

    An empty mask, or a seed outside every component, is raised rather than
    worked around: it says the contour or the level range is wrong, and the
    alternative is silently returning a hemispheric blob.
    """
    labels, count = ndimage.label(
        mask, structure=ndimage.generate_binary_structure(3, 1)
    )
    if count == 0:
        raise ValueError("the mask is empty at this contour")
    label = int(labels[seed])
    if label == 0:
        raise ValueError(
            f"the seed at level {seed[0]}, row {seed[1]}, column {seed[2]} is "
            f"not inside any component at this contour"
        )
    return labels == label


def seed_from_box_minimum(
    filtered: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> tuple[tuple[int, int, int], float]:
    """The most negative point of the filtered field inside the box.

    Nothing here needs tuning per event: the box comes from the tracking, so it
    holds an anticyclone, so its minimum is negative, so the contour taken as a
    fraction of that minimum has the minimum inside it by construction, and the
    flood fill always succeeds.
    """
    sub = filtered[:, rows, :][:, :, cols]
    if np.nanmin(sub) >= 0:
        raise ValueError(
            "the box holds no negative anomaly at the planetary scales, which "
            "contradicts the event having been tracked as an anticyclone"
        )
    flat = int(np.nanargmin(sub))
    k, j, i = np.unravel_index(flat, sub.shape)
    return (int(k), int(rows[j]), int(cols[i])), float(sub[k, j, i])


def split_planetary_eddy(
    ops: SphereOps,
    q_anom: np.ndarray,
    theta_top_anom: np.ndarray,
    upper_positions: list[int],
    lat0: float,
    lon0: float,
    kmin: int = KMIN,
    kmax: int = KMAX,
    obj_frac: float = OBJ_FRAC,
    lat_half: float = BOX_LAT_HALF,
    lon_half: float = BOX_LON_HALF,
) -> dict:
    """Split the upper source into a planetary part and the remainder.

    Args:
        ops: Operators on the solver grid.
        q_anom: Potential-vorticity anomaly on the interior levels,
            ``(nint, nlat, nlon)``.
        theta_top_anom: Top boundary temperature anomaly, ``(nlat, nlon)``.
        upper_positions: Positions within ``q_anom`` of the upper levels, which
            are the ones that take part in the flood fill.
        lat0, lon0: Event centre.
        obj_frac: Contour, as a fraction of the box minimum.

    Returns:
        ``q_p``, ``q_e`` on the interior levels -- both zero outside the upper
        ones -- ``theta_p``, ``theta_e`` for the top boundary, the ``mask``, the
        box minimum ``q_min`` and the ``contour`` actually used.  The two parts
        of each source sum to it on the levels they cover.
    """
    upper = np.asarray(upper_positions, dtype=int)
    rows, cols = event_box(ops, lat0, lon0, lat_half, lon_half)

    # 1. filter globally, on the upper levels that take part
    filtered = zonal_filter(ops, q_anom[upper], kmin, kmax)

    # 2. seed at the box minimum, contour at a fraction of it
    seed, q_min = seed_from_box_minimum(filtered, rows, cols)
    contour = obj_frac * q_min
    level_of_seed = int(np.nonzero(upper == seed[0])[0][0]) if seed[0] in upper else seed[0]

    # 3. the connected body holding the seed, sought inside the box only
    inside = filtered[:, rows, :][:, :, cols] < -abs(contour)
    j = int(np.nonzero(rows == seed[1])[0][0])
    i = int(np.nonzero(cols == seed[2])[0][0])
    box_mask = component_containing(inside, (seed[0], j, i))

    # 4. put it back on the globe; outside the box is not the object
    mask_upper = np.zeros_like(filtered, dtype=bool)
    mask_upper[np.ix_(np.arange(upper.size), rows, cols)] = box_mask

    # 5. filter again after masking, and give the top boundary the mask of the
    #    highest interior level it sits above
    masked = zonal_filter(ops, filtered * mask_upper, kmin, kmax)
    # Both parts live on the upper levels only.  Taking the remainder against the
    # whole anomaly instead would put the lower levels into the eddy part as well
    # as into the lower piece, and the decomposition would count them twice.
    q_p = np.zeros_like(q_anom)
    q_e = np.zeros_like(q_anom)
    q_p[upper] = masked
    q_e[upper] = q_anom[upper] - masked

    top_mask = mask_upper[-1]
    theta_p = zonal_filter(
        ops, zonal_filter(ops, theta_top_anom, kmin, kmax) * top_mask, kmin, kmax
    )
    theta_e = theta_top_anom - theta_p

    mask = np.zeros_like(q_anom, dtype=bool)
    mask[upper] = mask_upper
    return {
        "q_p": q_p,
        "q_e": q_e,
        "theta_p": theta_p,
        "theta_e": theta_e,
        "mask": mask,
        "q_min": q_min,
        "contour": -abs(contour),
        "seed": seed,
        "object_fraction": float(mask_upper.mean()),
        "top_fraction": float(top_mask.mean()),
    }


def scale_pieces(
    ops: SphereOps,
    q_anom: np.ndarray,
    theta_top_anom: np.ndarray,
    upper_positions: list[int],
    lat0: float,
    lon0: float,
    **kwargs,
) -> tuple[dict[str, np.ndarray], dict]:
    """Sources for a decomposition by depth and, within the upper levels, by scale.

    ``lower`` takes the levels below the split.  The upper levels are handed
    twice, once with their planetary part and once with the remainder, so the two
    upper pieces share a level list and differ only in their source -- which is
    why the split travels as a source override rather than as a level selection.

    Returns the sources and the diagnostics of the split.
    """
    upper = np.zeros_like(q_anom)
    upper[upper_positions] = q_anom[upper_positions]
    lower = q_anom - upper
    split = split_planetary_eddy(
        ops, q_anom, theta_top_anom, upper_positions, lat0, lon0, **kwargs
    )
    sources = {"lower": lower, "upper_p": split["q_p"], "upper_e": split["q_e"]}
    return sources, split
