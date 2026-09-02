"""Planetary/eddy split of the upper-level PV anomaly.

Splits the upper-level piece of a piecewise inversion into a **planetary** part
(zonal wavenumbers 1–4, confined to the tracked anticyclone) and an **eddy**
remainder, so the two invert separately and sum back to the unsplit upper
result. Pass D is linear in its PV source, so that sum is exact rather than
approximate — which is what makes this a legitimate piecewise inversion and not
a decomposition after the fact.

Three things here are easy to get wrong, and each was got wrong first:

**The filter has to be global.** Zonal wavenumber only exists on the full
circle. A ±60° event box has a fundamental of global k=3, so filtering inside
the box would call the event itself "planetary". :func:`zonal_filter` therefore
requires its last axis to span 360°, and the object search is confined to the
box *after* filtering.

**"Largest component" is not the object.** On a k≤4 field the zero line is a
smooth planetary curve, so its negative side is a few very large patches which
link through the weakly negative subtropics — with periodic longitude they
merge into one blob covering ~38 % of the hemisphere. That is every negative PV
anomaly in the NH, not this block. The component is selected by a seed instead.

**The seed is not the tracked centroid.** The Z500 blob centroid and the
upper-level PV minimum are different points and the gap grows with blob size;
on one verified case the centroid sat at ``q'_k4 = +150``, the 92nd percentile
of the field. :func:`seed_from_centroid` takes the local minimum within a
radius of the centroid instead.

Ported from the single-event pipeline in
``pv_inversion_isentropic/wu_cesm/scale_split_sanity/`` so the NPZ path and the
method pipeline share one implementation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

#: Zonal wavenumbers kept in the planetary part. k=0 is dropped: it is the
#: zonal mean, and putting it in the planetary piece would give that piece a
#: meridional-only offset with no wave structure.
KMIN, KMAX = 1, 4

#: Contour level defining the object, as a fraction of the event's **own** box
#: minimum. 0.35 was calibrated against the tracker's `area` column over 120
#: CESM events (60 blocking + 60 prp, member 91): it puts the object's
#: footprint at the same size as the Z500 blob at peak.
#:
#: The form matters more than the number. An absolute threshold cannot serve
#: both ends of a life cycle — the tracked blob is 4.4x larger at peak than at
#: onset — and a *band percentile* cannot either, for a definitional reason:
#: taking the most negative p % of the band pins the below-threshold AREA to
#: p % of the band, so the object cannot grow with the event and the ratio just
#: tracks 1/blob_area (measured: onset 3-4x peak at every percentile). Anchoring
#: on the event's own amplitude is scale-free, and is how anomaly contours are
#: conventionally drawn.
#:
#: Neither form makes the object/blob ratio constant across the life cycle
#: (onset stays ~2.5-3x peak for blocking, ~5-6x for prp). That is a physical
#: result, not a tuning failure: the k<=4 PV anomaly is a planetary-scale
#: feature and is already broad at onset, while the Z500 blob grows into it.
OBJ_FRAC = 0.35

#: Retained so a caller can still pass an absolute contour; ``None`` means use
#: :data:`OBJ_FRAC` against the box minimum.
THRESH = None

#: Radius [deg] around the tracked centroid for :func:`seed_from_centroid`,
#: which the single-event method pipeline still uses. The NPZ path seeds from
#: the box minimum instead and needs no radius.
SEED_RADIUS_DEG = 20.0


#: The four-piece decomposition, as 1-based level indices. ``upper_p`` and
#: ``upper_e`` cover the same levels; what separates them is the PV anomaly each
#: is given (see :func:`split_at_box_minimum`), which is why the split has to be
#: handed to the inversion as a PV source and not as a level list.
PIECES_SCALE: dict[str, list[int]] = {
    "surface": [1],
    "lower": [2, 3, 4],
    "upper_p": [5, 6, 7, 8, 9],
    "upper_e": [5, 6, 7, 8, 9],
}

#: 0-based indices into the 9-level Wu grid: the upper **interior** levels
#: (400/300/250/200 hPa) and the top boundary (100 hPa). Wu's ``Q`` is
#: identically zero at the boundary levels, so only the interior takes part in
#: the flood fill; the top boundary inherits the mask of the highest interior
#: level and carries the θ anomaly instead.
UPPER_INTERIOR_IDX = [4, 5, 6, 7]
TOP_IDX = 8


def zonal_filter(field: np.ndarray, kmin: int = KMIN, kmax: int = KMAX) -> np.ndarray:
    """Keep zonal wavenumbers ``kmin..kmax``. Last axis must span 360°."""
    F = np.fft.rfft(np.asarray(field, dtype=float), axis=-1)
    G = np.zeros_like(F)
    G[..., kmin:kmax + 1] = F[..., kmin:kmax + 1]
    return np.fft.irfft(G, n=np.shape(field)[-1], axis=-1)


def component_containing(mask: np.ndarray, seed_lev: int, seed_lat: int,
                         seed_lon: int) -> np.ndarray:
    """3-D 6-connected component of *mask* containing the seed point.

    Raises:
        ValueError: If the mask is empty, or the seed is not inside any
            component. That failure is useful — it says the threshold or the
            level range is wrong, rather than silently returning a hemispheric
            blob.
    """
    lab, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    if n == 0:
        raise ValueError("mask is empty at this threshold")
    lid = int(lab[seed_lev, seed_lat, seed_lon])
    if lid == 0:
        raise ValueError(
            f"seed (lev {seed_lev}, lat {seed_lat}, lon {seed_lon}) is not "
            f"inside any negative component at this threshold")
    return lab == lid


def seed_from_centroid(q_k4_level: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                       clat: float, clon: float,
                       radius: float = SEED_RADIUS_DEG) -> tuple[int, int]:
    """Local minimum of the filtered anomaly within *radius* of the centroid.

    Returns global ``(j, i)`` indices. Longitude distance is taken the short way
    round so a centroid near the date line still searches both sides.
    """
    jj = np.nonzero(np.abs(lat - clat) <= radius)[0]
    ii = np.nonzero(np.abs(((lon - clon + 180.0) % 360.0) - 180.0) <= radius)[0]
    if jj.size == 0 or ii.size == 0:
        raise ValueError(f"no grid points within {radius}° of ({clat}, {clon})")
    sub = q_k4_level[np.ix_(jj, ii)]
    a, b = np.unravel_index(int(np.nanargmin(sub)), sub.shape)
    return int(jj[a]), int(ii[b])


def seed_from_box_min(q_k4: np.ndarray, upper_idx, box_lat, box_lon
                      ) -> tuple[int, int, int]:
    """Global ``(lev, lat, lon)`` of the most negative filtered anomaly in the box.

    This is the seed the NPZ path uses. The box was drawn around a tracked
    anticyclone, so its minimum is negative; a negative point is by construction
    inside ``q'_k4 < 0``, so the flood fill that follows cannot fail. That is the
    whole reason to seed here rather than from the centroid — the centroid is a
    Z500 feature and need not sit on the upper-level PV minimum at all (measured
    on one case at ``q'_k4 = +150``, the 92nd percentile).

    Args:
        q_k4: ``(NL, NY, NX_global)`` k≤4-filtered PV anomaly.
        upper_idx: 0-based upper **interior** level indices to search.
        box_lat, box_lon: index arrays of the event box.

    Returns:
        Global indices ``(lev, lat, lon)``.

    Raises:
        ValueError: If the box holds no negative point — meaning the tracked
            anticyclone has no negative upper-level PV anomaly at all, which is
            a contradiction worth surfacing rather than papering over.
    """
    upper_idx = np.asarray(upper_idx, dtype=int)
    box_lat = np.asarray(box_lat, dtype=int)
    box_lon = np.asarray(box_lon, dtype=int)
    sub = q_k4[np.ix_(upper_idx, box_lat, box_lon)]
    k, j, i = np.unravel_index(int(np.nanargmin(sub)), sub.shape)
    if not (sub[k, j, i] < 0):
        raise ValueError(
            f"no negative k<=4 PV anomaly anywhere in the event box "
            f"(box minimum is {sub[k, j, i]:+.4g})")
    return int(upper_idx[k]), int(box_lat[j]), int(box_lon[i])


def nearest_index(values: np.ndarray, target: float, periodic: bool = False) -> int:
    """Index of the grid point nearest *target*; wraps if *periodic*."""
    v = np.asarray(values, dtype=float)
    d = np.abs((v - target + 180.0) % 360.0 - 180.0) if periodic else np.abs(v - target)
    return int(np.argmin(d))


def split_planetary_eddy(
    q_anom: np.ndarray,
    th_anom_top: np.ndarray,
    upper_idx,
    top_idx: int,
    box_lat,
    box_lon,
    seed_lev: int,
    seed_lat: int,
    seed_lon: int,
    kmin: int = KMIN,
    kmax: int = KMAX,
    thresh: float = 0.0,
) -> dict:
    """Split the upper PV anomaly and its top boundary θ into planetary/eddy.

    The low-level entry point, taking an **absolute** contour. Bulk callers
    should use :func:`split_at_box_minimum`, which derives the contour from the
    event's own amplitude. The default of 0.0 is "simply negative".

    Args:
        q_anom: ``(NL, NY, NX_global)`` Wu PV anomaly; last axis spans 360°.
        th_anom_top: ``(NY, NX_global)`` top boundary-θ anomaly.
        upper_idx: 0-based indices of the upper **interior** levels. Wu's ``Q``
            is identically zero at the two boundary levels, so only the interior
            takes part in the flood fill.
        top_idx: 0-based index of the top boundary level.
        box_lat, box_lon: index arrays of the event box. The filter is global;
            the object search is confined to the box.
        seed_lev, seed_lat, seed_lon: global indices of the seed. The component
            containing this point is the object.

    Returns:
        ``dict`` with ``q_p``/``q_e`` ``(NL, NY, NX)``, ``th_p``/``th_e``
        ``(NY, NX)``, ``mask`` ``(NL, NY, NX)``, and ``top_frac``.
    """
    upper_idx = np.asarray(upper_idx, dtype=int)
    box_lat = np.asarray(box_lat, dtype=int)
    box_lon = np.asarray(box_lon, dtype=int)

    # 1. filter GLOBALLY — zonal wavenumber only exists on the full circle
    q_k_int = zonal_filter(q_anom[upper_idx], kmin, kmax)

    # 2. crop to the event box and take the component holding the seed
    sub = q_k_int[:, box_lat, :][:, :, box_lon] < -abs(thresh)
    j = int(np.nonzero(box_lat == seed_lat)[0][0])
    i = int(np.nonzero(box_lon == seed_lon)[0][0])
    k = int(np.nonzero(upper_idx == seed_lev)[0][0])
    m_box = component_containing(sub, k, j, i)

    # 3. embed back into the global array; outside the box is not the object
    mask_int = np.zeros_like(q_k_int, dtype=bool)
    mask_int[np.ix_(np.arange(len(upper_idx)), box_lat, box_lon)] = m_box
    mask = np.zeros_like(q_anom, dtype=bool)
    mask[upper_idx] = mask_int
    # 4. the top boundary inherits the mask of the highest interior level
    mask[top_idx] = mask_int[-1]

    q_p = np.zeros_like(q_anom)
    q_e = np.zeros_like(q_anom)
    # Re-filter after masking: q'_k4 · M is NOT itself k≤4 (a sharp mask edge
    # puts power back into every wavenumber — measured 48 % in k=0 alone). The
    # cost is ~24 % of the planetary amplitude leaking outside the mask, which
    # is the price of the piece being genuinely wavenumber-limited.
    q_masked = q_k_int[:len(upper_idx)] * mask_int[:len(upper_idx)]
    q_p_upper = zonal_filter(q_masked, kmin, kmax)
    for n, kk in enumerate(upper_idx):
        q_p[kk] = q_p_upper[n]
        q_e[kk] = q_anom[kk] - q_p[kk]

    th_k = zonal_filter(th_anom_top, kmin, kmax)
    m_top = mask[top_idx]
    th_p = zonal_filter(th_k * m_top, kmin, kmax)
    th_e = th_anom_top - th_p

    return {"q_p": q_p, "q_e": q_e, "th_p": th_p, "th_e": th_e,
            "mask": mask, "top_frac": float(m_top.mean()),
            "seed": (int(seed_lev), int(seed_lat), int(seed_lon))}


def seed_near_centre(q_k4: np.ndarray, upper_idx, box_lat, box_lon,
                     centre_lat: int, centre_lon: int,
                     halo_lat: int, halo_lon: int) -> tuple[int, int, int]:
    """Global ``(lev, lat, lon)`` of the most negative anomaly NEAR the centre.

    :func:`seed_from_box_min` searches the whole event box, which is right only
    when the tracked feature *is* the box minimum. Measured on 30+30 m091
    events: for blocking it is (box min / local min = 1.0), but for a
    propagating high it is not (1.5), and the seed then runs away to a deeper
    unrelated system a median **4193 km** from the tracked centre — inverting
    the wrong feature while every residual diagnostic still balances.

    Restricting the search to a halo around the tracker's own centre brings
    that to 1375 km, and the distance from the centre to the nearest point of
    the resulting object from **1581 km to 117 km** — about one grid cell.
    Blocking is left alone (45 km before, 44 km after).

    Works in box-relative positions, so it inherits the circular ordering of
    *box_lon* and needs no longitude arithmetic of its own.

    Args:
        q_k4: ``(NL, NY, NX_global)`` already-filtered anomaly.
        upper_idx: 0-based upper interior level indices to search.
        box_lat, box_lon: index arrays of the event box, ordered as the flood
            fill will see them.
        centre_lat, centre_lon: **global** row/column of the tracked centre.
        halo_lat, halo_lon: half-width of the seed window, in grid cells.

    Returns:
        Global indices ``(lev, lat, lon)``.

    Raises:
        ValueError: If the centre is outside the box, or no negative anomaly
            lies within the halo — the tracked feature has no upper-level
            negative PV signature to invert, which is worth surfacing.
    """
    upper_idx = np.asarray(upper_idx, dtype=int)
    box_lat = np.asarray(box_lat, dtype=int)
    box_lon = np.asarray(box_lon, dtype=int)
    jp = np.nonzero(box_lat == int(centre_lat))[0]
    ip = np.nonzero(box_lon == int(centre_lon))[0]
    if jp.size == 0 or ip.size == 0:
        raise ValueError(
            f"tracked centre (row {centre_lat}, col {centre_lon}) is not inside "
            f"the event box — cannot seed near it")
    j0 = slice(max(0, int(jp[0]) - halo_lat), int(jp[0]) + halo_lat + 1)
    i0 = slice(max(0, int(ip[0]) - halo_lon), int(ip[0]) + halo_lon + 1)
    rows, cols = box_lat[j0], box_lon[i0]
    sub = q_k4[np.ix_(upper_idx, rows, cols)]
    k, a, b = np.unravel_index(int(np.nanargmin(sub)), sub.shape)
    if not (sub[k, a, b] < 0):
        raise ValueError(
            f"no negative k<=4 PV anomaly within the seed halo of the tracked "
            f"centre (local minimum is {sub[k, a, b]:+.4g})")
    return int(upper_idx[k]), int(rows[a]), int(cols[b])


def split_near_centre(
    q_anom: np.ndarray,
    th_anom_top: np.ndarray,
    upper_idx,
    top_idx: int,
    box_lat,
    box_lon,
    centre_lat: int,
    centre_lon: int,
    halo_lat: int,
    halo_lon: int,
    kmin: int = KMIN,
    kmax: int = KMAX,
    frac: float = OBJ_FRAC,
) -> dict:
    """:func:`split_planetary_eddy` seeded and scaled at the TRACKED centre.

    The order matters and is not interchangeable:

    1. filter to ``kmin..kmax`` on the **whole longitude circle** — a zonal
       wavenumber does not exist on a cropped strip;
    2. take the seed as the most negative filtered point within a halo of the
       tracker's centre (:func:`seed_near_centre`);
    3. set the contour at *frac* x **that local minimum**, not the box minimum,
       so the depth scale comes from the event rather than from whatever deeper
       system shares its box;
    4. flood fill across the **whole box**, so the object may still grow to its
       natural extent — the halo constrains only where the search starts.

    :func:`split_at_box_minimum` is the older policy: steps 2 and 3 both keyed
    off the box minimum. It agrees with this one whenever the tracked feature is
    the deepest thing in its box (blocking), and picks a different feature
    entirely when it is not (propagating highs).

    Returns:
        As :func:`split_planetary_eddy`, plus ``q_min`` (the *local* minimum),
        ``thresh_used`` and ``seed_source='near_centre'``.
    """
    q_k4 = zonal_filter(q_anom, kmin, kmax)
    lev, j, i = seed_near_centre(q_k4, upper_idx, box_lat, box_lon,
                                 centre_lat, centre_lon, halo_lat, halo_lon)
    q_min = float(q_k4[lev, j, i])
    out = split_planetary_eddy(q_anom, th_anom_top, upper_idx, top_idx,
                               box_lat, box_lon, lev, j, i,
                               kmin=kmin, kmax=kmax, thresh=frac * q_min)
    out.update(seed_source="near_centre", q_min=q_min,
               thresh_used=-abs(frac * q_min))
    return out


def split_at_box_minimum(
    q_anom: np.ndarray,
    th_anom_top: np.ndarray,
    upper_idx,
    top_idx: int,
    box_lat,
    box_lon,
    kmin: int = KMIN,
    kmax: int = KMAX,
    thresh: float | None = THRESH,
    frac: float = OBJ_FRAC,
) -> dict:
    """:func:`split_planetary_eddy` seeded from the box minimum.

    The entry point for bulk NPZ generation, where no one is looking at each
    event. Nothing here needs tuning per event: the box comes from the tracker,
    so it contains an anticyclone; its minimum is therefore negative; the
    contour is a fraction of that minimum, so the minimum is inside its own
    contour by construction; so the flood fill always succeeds and always
    returns the object the tracker found.

    Args:
        thresh: Absolute contour level. ``None`` (the default) derives it from
            the box minimum via *frac*.
        frac: Fraction of the box minimum to contour at, when *thresh* is None.

    Returns:
        As :func:`split_planetary_eddy`, plus ``q_min`` and ``thresh_used``.

    Raises:
        ValueError: If the box holds no negative k≤4 anomaly at all — which
            contradicts the tracking, and is surfaced rather than worked around.
    """
    q_k4 = zonal_filter(q_anom, kmin, kmax)
    lev, j, i = seed_from_box_min(q_k4, upper_idx, box_lat, box_lon)
    q_min = float(q_k4[lev, j, i])
    level = float(thresh) if thresh is not None else frac * q_min
    out = split_planetary_eddy(q_anom, th_anom_top, upper_idx, top_idx,
                               box_lat, box_lon, lev, j, i,
                               kmin=kmin, kmax=kmax, thresh=level)
    out.update(seed_source="box_min", q_min=q_min, thresh_used=-abs(level))
    return out
