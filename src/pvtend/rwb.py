"""Rossby Wave Breaking (RWB) identification.

Detects overturning Z/PV contours and classifies them as Anticyclonic
Wave Breaking (AWB) or Cyclonic Wave Breaking (CWB).

**Two classification methods:**

* ``method="bay"`` *(default)* -- MATLAB-consistent path-order algorithm.
  At each meridian through a bay, intersect the contour in path-trace
  order; if the first-visited max-latitude comes before the first-visited
  min-latitude -> CWB, else AWB.  Aggregated by median sign across
  meridians.  Ambiguous bays remain "UNK" (no tilt fallback).

* ``method="tilt"`` -- Centerline-tilt classifier.
  Fit a line to the midpoints of the upper/lower envelope of each bay.
  Slope < -0.15 -> AWB; slope > +0.15 -> CWB; |slope| <= 0.15 -> UNK (neutral).
  Threshold consistent with ESSOAr 10.22541/essoar.175821720.04705853.
  Activate only when explicitly requested (e.g. ``detect_rwb_events(..., method="tilt")``).

**Circumpolar-first approach** (matching Talia's MATLAB bay method):
  1. On the FULL Northern-Hemisphere field, extract all circumpolar
     contours that span the entire longitude range.
  2. Crop each circumpolar contour to the event-centred patch.
  3. Within the patch, detect overturning via meridian crossing counts.
  4. Classify via the chosen *method*.

If the full NH field is not available, falls back to the legacy
``sampled_longest_contours`` on the local patch.

References:
    Gabriel and Peters (2008) J. Atmos. Sci. -- Bay method.
    Peters D, Waugh D W (1996) J. Atmos. Sci. 53, 3013-3031.
    Thorncroft C D, Hoskins B J, McIntyre M E (1993) Q.J.R.M.S. 119, 17-55.
"""

from __future__ import annotations
from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Sequence
import numpy as np

try:
    from skimage.measure import find_contours
except ImportError:  # pragma: no cover
    find_contours = None  # type: ignore[assignment]


# =====================================================================
# Configuration
# =====================================================================
@dataclass
class RWBConfig:
    """Configuration for RWB detection and classification.

    Attributes:
        level_mode: Vertical level mode - 'wavg' or integer hPa.
        try_levels: Number of contour levels to probe on full NH field.
        min_vertices: Minimum vertices for a valid contour.
        n_meridians: Number of meridian probes for overturn detection.
        min_cross: Minimum crossings to flag overturning.
        n_samp: Number of sample points for polygon construction.
        min_points: Minimum valid points for polygon.
        area_min_deg2: Minimum bay area (deg²) filter.
        x_min_awb: AWB centroid x must be >= this.
        x_max_cwb: CWB centroid x must be <= this.
        circumpolar_min_lon_span: Minimum longitude span (°) to accept
            a contour as circumpolar (default 355°).
    """

    level_mode: str | int = "wavg"
    try_levels: int = 300
    min_vertices: int = 30
    n_meridians: int = 240
    min_cross: int = 3
    n_samp: int = 240
    min_points: int = 15
    area_min_deg2: float = 30.0
    x_min_awb: float = -1.0
    x_max_cwb: float = 1.0
    circumpolar_min_lon_span: float = 355.0
    # Plot styling
    pv_cmap: str = "coolwarm"
    base_contour_color: str = "k"
    color_awb: str = "dodgerblue"
    color_cwb: str = "tomato"
    fill_alpha: float = 0.28
    contour_ovt_color: str = "gold"
    contour_ovt_lw: float = 2.0
    contour_base_lw: float = 1.2

    # ── Backward-compat aliases (unused by new code) ──
    @property
    def max_keep(self) -> int:
        return 999  # not limited in circumpolar mode

    @max_keep.setter
    def max_keep(self, _v: int) -> None:
        pass  # silently ignored


# =====================================================================
# Circumpolar contour extraction  (MATLAB bay-method equivalent)
# =====================================================================
def circumpolar_contours(
    field_nh: np.ndarray,
    lat_nh: np.ndarray,
    lon_nh: np.ndarray,
    try_levels: int = 300,
    min_vertices: int = 30,
    min_lon_span: float = 355.0,
) -> list[dict]:
    """Extract circumpolar contours from a full Northern-Hemisphere field.

    Mimics the MATLAB ``blue_circumcontour`` approach:
      1. Double the field in longitude to handle periodicity.
      2. Run ``find_contours`` at ``try_levels`` evenly-spaced values.
      3. Keep only the **longest** contour at each level that spans ≥
         ``min_lon_span`` degrees of longitude (i.e. circumpolar).
      4. Return contours in (lon, lat) geographic coordinates.

    Args:
        field_nh: 2-D array (nlat, nlon), NH field (e.g. Z at one
            pressure level). Latitude axis must be **ascending**
            (equator → pole) or **descending** — handled automatically.
        lat_nh: 1-D latitude array corresponding to axis 0 of *field_nh*.
        lon_nh: 1-D longitude array corresponding to axis 1 (0→360 or
            −180→180).
        try_levels: Number of contour levels to probe.
        min_vertices: Minimum contour vertices.
        min_lon_span: Minimum longitude span to qualify as circumpolar.

    Returns:
        List of dicts ``{'lev': float, 'lon': ndarray, 'lat': ndarray}``
        with coordinates in the **original** geographic system.
    """
    if find_contours is None:
        raise ImportError(
            "scikit-image is required for RWB detection. "
            "Install it with: pip install scikit-image"
        )

    lat = np.asarray(lat_nh, dtype=float)
    lon = np.asarray(lon_nh, dtype=float)
    f2d = np.asarray(field_nh, dtype=float)

    # Ensure latitude is ascending (equator→pole) for find_contours
    lat_ascending = lat[0] < lat[-1]
    if not lat_ascending:
        lat = lat[::-1]
        f2d = f2d[::-1, :]

    # Safe fill NaNs
    if not np.isfinite(f2d).all():
        finite = np.isfinite(f2d)
        fill = np.nanmedian(f2d[finite]) if np.any(finite) else 0.0
        f2d = np.where(finite, f2d, fill)

    nlon = len(lon)
    dlon = float(lon[1] - lon[0]) if nlon > 1 else 1.0

    # ── Double in longitude (like MATLAB's [lon; lon(1:80)+360]) ──
    # We append the first ~half of the longitudes shifted by 360°
    n_wrap = min(nlon // 2 + 1, nlon)
    lon_ext = np.concatenate([lon, lon[:n_wrap] + 360.0])
    f2d_ext = np.concatenate([f2d, f2d[:, :n_wrap]], axis=1)

    vmin = float(np.nanmin(f2d_ext))
    vmax = float(np.nanmax(f2d_ext))
    probe = np.linspace(vmin, vmax, try_levels)

    r_idx = np.arange(len(lat), dtype=float)
    c_idx = np.arange(len(lon_ext), dtype=float)

    results: list[dict] = []
    for lev in probe:
        cs = find_contours(f2d_ext, level=lev)
        if not cs:
            continue
        # Take the longest contour at this level
        cmax = max(cs, key=len)
        if len(cmax) < min_vertices:
            continue
        rows, cols = cmax[:, 0], cmax[:, 1]
        lon_c = np.interp(cols, c_idx, lon_ext)
        lat_c = np.interp(rows, r_idx, lat)

        # Circumpolarity check: must span ≥ min_lon_span
        lon_span = float(np.max(lon_c) - np.min(lon_c))
        if lon_span < min_lon_span:
            continue

        # Must stay in NH (all lat ≥ 0)
        if np.min(lat_c) < 0:
            continue

        results.append({"lev": float(lev), "lon": lon_c, "lat": lat_c})

    return results


def crop_contour_to_patch(
    contour: dict,
    centre_lat: float,
    centre_lon: float,
    half_dlat: float = 21.0,
    half_dlon: float = 36.0,
) -> dict | None:
    """Crop a circumpolar contour to an event-centred patch, returning
    relative coordinates (Δlon, Δlat).

    Args:
        contour: Dict with 'lev', 'lon', 'lat' from ``circumpolar_contours``.
        centre_lat, centre_lon: Patch centre in geographic coords.
        half_dlat, half_dlon: Half-extents of patch (degrees).

    Returns:
        Dict ``{'lev', 'x', 'y'}`` in relative coordinates, or None if
        too few points survive the crop (< 10).
    """
    lon_c = np.asarray(contour["lon"], dtype=float)
    lat_c = np.asarray(contour["lat"], dtype=float)

    # Convert to relative coords
    dx = lon_c - centre_lon
    dy = lat_c - centre_lat

    # Handle longitude wrapping (bring dx into ±180 range)
    dx = (dx + 180.0) % 360.0 - 180.0

    # Mask to patch bounds
    in_patch = (np.abs(dx) <= half_dlon) & (np.abs(dy) <= half_dlat)
    if in_patch.sum() < 10:
        return None

    # Keep contiguous runs within patch (preserve path order)
    # Find contiguous blocks of True values
    diffs = np.diff(in_patch.astype(int))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    if in_patch[0]:
        starts = np.concatenate([[0], starts])
    if in_patch[-1]:
        ends = np.concatenate([ends, [len(in_patch)]])

    if len(starts) == 0 or len(ends) == 0:
        return None

    # Take the longest contiguous block within the patch
    best_start, best_end, best_len = 0, 0, 0
    for s, e in zip(starts, ends):
        if e - s > best_len:
            best_start, best_end, best_len = s, e, e - s

    if best_len < 10:
        return None

    return {
        "lev": contour["lev"],
        "x": dx[best_start:best_end],
        "y": dy[best_start:best_end],
    }


# =====================================================================
# Legacy helper (backward compatibility)
# =====================================================================
def sampled_longest_contours(
    field2d: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    try_levels: int = 300,
    max_keep: int = 12,
    min_vertices: int = 30,
) -> list[dict]:
    """Sample the longest contours on a LOCAL patch (legacy helper).

    .. deprecated:: 0.4.0
        Prefer ``circumpolar_contours`` + ``crop_contour_to_patch``.

    Parameters:
        field2d: 2D field to contour (e.g., geopotential height).
        x_coords: 1D x-axis values (relative longitude degrees).
        y_coords: 1D y-axis values (relative latitude degrees).
        try_levels: Number of levels to probe between min/max.
        max_keep: Maximum number of contours to retain.
        min_vertices: Minimum vertices for valid contour.

    Returns:
        List of dicts with keys: 'lev', 'x', 'y'.
    """
    # Safe field for contours
    arr = np.array(field2d, dtype=float, copy=True)
    if not np.isfinite(arr).all():
        finite = np.isfinite(arr)
        fill = np.nanmedian(arr[finite]) if np.any(finite) else 0.0
        arr[~finite] = fill

    if find_contours is None:
        raise ImportError(
            "scikit-image is required for RWB detection. "
            "Install it with: pip install scikit-image"
        )

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    probe = np.linspace(vmin, vmax, try_levels)

    r_index = np.arange(len(y_coords), dtype=float)
    c_index = np.arange(len(x_coords), dtype=float)

    candidates = []
    for lev in probe:
        cs = find_contours(arr, level=lev)
        if not cs:
            continue
        cmax = max(cs, key=len)
        if len(cmax) < min_vertices:
            continue
        rows, cols = cmax[:, 0], cmax[:, 1]
        xline = np.interp(cols, c_index, x_coords)
        yline = np.interp(rows, r_index, y_coords)
        candidates.append({"lev": float(lev), "x": xline, "y": yline})

    if not candidates:
        return []
    idxs = (np.linspace(0, len(candidates) - 1,
                        num=min(max_keep, len(candidates)))
            .round().astype(int))
    return [candidates[i] for i in idxs]


# =====================================================================
# Low-level geometry helpers
# =====================================================================
def _vertical_intersections_sorted(xline, yline, x0):
    """All y where polyline crosses x=x0, sorted by value."""
    ys = []
    for i in range(len(xline) - 1):
        x1, x2 = xline[i], xline[i + 1]
        y1, y2 = yline[i], yline[i + 1]
        dx = x2 - x1
        if dx == 0.0:
            if np.isclose(x1, x0):
                ys.extend([y1, y2])
            continue
        if (x0 >= min(x1, x2)) and (x0 <= max(x1, x2)):
            t = (x0 - x1) / dx
            ys.append(y1 + t * (y2 - y1))
    return np.sort(np.asarray(ys)) if ys else np.asarray([])


def _vertical_intersections_pathorder(xline, yline, x0):
    """All y where polyline crosses x=x0, in contour trace (path) order."""
    ys_path = []
    for i in range(len(xline) - 1):
        x1, x2 = xline[i], xline[i + 1]
        y1, y2 = yline[i], yline[i + 1]
        dx = x2 - x1
        if dx == 0.0:
            continue
        if (x0 >= min(x1, x2)) and (x0 <= max(x1, x2)):
            t = (x0 - x1) / dx
            ys_path.append(y1 + t * (y2 - y1))
    return ys_path


def overturn_x_intervals(
    xline: np.ndarray,
    yline: np.ndarray,
    n_meridians: int = 200,
    min_cross: int = 3,
) -> list[tuple[float, float]]:
    """Find x-intervals where contour is overturning (folded).

    Parameters:
        xline, yline: Contour coordinates.
        n_meridians: Number of meridians to probe.
        min_cross: Minimum crossings to flag overturning.

    Returns:
        List of (x_start, x_end) intervals.
    """
    xmin, xmax = float(np.min(xline)), float(np.max(xline))
    xm = np.linspace(xmin, xmax, n_meridians)
    cross = np.zeros_like(xm, dtype=int)
    for j, x0 in enumerate(xm):
        ys = _vertical_intersections_sorted(xline, yline, x0)
        cross[j] = len(ys)
    folded = cross >= min_cross
    if not np.any(folded):
        return []
    intervals, in_run, start = [], False, None
    for j, flag in enumerate(folded):
        if flag and not in_run:
            in_run, start = True, xm[j]
        elif not flag and in_run:
            in_run = False
            intervals.append((start, xm[j]))
    if in_run:
        intervals.append((start, xm[-1]))
    return intervals


def classify_bay(
    xline: np.ndarray,
    yline: np.ndarray,
    xa: float,
    xb: float,
    n_samp: int = 200,
    min_valid: int = 5,
) -> tuple[str, float]:
    """Classify a bay as AWB or CWB using path-order sign.

    MATLAB-consistent: if first max(y) index < first min(y) index -> CWB.

    **No tilt-slope fallback** — returns "UNK" if sign is ambiguous.

    Parameters:
        xline, yline: Contour trace coordinates.
        xa, xb: x-interval of the bay.
        n_samp: Number of sample meridians.
        min_valid: Minimum valid signs for classification.

    Returns:
        (wb_type, sign_median) where wb_type in {"AWB", "CWB", "UNK"}.
    """
    xm = np.linspace(xa, xb, n_samp)
    signs = []
    for x0 in xm:
        ys_path = _vertical_intersections_pathorder(xline, yline, x0)
        if len(ys_path) < 2:
            continue
        y_max = max(ys_path)
        y_min = min(ys_path)
        idx_max = next(i for i, v in enumerate(ys_path) if np.isclose(v, y_max))
        idx_min = next(i for i, v in enumerate(ys_path) if np.isclose(v, y_min))
        signs.append(1.0 if idx_max < idx_min else -1.0)
    if len(signs) < min_valid:
        return "UNK", np.nan
    med = float(np.nanmedian(signs))
    if med > 0:
        return "CWB", med
    elif med < 0:
        return "AWB", med
    return "UNK", med


def centerline_tilt(
    xm: np.ndarray,
    y_min: np.ndarray,
    y_max: np.ndarray,
) -> float:
    """Slope of the bay centreline (mid-point of envelope).

    Parameters:
        xm: x-coordinates of envelope samples.
        y_min, y_max: Lower/upper envelope y-values.

    Returns:
        Slope (deg lat / deg lon).  Negative -> AWB; positive -> CWB.
        Returns ``np.nan`` if fewer than 3 valid points.
    """
    y_c = 0.5 * (y_max + y_min)
    if xm.size < 3:
        return np.nan
    slope = np.polyfit(xm, y_c, 1)[0]
    return float(slope)


# Tilt dead-zone threshold (consistent with ESSOAr 10.22541/essoar.175821720.04705853)
TILT_SLOPE_THRESHOLD = 0.15


def classify_tilt(
    xline: np.ndarray,
    yline: np.ndarray,
    xa: float,
    xb: float,
    n_samp: int = 200,
    min_points: int = 10,
    slope_threshold: float = TILT_SLOPE_THRESHOLD,
) -> tuple[str, float]:
    """Classify a bay as AWB or CWB using centerline tilt slope.

    Slope < -threshold -> AWB; slope > +threshold -> CWB;
    |slope| <= threshold -> UNK (neutral).
    Default threshold = 0.15 (consistent with ESSOAr preprint).

    Parameters:
        xline, yline: Contour trace coordinates.
        xa, xb: x-interval of the bay.
        n_samp: Number of sample meridians.
        min_points: Minimum valid sample points for envelope.
        slope_threshold: Dead-zone half-width (default 0.15).

    Returns:
        (wb_type, slope) where wb_type in {"AWB", "CWB", "UNK"}.
    """
    poly = envelope_polygon(xline, yline, xa, xb,
                            n_samp=n_samp, min_points=min_points)
    if poly is None:
        return "UNK", np.nan
    _xp, _yp, xm, y_min_arr, y_max_arr = poly
    slope = centerline_tilt(xm, y_min_arr, y_max_arr)
    if not np.isfinite(slope):
        return "UNK", np.nan
    if slope < -slope_threshold:
        return "AWB", slope
    elif slope > slope_threshold:
        return "CWB", slope
    return "UNK", slope


def envelope_polygon(
    xline: np.ndarray,
    yline: np.ndarray,
    xa: float,
    xb: float,
    n_samp: int = 200,
    min_points: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Construct envelope polygon for an overturning interval.

    Parameters:
        xline, yline: Contour coordinates.
        xa, xb: x-interval.
        n_samp: Sample points.
        min_points: Minimum valid sample points.

    Returns:
        (xp, yp, xm, y_min, y_max) or None.
    """
    xm = np.linspace(xa, xb, n_samp)
    y_min_arr = np.full_like(xm, np.nan, dtype=float)
    y_max_arr = np.full_like(xm, np.nan, dtype=float)
    for j, x0 in enumerate(xm):
        ys = _vertical_intersections_sorted(xline, yline, x0)
        if ys.size >= 2:
            y_min_arr[j] = ys[0]
            y_max_arr[j] = ys[-1]
    mask = np.isfinite(y_min_arr) & np.isfinite(y_max_arr)
    if mask.sum() < min_points:
        return None
    xm = xm[mask]
    y_min_arr = y_min_arr[mask]
    y_max_arr = y_max_arr[mask]
    xp = np.concatenate([xm, xm[::-1]])
    yp = np.concatenate([y_min_arr, y_max_arr[::-1]])
    return xp, yp, xm, y_min_arr, y_max_arr


def poly_area_centroid(
    xp: np.ndarray, yp: np.ndarray,
) -> tuple[float, tuple[float, float]]:
    """Signed planar area and centroid of a polygon.

    Parameters:
        xp, yp: Polygon vertex coordinates.

    Returns:
        (area, (cx, cy)) where area is in squared coordinate units.
    """
    x = np.asarray(xp)
    y = np.asarray(yp)
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    A = 0.5 * np.nansum(cross)
    if np.isclose(A, 0.0):
        return 0.0, (np.nan, np.nan)
    Cx = (1.0 / (6.0 * A)) * np.nansum((x + x1) * cross)
    Cy = (1.0 / (6.0 * A)) * np.nansum((y + y1) * cross)
    return float(A), (float(Cx), float(Cy))


# =====================================================================
# Main entry point: circumpolar-first RWB detection
# =====================================================================
def detect_rwb_events(
    field2d_patch: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    cfg: RWBConfig | None = None,
    *,
    field_nh: np.ndarray | None = None,
    lat_nh: np.ndarray | None = None,
    lon_nh: np.ndarray | None = None,
    centre_lat: float | None = None,
    centre_lon: float | None = None,
    method: str = "bay",
) -> list[dict]:
    """Detect and classify all RWB bays in a 2D field.

    **Circumpolar-first mode** (preferred):
        Pass the full-NH field via *field_nh / lat_nh / lon_nh* together
        with *centre_lat / centre_lon*.  The algorithm first finds all
        circumpolar contours on the full hemisphere, then crops them to
        the patch for overturning analysis.

    **Legacy local-patch mode** (fallback):
        If *field_nh* is not supplied, falls back to
        ``sampled_longest_contours`` on the local patch.

    Parameters:
        field2d_patch: 2D field on the event patch (nlat_p, nlon_p).
        x_coords: 1D relative x coordinates (Δlon).
        y_coords: 1D relative y coordinates (Δlat).
        cfg: RWB configuration (defaults used if None).
        field_nh: Full NH 2D field (nlat, nlon).  Enables circumpolar mode.
        lat_nh: Full NH latitude array.
        lon_nh: Full NH longitude array.
        centre_lat: Patch centre latitude.
        centre_lon: Patch centre longitude.
        method: Classification method -- ``"bay"`` (default, path-order,
            no tilt fallback) or ``"tilt"`` (centerline-tilt slope).

    Returns:
        List of dicts with: wb_type, area, centroid, contour_level,
        polygon (xp, yp), interval (xa, xb), method.
    """
    if cfg is None:
        cfg = RWBConfig()

    if method not in ("bay", "tilt"):
        raise ValueError(f"method must be 'bay' or 'tilt', got {method!r}")

    # ── Decide which contour source to use ──
    use_circumpolar = (
        field_nh is not None
        and lat_nh is not None
        and lon_nh is not None
        and centre_lat is not None
        and centre_lon is not None
    )

    if use_circumpolar:
        # ── Circumpolar-first approach ──
        half_dlat = float(np.max(np.abs(y_coords)))
        half_dlon = float(np.max(np.abs(x_coords)))

        circ = circumpolar_contours(
            field_nh, lat_nh, lon_nh,
            try_levels=cfg.try_levels,
            min_vertices=cfg.min_vertices,
            min_lon_span=cfg.circumpolar_min_lon_span,
        )
        # Crop each to the event patch
        contours: list[dict] = []
        for cc in circ:
            cropped = crop_contour_to_patch(
                cc, centre_lat, centre_lon,
                half_dlat=half_dlat, half_dlon=half_dlon,
            )
            if cropped is not None:
                contours.append(cropped)
    else:
        # ── Legacy local-patch fallback ──
        contours = sampled_longest_contours(
            field2d_patch, x_coords, y_coords,
            try_levels=cfg.try_levels,
            max_keep=12,
            min_vertices=cfg.min_vertices,
        )

    # ── Overturning detection + classification (same for both modes) ──
    bays: list[dict] = []
    for c in contours:
        xline, yline = c["x"], c["y"]
        intervals = overturn_x_intervals(
            xline, yline,
            n_meridians=cfg.n_meridians,
            min_cross=cfg.min_cross,
        )
        for xa, xb in intervals:
            poly = envelope_polygon(xline, yline, xa, xb,
                                     n_samp=cfg.n_samp,
                                     min_points=cfg.min_points)
            if poly is None:
                continue
            xp, yp, xm, y_min, y_max = poly
            area, (cx, cy) = poly_area_centroid(xp, yp)
            if abs(area) < cfg.area_min_deg2:
                continue

            if method == "bay":
                wb_type, sign = classify_bay(
                    xline, yline, xa, xb,
                    n_samp=max(80, cfg.n_samp // 2),
                )
            else:  # method == "tilt"
                wb_type, sign = classify_tilt(
                    xline, yline, xa, xb,
                    n_samp=cfg.n_samp,
                    min_points=cfg.min_points,
                )

            # Centroid-x gates
            if wb_type == "AWB" and cx < cfg.x_min_awb:
                continue
            if wb_type == "CWB" and cx > cfg.x_max_cwb:
                continue

            bays.append({
                "wb_type": wb_type,
                "area": abs(area),
                "centroid": (cx, cy),
                "sign": sign,
                "contour_level": c["lev"],
                "polygon_x": xp,
                "polygon_y": yp,
                "interval": (xa, xb),
                "method": method,
            })

    return bays


def nearest_level_index(levels: np.ndarray, hpa: int) -> int:
    """Find nearest pressure level index."""
    return int(np.abs(levels - hpa).argmin())


def weighted_mean_2d(
    arr3d: np.ndarray, z3d_m: np.ndarray, H_SCALE: float,
) -> np.ndarray:
    """Exponential height-weighted vertical average.

    Parameters:
        arr3d: 3D field (nlev, nlat, nlon).
        z3d_m: Geopotential height [m] (nlev, nlat, nlon).
        H_SCALE: E-folding height scale [m].

    Returns:
        2D vertically-weighted average.
    """
    w = np.exp(-z3d_m / float(H_SCALE))
    valid = np.isfinite(arr3d)
    num = np.nansum(np.where(valid, arr3d * w, 0.0), axis=0)
    den = np.nansum(np.where(valid, w, 0.0), axis=0)
    out = np.full_like(num, np.nan, dtype=np.float64)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def reduce_to_2d(
    arr: np.ndarray,
    levels: np.ndarray,
    level_mode: int | str,
    z3d_m: np.ndarray | None = None,
    H_SCALE: float = 7000.0,
) -> np.ndarray:
    """Get 2D slice from 3D array using level index or weighted mean.

    Parameters:
        arr: Array, either 2D (returned as-is) or 3D (nlev, nlat, nlon).
        levels: Pressure levels [hPa].
        level_mode: Integer hPa or 'wavg'.
        z3d_m: Geopotential height for wavg (required if level_mode='wavg').
        H_SCALE: E-folding height scale [m].

    Returns:
        2D field.
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D, got shape {arr.shape}")

    if isinstance(level_mode, int):
        li = nearest_level_index(levels, level_mode)
        return arr[li]
    if isinstance(level_mode, str) and level_mode.lower() == "wavg":
        if z3d_m is None:
            raise ValueError("z3d_m required for wavg mode")
        return weighted_mean_2d(arr, z3d_m, H_SCALE)
    raise ValueError(f"Invalid level_mode={level_mode!r}")
