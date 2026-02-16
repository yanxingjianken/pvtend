"""Rossby Wave Breaking (RWB) identification.

Detects overturning Z/PV contours on event-centred patches and classifies
them as Anticyclonic Wave Breaking (AWB) or Cyclonic Wave Breaking (CWB)
using a MATLAB-consistent path-order algorithm.

Key features:
- Samples longest contours across multiple Z or PV levels
- Detects overturning (folding) via meridian intersection counting
- Classifies AWB/CWB using path-order of max/min y intersections
- Applies centroid-x gates to filter spurious detections
- Constructs envelope polygons for visualization

References:
    Peters D, Waugh D W (1996) J. Atmos. Sci. 53, 3013-3031.
    Thorncroft C D, Hoskins B J, McIntyre M E (1993) Q.J.R.M.S. 119, 17-55.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from skimage.measure import find_contours


@dataclass
class RWBConfig:
    """Configuration for RWB detection and classification.

    Attributes:
        level_mode: Vertical level mode - 'wavg' or integer hPa.
        try_levels: Number of contour levels to probe.
        max_keep: Maximum contours to keep.
        min_vertices: Minimum vertices for valid contour.
        n_meridians: Number of meridian probes for overturn detection.
        min_cross: Minimum crossings to flag overturning.
        n_samp: Number of sample points for polygon construction.
        min_points: Minimum valid points for polygon.
        area_min_deg2: Minimum bay area (deg²) filter.
        x_min_awb: AWB centroid x must be >= this.
        x_max_cwb: CWB centroid x must be <= this.
    """

    level_mode: str | int = "wavg"
    try_levels: int = 300
    max_keep: int = 12
    min_vertices: int = 30
    n_meridians: int = 240
    min_cross: int = 3
    n_samp: int = 240
    min_points: int = 15
    area_min_deg2: float = 30.0
    x_min_awb: float = -1.0
    x_max_cwb: float = 1.0
    # Plot styling
    pv_cmap: str = "coolwarm"
    base_contour_color: str = "k"
    color_awb: str = "dodgerblue"
    color_cwb: str = "tomato"
    fill_alpha: float = 0.28
    contour_ovt_color: str = "gold"
    contour_ovt_lw: float = 2.0
    contour_base_lw: float = 1.2


def sampled_longest_contours(
    field2d: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    try_levels: int = 300,
    max_keep: int = 12,
    min_vertices: int = 30,
) -> list[dict]:
    """Sample the longest contours across multiple levels.

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


def detect_rwb_events(
    field2d: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    cfg: RWBConfig | None = None,
) -> list[dict]:
    """Detect and classify all RWB bays in a 2D field.

    Parameters:
        field2d: 2D field (e.g., geopotential height or PV).
        x_coords: 1D relative x coordinates.
        y_coords: 1D relative y coordinates.
        cfg: RWB configuration (defaults used if None).

    Returns:
        List of dicts with: wb_type, area, centroid, contour_level,
        polygon (xp, yp), interval (xa, xb).
    """
    if cfg is None:
        cfg = RWBConfig()

    contours = sampled_longest_contours(
        field2d, x_coords, y_coords,
        try_levels=cfg.try_levels,
        max_keep=cfg.max_keep,
        min_vertices=cfg.min_vertices,
    )

    bays = []
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

            wb_type, sign = classify_bay(
                xline, yline, xa, xb,
                n_samp=max(80, cfg.n_samp // 2),
            )
            if wb_type == "UNK":
                # Tilt fallback
                y_c = 0.5 * (y_max + y_min)
                if xm.size >= 3:
                    slope = np.polyfit(xm, y_c, 1)[0]
                    wb_type = "AWB" if slope < 0 else "CWB"

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
