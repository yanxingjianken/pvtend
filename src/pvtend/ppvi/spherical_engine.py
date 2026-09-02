"""Drive the global spherical inversion from this pipeline's data and conventions.

The inversion solves on the closed sphere. There is no lateral boundary, so there
is no wall piece and no boundary condition to choose: the decomposition is exactly
the sources -- potential vorticity level by level, plus the two boundary
temperatures -- and everything they do not account for is in
``*_rot_anom_residual_ppvi``. It also has nothing to say about where the event is,
so an event at eighty degrees is the same calculation as one at forty.

That shapes this module. The solver is handed the whole northern hemisphere and the
crop happens afterwards; reading the full field is not overhead here, it is the
point.

The solver itself is the vendored :mod:`pvtend.ppvi.spherical` package and is not
edited here. This module owns what is specific to this pipeline: the archive's row
and column order, the reuse of one engine across a worker's events, and the crop
back onto the event patch -- by the patch's own row latitudes and column indices,
with the rows past the pole reading the antimeridian -- so the delivered arrays sit
on exactly the grid the rest of the record uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Levels the inversion works on, bottom-up, in hPa.
WU_PLEVS = [1000, 850, 700, 500, 400, 300, 250, 200, 100]

#: The four-piece decomposition by depth and, within the upper levels, by scale.
#: ``upper_p`` and ``upper_e`` share their levels and differ in the source they are
#: given, which is why the split travels as a source override.
SCALE_PIECES = ("surface", "lower", "upper_p", "upper_e")


@dataclass
class SphericalConfig:
    """What the spherical engine needs beyond the data.

    Attributes:
        solver_nlat, solver_nlon: The Gaussian grid the inversion runs on.  It is
            not the data grid: it excludes the poles, so no metric factor is ever
            divided by at a singular point, and it is chosen for the truncation
            rather than to match the archive.
        lmax: Spectral truncation.  ``None`` takes the largest the solver grid
            carries without aliasing the quadratic terms in the equations, which
            is two thirds of the number of latitudes.
        pieces: ``"scale"`` or ``"per_level"``.
        newton_max_steps: Cap on the nonlinear iteration of the total inversion.
            Real events converge in four to ten steps; the cap is a guard.
    """

    solver_nlat: int = 128
    solver_nlon: int = 256
    lmax: int | None = None
    pieces: str = "scale"
    newton_max_steps: int = 60

    def inversion_config(self):
        """The vendored solver's configuration this run uses.

        Everything not named here is the solver's default: the operator-consistent
        potential-vorticity source, the adaptive deformation limiter, the 5-20 N
        taper of the mirrored hemisphere, the Coriolis floor at 12 degrees.
        """
        from .spherical.config import InversionConfig, NewtonConfig

        if self.pieces not in ("scale", "per_level"):
            raise ValueError(f"pieces must be scale or per_level, got {self.pieces!r}")
        return InversionConfig(newton=NewtonConfig(max_steps=int(self.newton_max_steps)))


@dataclass
class HemisphereAxes:
    """How an archive's rows and columns map onto the engine's axes.

    The engine has one set of conventions and they are not negotiable, because a
    field that violates them still looks like weather: latitude ascending from
    the equator to the pole, longitude ascending from zero.  ERA5 is stored north
    to south; CESM f09 is stored south to north on a -180..180 longitude axis.
    Both are permutations of what the engine wants, and this records them once
    per grid so every cube is permuted the same way and the crop can be undone.

    Attributes:
        nh_rows: Archive latitude indices of the northern rows, ordered so that
            latitude ascends.
        lat_nh: Their latitudes, ascending.
        lon_order: Archive longitude indices ordered so that ``lon mod 360``
            ascends from zero.
        lon: The engine's longitude axis, in ``[0, 360)``.
        col_of: Engine column for each archive column index -- the inverse of
            ``lon_order``.
        dlat: Latitude spacing, degrees.
    """

    nh_rows: np.ndarray
    lat_nh: np.ndarray
    lon_order: np.ndarray
    lon: np.ndarray
    col_of: np.ndarray
    dlat: float

    @property
    def nlon(self) -> int:
        return int(self.lon.size)

    def far_columns(self, cols: np.ndarray) -> np.ndarray:
        """Engine columns half a turn away: the antimeridian of each column."""
        if self.nlon % 2:
            raise ValueError("the continuation across the pole needs an even "
                             f"number of longitudes, got {self.nlon}")
        return (np.asarray(cols) + self.nlon // 2) % self.nlon


def hemisphere_axes(lat_all: np.ndarray, lon_all: np.ndarray) -> HemisphereAxes:
    """Describe the archive's northern hemisphere in the engine's order."""
    lat_all = np.asarray(lat_all, dtype=float)
    lon_all = np.asarray(lon_all, dtype=float)
    nh_rows = np.where(lat_all >= -1e-9)[0]
    if nh_rows.size < 2:
        raise ValueError(f"no northern hemisphere in latitudes {lat_all.min()}..{lat_all.max()}")
    nh_rows = nh_rows[np.argsort(lat_all[nh_rows], kind="stable")]
    lat_nh = lat_all[nh_rows]
    lon_mod = np.mod(lon_all, 360.0)
    lon_order = np.argsort(lon_mod, kind="stable")
    lon = lon_mod[lon_order]
    col_of = np.empty(lon_all.size, dtype=int)
    col_of[lon_order] = np.arange(lon_all.size)
    dlat = float(np.median(np.diff(lat_nh)))
    return HemisphereAxes(
        nh_rows=nh_rows, lat_nh=lat_nh, lon_order=lon_order, lon=lon, col_of=col_of, dlat=dlat
    )


def check_conventions(lat: np.ndarray, lon: np.ndarray, fields) -> None:
    """Refuse a state that violates a convention, rather than inverting it anyway."""
    if lat[0] > lat[-1]:
        raise ValueError("latitude must ascend from the equator to the pole")
    if lat[0] < -1e-6:
        raise ValueError(f"expected a northern hemisphere, got latitudes from {lat[0]}")
    if abs(lon[0]) > 1e-6:
        raise ValueError(f"longitude must start at zero, got {lon[0]}")
    heights = fields[0]
    if np.nanmax(heights) > 1.0e5:
        raise ValueError(
            "the height field reaches above 100 km, which is a geopotential rather "
            "than a height; divide by standard gravity first"
        )
    temperature = fields[1]
    if np.nanmin(temperature) > 200.0 and np.nanmax(temperature) > 400.0:
        raise ValueError(
            "the temperature field looks like a potential temperature; the "
            "inversion forms that itself and needs the temperature"
        )


class SphericalEngine:
    """One engine per archive grid, built once and reused for every event.

    The spectral tables are the fixed cost; a worker builds them on its first
    event and inverts every later state through the same object.  ``fits`` says
    whether a dataset's axes are the ones it was built for.
    """

    def __init__(self, axes: HemisphereAxes, cfg: SphericalConfig | None = None) -> None:
        from .spherical.pipeline import SphereEngine

        self.axes = axes
        self.cfg = cfg or SphericalConfig()
        self.engine = SphereEngine.build(
            axes.lat_nh,
            axes.lon,
            cfg=self.cfg.inversion_config(),
            solver_nlat=self.cfg.solver_nlat,
            solver_nlon=self.cfg.solver_nlon,
            lmax=self.cfg.lmax,
        )

    def fits(self, axes: HemisphereAxes) -> bool:
        return self.engine.fits(axes.lat_nh, axes.lon)

    def invert(self, mean_fields, event_fields, centre):
        """Invert one event; fields ``(height, temperature, u, v)`` in engine order.

        Args:
            mean_fields, event_fields: Each ``(nlev, nlat_nh, nlon)``, bottom-up,
                rows and columns already in the engine's order (see
                :func:`to_engine_order`), heights in metres.
            centre: ``(latitude, longitude)`` of the event; any longitude
                convention.

        Returns:
            The vendored :class:`~pvtend.ppvi.spherical.pipeline.HemisphereInversion`:
            pieces on the mirrored data grid, its ``northern`` method giving the
            hemisphere's own rows.
        """
        from .spherical.pipeline import invert_hemisphere

        check_conventions(self.axes.lat_nh, self.axes.lon, mean_fields)
        check_conventions(self.axes.lat_nh, self.axes.lon, event_fields)
        # The Cartesian components are formed as well: they are what the pole
        # row of a patch is built from, where the eastward and northward
        # components of the regular grid are undefined.
        return invert_hemisphere(
            self.engine,
            tuple(np.ascontiguousarray(f, dtype=float) for f in mean_fields),
            tuple(np.ascontiguousarray(f, dtype=float) for f in event_fields),
            (float(centre[0]), float(centre[1]) % 360.0),
            pieces_mode=self.cfg.pieces,
            rotated_track=True,
        )


def to_engine_order(cube_nh: np.ndarray, axes: HemisphereAxes) -> np.ndarray:
    """Reorder the columns of a cube whose rows are already ``axes.nh_rows``."""
    return np.ascontiguousarray(np.asarray(cube_nh, dtype=float)[..., axes.lon_order])


def patch_row_index(lat_vec: np.ndarray, lat_nh: np.ndarray, dlat: float) -> np.ndarray:
    """Engine row for each patch-row latitude, ``-1`` where there is none.

    Matched by value, so it holds for either ordering of the patch.  Patch rows and
    engine rows are both exact grid latitudes, so real matches are exact; the
    tolerance only excludes a row one spacing away.  A patch row past the pole is
    NaN and stays unmatched.
    """
    lat_vec = np.asarray(lat_vec, dtype=float)
    row_for = np.full(lat_vec.shape[0], -1, dtype=int)
    tol = 0.25 * float(dlat)
    for j, plat in enumerate(lat_vec):
        if not np.isfinite(plat):
            continue
        r = int(np.abs(lat_nh - plat).argmin())
        if abs(lat_nh[r] - plat) < tol:
            row_for[j] = r
    return row_for


def crop_to_patch(
    field_nh: np.ndarray,
    row_for: np.ndarray,
    cols: np.ndarray,
    sign: np.ndarray | None = None,
) -> np.ndarray:
    """``(nlev, nlat_nh, nlon)`` in engine order to ``(nlev, yp, xp)`` on the patch.

    Args:
        field_nh: The hemisphere's rows, latitude ascending, engine columns.
        row_for: From :func:`patch_row_index`, on the rows' geographic latitude.
        cols: Engine column for each patch column, ``(xp,)`` for every row or
            ``(yp, xp)`` row by row -- the rows past the pole read the
            antimeridian.
        sign: Optional factor per row, ``-1`` on the rows past the pole for a
            horizontal vector component.

    Rows the patch has and the hemisphere does not are NaN.
    """
    field_nh = np.asarray(field_nh, dtype=float)
    cols = np.asarray(cols)
    yp = row_for.size
    xp = cols.shape[-1]
    out = np.full(field_nh.shape[:-2] + (yp, xp), np.nan, dtype=float)
    have = row_for >= 0
    if have.any():
        col_rows = cols[have] if cols.ndim == 2 else np.broadcast_to(cols, (int(have.sum()), xp))
        out[..., have, :] = field_nh[..., row_for[have][:, None], col_rows]
        if sign is not None:
            out[..., have, :] *= np.asarray(sign, dtype=float)[have][:, None]
    return out


def pole_row_winds(cartesian, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eastward and northward components on the pole row, column by column.

    On the pole the components of the regular grid are undefined; the wind is
    not. For the column at longitude λ the patch's frame is the local frame of
    meridian λ: eastward ``e_λ = (−sin λ, cos λ, 0)`` and northward the
    direction of travel through the pole, ``(−cos λ, −sin λ, 0)``. Projecting
    the Cartesian wind ``(Vx, Vy, Vz)`` onto them gives the pair the archive
    itself stores on its pole row.

    Args:
        cartesian: ``(Vx, Vy, Vz)``, each ``(..., nlon)`` on the pole row.
        lon: The columns' longitudes in degrees.
    """
    vx, vy, _ = cartesian
    lam = np.radians(np.asarray(lon, dtype=float))
    u = -vx * np.sin(lam) + vy * np.cos(lam)
    v = -vx * np.cos(lam) - vy * np.sin(lam)
    return u, v


def piece_keys(pieces: str = "scale") -> list[str]:
    """The piece names this engine writes, for a given decomposition.

    There is no ``wall``: a closed domain has no boundary for a response to come
    from. Whatever the sources do not account for is in the residual.
    """
    if pieces == "scale":
        return list(SCALE_PIECES)
    if pieces == "per_level":
        return [str(p) for p in WU_PLEVS]
    raise ValueError(f"pieces must be scale or per_level, got {pieces!r}")
