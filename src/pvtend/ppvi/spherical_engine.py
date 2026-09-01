"""Drive the global spherical inversion from this pipeline's data and conventions.

Two inversion engines are available and they answer the same question differently.

The *windowed* engine solves on a fixed 10.5-85.5 degree latitude band with a
longitude window centred on the event, by relaxation, and carries the response to
its own lateral walls as a piece of the decomposition.

The *spherical* engine solves on the closed sphere.  There is no lateral boundary,
so there is no wall piece and no boundary condition to choose: the decomposition is
exactly the sources -- potential vorticity level by level, plus the two boundary
temperatures.  It also has nothing to say about where the event is, so an event at
eighty degrees is the same calculation as one at forty.

That difference drives the shape of this module.  The windowed engine is handed the
window; the spherical engine is handed the whole northern hemisphere and the crop
happens afterwards.  Reading the full field is not overhead here -- it is the point.

The two engines write the same output keys, with one difference that follows from
the physics: ``*_ppvi_wall`` exists only for the windowed engine, because only a
bounded domain has a wall.  Everything not attributed to a source is in
``*_rot_anom_residual_ppvi`` for both.
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
        blend: Taper the reference coefficients to their smooth equatorial limits
            across a band about the equator.  On by default because a hemisphere
            mirrored onto the sphere has a kink there wherever the mean zonal wind
            does not vanish.
        newton_max_steps: Cap on the nonlinear iteration of the total inversion.
    """

    solver_nlat: int = 128
    solver_nlon: int = 256
    lmax: int | None = None
    pieces: str = "scale"
    blend: bool = True
    newton_max_steps: int = 60


def northern_hemisphere_state(
    dataset,
    time_index: int,
    plevs=WU_PLEVS,
    names: dict[str, str] | None = None,
    lat_name: str = "lat",
    lon_name: str = "lon",
    level_name: str = "level",
    z_divisor: float = 1.0,
) -> tuple[np.ndarray, ...]:
    """Read one state as the spherical engine needs it: the whole hemisphere.

    The engine has one set of conventions and they are not negotiable, because a
    field that violates them still looks like weather:

    * latitude ascending from the equator to the pole, longitude ascending from
      zero, level descending in pressure;
    * ``z`` a geopotential *height* in metres -- one archive stores height and
      another stores geopotential, so the divisor is a parameter rather than a
      guess;
    * ``t`` a temperature, not a potential temperature.  The inversion forms
      potential temperature itself, and a state handed the wrong one is out by a
      factor of the Exner function and entirely plausible.

    Args:
        dataset: An open dataset for one archive.
        time_index: Step to take.
        plevs: Levels to select, bottom-up.
        names: Variable names, defaulting to ``z``/``t``/``u``/``v``.
        z_divisor: Divide the height field by this.  Standard gravity for an
            archive that stores geopotential; one for an archive that stores
            height.

    Returns:
        ``(height, temperature, u, v, lat, lon)``, the first four shaped
        ``(nlev, nlat, nlon)``.
    """
    names = names or {"z": "z", "t": "t", "u": "u", "v": "v"}
    snapshot = dataset.isel({_time_dim(dataset): time_index})
    snapshot = snapshot.sel({level_name: list(plevs)})
    snapshot = snapshot.sortby(lat_name).sortby(lon_name)

    lon = np.asarray(snapshot[lon_name].values, dtype=float)
    if lon.min() < -1e-9:  # a -180..180 axis, which the transform cannot take
        snapshot = snapshot.assign_coords({lon_name: lon % 360.0}).sortby(lon_name)
    snapshot = snapshot.sel({lat_name: slice(0.0, 90.0)})

    lat = np.asarray(snapshot[lat_name].values, dtype=float)
    lon = np.asarray(snapshot[lon_name].values, dtype=float)
    fields = [
        np.asarray(snapshot[names[key]].values, dtype=float) for key in ("z", "t", "u", "v")
    ]
    fields[0] = fields[0] / z_divisor
    _check_conventions(lat, lon, fields)
    return (*fields, lat, lon)


def _time_dim(dataset) -> str:
    for name in ("time", "valid_time", "slot"):
        if name in dataset.dims:
            return name
    raise ValueError(f"no recognised time dimension in {list(dataset.dims)}")


def _check_conventions(lat, lon, fields) -> None:
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
            "than a height; pass z_divisor=9.80665"
        )
    temperature = fields[1]
    if np.nanmin(temperature) > 200.0 and np.nanmax(temperature) > 400.0:
        raise ValueError(
            "the temperature field looks like a potential temperature; the "
            "inversion forms that itself and needs the temperature"
        )


def invert_event(
    mean_fields,
    event_fields,
    lat,
    lon,
    centre,
    cfg: SphericalConfig | None = None,
    lat_half: float = 30.0,
    lon_half: float = 60.0,
):
    """Invert one event on the sphere and crop the pieces to its patch.

    Args:
        mean_fields, event_fields: ``(height, temperature, u, v)`` for the
            climatology and the event, from :func:`northern_hemisphere_state`.
        lat, lon: The axes those fields are on.
        centre: ``(latitude, longitude)`` of the event, the longitude in the same
            convention as ``lon``.
        cfg: Engine configuration.
        lat_half, lon_half: Half-widths of the output patch, in degrees.

    Returns:
        The object the vendored pipeline returns: ``arrays`` carrying this
        package's output keys, and ``meta``.
    """
    from .spherical.config import InversionConfig, MirrorConfig, NewtonConfig
    from .spherical.pipeline import invert_event as _invert

    cfg = cfg or SphericalConfig()
    if cfg.pieces not in ("scale", "per_level"):
        raise ValueError(f"pieces must be scale or per_level, got {cfg.pieces!r}")

    inversion = InversionConfig(
        mirror=MirrorConfig(blend=cfg.blend),
        newton=NewtonConfig(max_steps=cfg.newton_max_steps),
    )
    return _invert(
        mean_fields,
        event_fields,
        lat,
        lon,
        (float(centre[0]), float(centre[1]) % 360.0),
        cfg=inversion,
        lat_half=lat_half,
        lon_half=lon_half,
        solver_nlat=cfg.solver_nlat,
        solver_nlon=cfg.solver_nlon,
        lmax=cfg.lmax,
        pieces_mode=cfg.pieces,
    )


def piece_keys(pieces: str = "scale") -> list[str]:
    """The piece names this engine writes, for a given decomposition.

    There is no ``wall``: a closed domain has no boundary for a response to come
    from. Whatever the sources do not account for is in the residual.
    """
    if pieces == "scale":
        return list(SCALE_PIECES)
    return [str(p) for p in WU_PLEVS]
