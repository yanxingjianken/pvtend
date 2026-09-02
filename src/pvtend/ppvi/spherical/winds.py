"""Induced winds and the two ways of cropping them around an event.

The rotational wind comes straight from the streamfunction spectrally, so it is
finite everywhere including the pole rows -- there is no finite difference across
a singular point and no edge ring to discard.

Two croppings are offered, because a latitude-longitude patch cannot describe a
neighbourhood of the pole.  Once the box would run past 90 degrees, "twenty
degrees north of the centre" is not a place; the patch has to leave those rows
empty however the field was computed.

``geographic_patch``
    The familiar box: an event-centred window on the source grid, longitudes
    wrapped, rows beyond the pole left as NaN.

``rotated_patch``
    The event centre is carried to the equator of a rotated frame and the box is
    cut there, so the patch is complete and the same shape wherever the event sat.
    Wind components are rotated into the same frame, which is what makes a
    composite of high-latitude events meaningful -- "eastward" has to mean the
    same relative direction at every member.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from .sphere import SphereOps


def _require_regular(lat: np.ndarray, what: str) -> float:
    """Return the spacing of an equally spaced latitude axis, or refuse.

    Both croppings index the latitude axis linearly, so an unevenly spaced one --
    a Gaussian grid, say -- would place every row but the centre at the wrong
    latitude while still producing a plausible-looking patch.  Synthesise onto a
    regular grid first.
    """
    lat = np.asarray(lat, dtype=float)
    steps = np.diff(lat)
    if lat.size < 2 or not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9):
        raise ValueError(
            f"{what} needs an equally spaced latitude axis; this one is not "
            f"(spacings from {steps.min():.4f} to {steps.max():.4f})"
        )
    return float(steps[0])


def rotational_wind_stack(
    ops: SphereOps, psi_spec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(u, v)`` on the solver grid for a stack of streamfunction spectra."""
    nlev = psi_spec.shape[0]
    u = np.empty((nlev, ops.grid.nlat, ops.grid.nlon))
    v = np.empty_like(u)
    for k in range(nlev):
        u[k], v[k] = ops.rotational_wind(psi_spec[k])
    return u, v


@dataclass
class Patch:
    """A cropped field with the axes needed to interpret it."""

    values: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    lat_rel: np.ndarray
    lon_rel: np.ndarray
    frame: str

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape


def geographic_patch(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float,
    lon0: float,
    lat_half: float = 30.0,
    lon_half: float = 60.0,
    dlat: float | None = None,
    dlon: float | None = None,
) -> Patch:
    """Crop an event-centred latitude-longitude box, wrapping in longitude.

    Rows whose latitude would exceed 90 degrees are returned as NaN.  Not because
    they name nothing -- 90 + d degrees north is 90 - d on the opposite meridian,
    a real place with a real wind, and :func:`_pad_sphere` below depends on
    exactly that -- but because this container cannot label them.  Every column
    of the box carries one longitude, and continuing past the pole turns the
    longitude by half a circle, so those rows belong to a different longitude
    axis than the rest of the patch.  The rotated crop has no such trouble and
    keeps them.

    Marking them missing rather than dropping them is the contract downstream
    consumers expect, and it is what lets every event's patch keep the same
    declared shape however far north the centre sits.

    Args:
        field: ``(..., nlat, nlon)`` on a regular grid, latitude ascending.
        lat, lon: The field's axes in degrees.
        lat0, lon0: Event centre.
        lat_half, lon_half: Half-widths of the box in degrees.
        dlat, dlon: Grid spacings; inferred when omitted.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    spacing = _require_regular(lat, "geographic_patch")
    dlat = spacing if dlat is None else dlat
    dlon = float(lon[1] - lon[0]) if dlon is None else dlon

    npad_lat = int(round(lat_half / dlat))
    npad_lon = int(round(lon_half / dlon))
    ilat = int(np.argmin(np.abs(lat - lat0)))
    ilon = int(np.argmin(np.abs((lon - lon0 + 180.0) % 360.0 - 180.0)))

    rows = np.arange(ilat - npad_lat, ilat + npad_lat + 1)
    cols = (np.arange(ilon - npad_lon, ilon + npad_lon + 1)) % lon.size
    inside = (rows >= 0) & (rows < lat.size)

    shape = field.shape[:-2] + (rows.size, cols.size)
    out = np.full(shape, np.nan)
    out[..., inside, :] = field[..., rows[inside][:, None], cols[None, :]]

    lat_out = np.full(rows.size, np.nan)
    lat_out[inside] = lat[rows[inside]]
    return Patch(
        values=out,
        lat=lat_out,
        lon=lon[cols],
        lat_rel=(np.arange(rows.size) - npad_lat) * dlat,
        lon_rel=(np.arange(cols.size) - npad_lon) * dlon,
        frame="geographic",
    )


def rotate_to_pole(
    lat_deg: np.ndarray, lon_deg: np.ndarray, lat0: float, lon0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Geographic coordinates of points given in a frame centred on ``(lat0, lon0)``.

    The rotated frame puts the event centre on its own equator at zero longitude,
    with rotated north pointing along the geographic meridian through the centre.
    Rotated latitude is then distance from the centre along that meridian and
    rotated longitude is the perpendicular direction, so a box cut in this frame
    stays a box no matter how close to the pole the centre is.
    """
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    lam = np.radians(np.asarray(lon_deg, dtype=float))
    phi0 = np.radians(float(lat0))
    lam0 = np.radians(float(lon0))

    x = np.cos(phi) * np.cos(lam)
    y = np.cos(phi) * np.sin(lam)
    z = np.sin(phi)

    # Columns of the rotation: where the rotated frame's own axes point in
    # geographic space.  The rotated x axis is the event centre, the rotated z
    # axis is 90 degrees along the meridian through it, and the rotated y axis is
    # the local eastward direction there -- which makes the frame right-handed and
    # fixes the sign of the longitude.
    centre = np.array(
        [np.cos(phi0) * np.cos(lam0), np.cos(phi0) * np.sin(lam0), np.sin(phi0)]
    )
    north = np.array(
        [-np.sin(phi0) * np.cos(lam0), -np.sin(phi0) * np.sin(lam0), np.cos(phi0)]
    )
    east = np.array([-np.sin(lam0), np.cos(lam0), 0.0])

    xg = centre[0] * x + east[0] * y + north[0] * z
    yg = centre[1] * x + east[1] * y + north[1] * z
    zg = centre[2] * x + east[2] * y + north[2] * z

    lat_geo = np.degrees(np.arcsin(np.clip(zg, -1.0, 1.0)))
    lon_geo = np.degrees(np.arctan2(yg, xg)) % 360.0
    return lat_geo, lon_geo


def frame_rotation_angle(
    lat_geo: np.ndarray, lon_geo: np.ndarray, lat0: float, lon0: float
) -> np.ndarray:
    """Angle from geographic north to rotated north, at each point [radians].

    This is the initial bearing from the point towards the rotated frame's pole,
    which sits 90 degrees along the meridian through the event centre.
    """
    pole_lat = 90.0 - float(lat0)
    pole_lon = (float(lon0) + 180.0) % 360.0
    phi = np.radians(np.asarray(lat_geo, dtype=float))
    phi_p = np.radians(pole_lat)
    dlam = np.radians(np.asarray(lon_geo, dtype=float) - pole_lon)
    return np.arctan2(
        np.sin(-dlam) * np.cos(phi_p),
        np.cos(phi) * np.sin(phi_p) - np.sin(phi) * np.cos(phi_p) * np.cos(-dlam),
    )


#: Rows and columns of padding added before interpolating.  A cubic spline is
#: fitted by a recursive filter whose boundary handling contaminates the first few
#: samples; the influence decays by about a factor of four per cell, so a dozen
#: cells of genuine data on each side leaves nothing measurable.  Without it the
#: overshoot near a pole reaches over a percent.
_PAD = 12


def _pad_sphere(field: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Extend a scalar field past both poles and around in longitude.

    Continuing a scalar across a pole is a half-turn in longitude: the value at
    ``90 + d`` degrees north is the value at ``90 - d`` on the opposite meridian.
    With that and a wrap in longitude, every interpolation stencil sits on real
    data, and the boundary mode of the interpolator stops mattering.
    """
    shift = lon.size // 2
    top = np.roll(field[..., -_PAD - 1 : -1, :][..., ::-1, :], shift, axis=-1)
    bottom = np.roll(field[..., 1 : _PAD + 1, :][..., ::-1, :], shift, axis=-1)
    padded = np.concatenate([bottom, field, top], axis=-2)
    return np.concatenate(
        [padded[..., -_PAD:], padded, padded[..., :_PAD]], axis=-1
    )


def _sample(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, lat_q, lon_q):
    """Cubic sample of a regular grid, valid up to and across the poles.

    The field must be a genuine scalar; wind components are not, and are sampled
    through their Cartesian counterparts in :func:`rotated_patch`.
    """
    dlat = _require_regular(lat, "rotated_patch sampling")
    dlon = float(lon[1] - lon[0])
    padded = _pad_sphere(np.asarray(field, dtype=float), lon)
    rows = (np.asarray(lat_q) - lat[0]) / dlat + _PAD
    cols = ((np.asarray(lon_q) - lon[0]) % 360.0) / dlon + _PAD
    flat = padded.reshape((-1,) + padded.shape[-2:])
    out = np.stack(
        [
            map_coordinates(
                layer, [rows.ravel(), cols.ravel()], order=3, mode="nearest"
            ).reshape(rows.shape)
            for layer in flat
        ]
    )
    return out.reshape(field.shape[:-2] + rows.shape)


def to_cartesian(u, v, lat, lon):
    """Wind components in a fixed Earth-centred frame.

    Unlike the eastward and northward components, these are smooth scalars
    everywhere -- the singularity at the pole is in the local basis, not in the
    wind -- so they can be interpolated across it.
    """
    phi = np.radians(lat)[:, None]
    lam = np.radians(lon)[None, :]
    e_lon = (-np.sin(lam), np.cos(lam), np.zeros_like(lam))
    e_lat = (-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi))
    return tuple(u * a + v * b for a, b in zip(e_lon, e_lat))


def from_cartesian(vec, lat_q, lon_q):
    """Project a Cartesian wind back onto the local eastward/northward basis."""
    phi = np.radians(lat_q)
    lam = np.radians(lon_q)
    e_lon = (-np.sin(lam), np.cos(lam), np.zeros_like(lam))
    e_lat = (-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi))
    u = sum(c * a for c, a in zip(vec, e_lon))
    v = sum(c * b for c, b in zip(vec, e_lat))
    return u, v


def rotated_patch(
    u: np.ndarray,
    v: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float,
    lon0: float,
    lat_half: float = 30.0,
    lon_half: float = 60.0,
    dlat: float = 1.0,
    dlon: float = 1.0,
    cartesian: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[Patch, Patch]:
    """Crop a wind pair in a frame whose equator passes through the event centre.

    Complete for any centre, including one at the pole, and the same shape for
    every event -- which is what a composite needs.  The components are rotated
    into the same frame, so "eastward" means the same relative direction
    everywhere in the patch.

    The sampling is a cubic interpolation rather than a spectral evaluation at the
    rotated points; on a grid fine enough to resolve the solution that costs far
    less than the truncation already applied, and the alternative needs a rotation
    of the spectrum itself.  It goes through the Cartesian components of the wind,
    which are smooth across the pole where the eastward and northward ones are not.

    ``cartesian`` supplies those components directly.  Pass them whenever ``u``
    and ``v`` are undefined on the pole rows -- as they are on any grid that has
    one -- because a single missing value spreads through the spline's prefilter
    and empties the whole patch.
    """
    lat_rel = np.arange(-lat_half, lat_half + 0.5 * dlat, dlat)
    lon_rel = np.arange(-lon_half, lon_half + 0.5 * dlon, dlon)
    lon_mesh, lat_mesh = np.meshgrid(lon_rel, lat_rel)
    lat_geo, lon_geo = rotate_to_pole(lat_mesh, lon_mesh, lat0, lon0)

    if cartesian is None:
        cartesian = to_cartesian(np.asarray(u), np.asarray(v), lat, lon)
    sampled = tuple(_sample(c, lat, lon, lat_geo, lon_geo) for c in cartesian)
    u_geo, v_geo = from_cartesian(sampled, lat_geo, lon_geo)

    alpha = frame_rotation_angle(lat_geo, lon_geo, lat0, lon0)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    u_rot = u_geo * cos_a - v_geo * sin_a
    v_rot = u_geo * sin_a + v_geo * cos_a

    common = dict(
        lat=lat_geo, lon=lon_geo, lat_rel=lat_rel, lon_rel=lon_rel, frame="rotated"
    )
    return Patch(values=u_rot, **common), Patch(values=v_rot, **common)
