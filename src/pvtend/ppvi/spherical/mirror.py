"""Extending a hemisphere of data to the sphere the solver needs.

The data are northern-hemisphere only, but the operator is global.  The extension
has to keep the inversion elliptic and must not let the invented hemisphere leak
into the answer.  Both follow from one choice: mirror everything as an *even*
function of latitude and use ``|f|``.

With even coefficients and even sources, the operator commutes with a reflection
about the equator, so the solution is exactly even and its northern half solves
the northern problem under a homogeneous Neumann condition at the equator.  The
southern half is then scaffolding in the strict sense -- it is determined by the
northern half and carries no independent information.

Mirroring the *winds* instead (eastward even, northward odd, as a vector field
demands) would make the streamfunction odd and the relative vorticity odd, so
``f + zeta`` would change sign across the equator, ellipticity would be lost and
the scaffold would contaminate the north at order ``zeta/f``.  That combination
is the one to avoid.

An even mirror is only continuous, not smooth, wherever the mean zonal wind does
not vanish at the equator.  :func:`blend_to_limit` tapers the *coefficients* to
their smooth equatorial limits across a band around the equator; sources and
solutions are never blended, so the operator stays shared between pieces and the
exact sum-closure of the piecewise inversion is untouched.
"""
from __future__ import annotations

import numpy as np

from .levels import OMEGA


def coriolis_star(lat_deg: np.ndarray, floor_deg: float = 12.0) -> np.ndarray:
    """Equator-symmetric Coriolis parameter with a smooth floor [s^-1].

    ``2 Omega sqrt(sin^2(lat) + sin^2(floor))`` replaces ``|f|``, whose corner at
    the equator would otherwise appear inside ``grad(f)`` in the balance term.

    The floor is not a small correction near the tropics.  With the default 12
    degrees the ratio to ``|f|`` is 25.4 at 0.5N, 2.59 at 5N, 1.52 at 10.5N, 1.17
    at 20N, 1.12 at 24N, 1.08 at 30N and 1.04 at 45N -- it falls below a few
    percent only in mid-latitudes.  Anything quoted about the subtropical part of
    a solution has to be checked against a run with a smaller floor before it is
    called physics.
    """
    s = np.sin(np.radians(np.asarray(lat_deg, dtype=float)))
    s0 = np.sin(np.radians(float(floor_deg)))
    return 2.0 * OMEGA * np.sqrt(s * s + s0 * s0)


def _layout(lat_nh: np.ndarray) -> tuple[bool, float]:
    """Classify a northern-hemisphere latitude axis, returning (shares_equator, dlat)."""
    lat = np.asarray(lat_nh, dtype=float).ravel()
    if lat.size < 2:
        raise ValueError("need at least two latitudes to mirror")
    if lat[0] > lat[-1]:
        raise ValueError("latitudes must ascend from the equator to the pole")
    dlat = float(lat[1] - lat[0])
    tol = max(1e-3, 1e-3 * dlat)
    if abs(lat[0]) <= tol:
        return True, dlat
    if abs(abs(lat[0]) - 0.5 * dlat) <= tol:
        return False, dlat
    raise ValueError(
        f"latitudes start at {lat[0]:.4f} with spacing {dlat:.4f}: neither on the "
        f"equator nor half a step from it, so this band is not a hemisphere"
    )


def mirrored_latitudes(lat_nh: np.ndarray) -> np.ndarray:
    """Global latitude axis produced by :func:`mirror_even`."""
    lat = np.asarray(lat_nh, dtype=float).ravel()
    shares_equator, _ = _layout(lat)
    if shares_equator:
        return np.concatenate([-lat[1:][::-1], lat])
    return np.concatenate([-lat[::-1], lat])


def mirror_even(field: np.ndarray, lat_nh: np.ndarray) -> np.ndarray:
    """Reflect a northern-hemisphere field onto the sphere as an even function.

    Args:
        field: Array with latitude on axis ``-2``, ascending from the equator, and
            longitude on axis ``-1``.
        lat_nh: The northern latitudes, ascending.

    Returns:
        The global field; latitude axis has ``2 n - 1`` rows when a row sits on the
        equator and ``2 n`` when the rows straddle it.
    """
    lat = np.asarray(lat_nh, dtype=float).ravel()
    shares_equator, _ = _layout(lat)
    if field.shape[-2] != lat.size:
        raise ValueError(
            f"field has {field.shape[-2]} latitude rows but the axis has {lat.size}"
        )
    south = field[..., 1:, :][..., ::-1, :] if shares_equator else field[..., ::-1, :]
    return np.concatenate([south, field], axis=-2)


def mirror_odd(field: np.ndarray, lat_nh: np.ndarray) -> np.ndarray:
    """Reflect a northern-hemisphere field onto the sphere as an odd function.

    This is the parity of the northward wind component, whose sign flips under a
    reflection.  It is used only to build a mirrored *vector* field whose relative
    vorticity agrees with the hemisphere's own -- the state the solver works with
    is mirrored evenly instead; see :mod:`pvinv_sph.prepare`.
    """
    lat = np.asarray(lat_nh, dtype=float).ravel()
    shares_equator, _ = _layout(lat)
    if field.shape[-2] != lat.size:
        raise ValueError(
            f"field has {field.shape[-2]} latitude rows but the axis has {lat.size}"
        )
    if shares_equator:
        south = -field[..., 1:, :][..., ::-1, :]
        out = np.concatenate([south, field], axis=-2)
        out[..., lat.size - 1, :] = 0.0  # an odd field vanishes on the equator
        return out
    return np.concatenate([-field[..., ::-1, :], field], axis=-2)


def restrict_to_nh(field_global: np.ndarray, lat_nh: np.ndarray) -> np.ndarray:
    """Inverse of :func:`mirror_even`: keep the northern rows."""
    lat = np.asarray(lat_nh, dtype=float).ravel()
    return field_global[..., -lat.size :, :]


def blend_weight(
    lat_deg: np.ndarray, blend_south: float = 5.0, blend_north: float = 20.0
) -> np.ndarray:
    """Smooth ramp: 0 equatorward of ``blend_south``, 1 poleward of ``blend_north``.

    A quintic smoothstep, so the weight is twice differentiable at both ends and
    the blended coefficients keep a rapidly converging spectrum.
    """
    if not blend_north > blend_south >= 0.0:
        raise ValueError(
            f"need 0 <= blend_south < blend_north, got {blend_south}, {blend_north}"
        )
    x = (np.abs(np.asarray(lat_deg, dtype=float)) - blend_south) / (
        blend_north - blend_south
    )
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def blend_to_limit(
    field: np.ndarray,
    lat_deg: np.ndarray,
    limit: np.ndarray | float,
    blend_south: float = 5.0,
    blend_north: float = 20.0,
) -> np.ndarray:
    """Taper a coefficient field to ``limit`` across the equatorial band.

    Args:
        field: Array with latitude on axis ``-2``.
        lat_deg: Latitudes of that axis.
        limit: Value to relax towards -- zero for the deformation and cross-term
            coefficients, the level's area mean for static stability.
        blend_south, blend_north: Edges of the ramp, in degrees of latitude.
    """
    w = blend_weight(lat_deg, blend_south, blend_north)
    shape = (1,) * (field.ndim - 2) + (lat_deg.size, 1)
    w = w.reshape(shape)
    return limit + w * (field - limit)
