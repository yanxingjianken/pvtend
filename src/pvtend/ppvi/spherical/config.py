"""Configuration objects for a spherical piecewise inversion.

Every tunable the solver reads lives here, so a run is reproducible from one
dataclass.  Every value is stated in SI -- the level set, the potential-vorticity
floors, the absolute-vorticity and static-stability clamps alike -- because the
solver carries no non-dimensionalisation of its own and a scaled literal would
have no fixed physical meaning to check against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .mirror import coriolis_star

ClampMode = Literal["parity", "clean"]


@dataclass(frozen=True)
class SolverGrid:
    """Sizes of the Gaussian grid the solver collocates on.

    ``dealias`` sets the classical 3/2 rule: the balance and potential-vorticity
    equations are quadratic, so products of two fields truncated at ``lmax`` are
    represented without aliasing on a grid carrying ``3 lmax / 2``.  Turning it
    off collocates on the smallest grid that resolves ``lmax``, so the products
    are formed pointwise there and whatever they alias back onto the resolved
    scales is kept.
    """

    lmax: int
    dealias: bool = True

    @property
    def nlat(self) -> int:
        target = (3 * self.lmax + 1) // 2 + 1 if self.dealias else self.lmax + 1
        return int(target + (target % 2))  # even, so no row lands on the equator

    @property
    def nlon(self) -> int:
        return int(2 ** math.ceil(math.log2(max(4, 2 * self.nlat))))


@dataclass(frozen=True)
class MirrorConfig:
    """How the northern hemisphere is extended to a full sphere.

    The inversion is elliptic only where the absolute vorticity keeps one sign,
    so the southern scaffold uses ``|f|`` rather than ``f``.  With every
    coefficient and source mirrored as an even function, the operator commutes
    with a reflection about the equator and the solution is exactly even: the
    northern half then solves the northern problem under a homogeneous Neumann
    equator, and nothing crosses from the scaffold.

    An even mirror of the mean state has a kink at the equator wherever the mean
    zonal wind does not vanish there.  ``blend_south``/``blend_north`` taper the
    *coefficients* (never the sources, never the solution) to their smooth
    equatorial limits across that band, which keeps the operator shared between
    pieces and so leaves the exact sum-closure untouched.

    ``f_floor_deg`` smooths ``|f|`` itself into
    ``2 Omega sqrt(sin^2(lat) + sin^2(phi0))``.  At the default of twelve degrees
    the ratio to ``|f|`` is 1.53 at 10.5N, 1.16 at 20N and 1.08 at 30N, so the
    floor reaches well into the subtropics; together with the taper it makes a
    band around the equator where the operator is not the one being advertised.
    Both are set for the equator, not for the region a mid-latitude event's patch
    can reach, and a solution quoted in that band has to be checked against a run
    with a smaller floor and a narrower taper.

    ``blend`` should be off for a state that is already smooth across the
    equator, because the taper is a real modification of the system: with it on,
    the converged state satisfies the blended equations rather than the ones as
    posed, which measures about half a metre of equivalent geopotential height on
    a global synthetic case that needs no blending at all.  With it off on a
    mirrored state, the kink at the equator is a vortex sheet and rings.
    """

    blend: bool = True
    blend_south: float = 5.0
    blend_north: float = 20.0
    f_floor_deg: float = 12.0

    def f_star(self, lat_deg):
        """Smoothed, equator-symmetric Coriolis parameter [s^-1]."""
        return coriolis_star(lat_deg, self.f_floor_deg)


@dataclass(frozen=True)
class ClampConfig:
    """Floors that keep the linearised operator elliptic.

    ``avo_min`` and ``stb_min`` are the smallest absolute vorticity and static
    stability the linearised operator is allowed to see; the system is elliptic
    only where both keep one sign.  They are the SI values equivalent to a floor
    of 0.01 in the non-dimensional scaling this system is often written in,
    evaluated at the reference configuration (1.5 degrees, nine levels).  Holding
    them in SI is deliberate: the non-dimensional scale factors depend on both
    horizontal resolution and level count, so the same scaled literal silently
    changes physical meaning when either of those is altered.

    ``mode`` selects the shape of the floor.  ``"parity"`` tests against a tenth
    of the floor and then assigns the floor itself, so a point that fails is
    nudged clear of the threshold rather than pinned to it; it also stops the
    potential-vorticity redistribution after a single pass, which leaves a small
    conservation slack that is reported rather than hidden.  ``"clean"`` tests
    against the floor itself and iterates the redistribution until it conserves.
    """

    avo_min: float = 1.0e-6
    stb_min: float = 1.0e-4
    deformation_margin: float = 0.05
    mode: ClampMode = "parity"


@dataclass(frozen=True)
class KrylovConfig:
    """Stopping rules for the linear solves.

    ``maxiter`` is a budget of operator applications, not of restart cycles;
    :func:`pvinv_sph.krylov.solve` converts it for SciPy.  A well-conditioned
    piece converges in a few tens of applications, so the default leaves ample
    room while still bounding a pathological event to seconds rather than
    minutes.
    """

    rtol: float = 1.0e-8
    maxiter: int = 400
    restart: int = 60
    method: Literal["gmres", "bicgstab"] = "gmres"


@dataclass(frozen=True)
class NewtonConfig:
    """Stopping rules for the nonlinear (total-field) inversion.

    ``phi_tol`` stops the iteration when a Newton step moves the geopotential
    anywhere on the grid by less than a tenth of a metre of equivalent height.
    It is quoted in SI geopotential, ``g * 0.1``, because that is the unit the
    solved field carries.
    """

    phi_tol: float = 0.981
    max_steps: int = 20
    armijo: float = 1.0e-4
    max_backtracks: int = 8
    eisenstat_walker: bool = True


@dataclass(frozen=True)
class PVFloorConfig:
    """Potential-vorticity floors, in PVU.

    The total inversion floors the field at ``qmin_total`` and gives the added
    amount back to the unclamped points so the volume integral is preserved.  The
    perturbation inversion floors the event and mean states *separately* at
    ``qmin_pieces`` before differencing, which keeps a piece's source from
    inheriting a clamp that was applied to the other state.
    """

    qmin_total: float = 0.01
    qmin_pieces: float = 1.0e-5


@dataclass(frozen=True)
class InversionConfig:
    """Everything one inversion needs beyond the data itself."""

    levels: str = "NL9"
    lmax: int | None = None
    solver_dealias: bool = True
    mirror: MirrorConfig = field(default_factory=MirrorConfig)
    clamps: ClampConfig = field(default_factory=ClampConfig)
    krylov: KrylovConfig = field(default_factory=KrylovConfig)
    newton: NewtonConfig = field(default_factory=NewtonConfig)
    pv_floor: PVFloorConfig = field(default_factory=PVFloorConfig)

    def solver_grid(self, lmax: int) -> SolverGrid:
        return SolverGrid(lmax=lmax, dealias=self.solver_dealias)
