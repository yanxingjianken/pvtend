"""Turning a hemisphere of model output into a state the global solver can use.

Three things happen here, and the order matters.

**Below-ground filling.**  Pressure levels below the surface hold no data.  They
are filled by carrying temperature and wind down and integrating the geopotential
hydrostatically, so the column stays continuous without inventing a horizontal
structure the surface did not have.  This is not a rare correction: on CESM's f09
grid it touches about half the northern hemisphere at 1000 hPa.

**Mirroring, twice, with different parities.**  The wind is mirrored as the
vector it is -- eastward even, northward odd -- because that is what makes the
relative vorticity of the mirrored field agree with the hemisphere's own.  Then
the vorticity is taken, and *that* is re-mirrored as an even scalar.

The second mirror is the one the solver needs.  With ``|f|`` in the southern
scaffold, an even vorticity gives an even absolute vorticity, an even
geopotential gives an even static stability, and the whole operator commutes with
a reflection about the equator -- so the solution is exactly even and the northern
half solves the northern problem.  Mirroring the wind and stopping there would
leave the vorticity odd, the absolute vorticity asymmetric, and the scaffold free
to leak into the answer at order ``zeta/f``.

Everything else follows from the even vorticity and the even geopotential, and
comes out even too: the streamfunction, the wind derived from it (eastward odd,
northward even -- the mirror image of a circulation with the same sense of
rotation, which is what ``|f|`` describes), the potential temperature and the
potential vorticity.

**Regridding.**  Scalars move to the Gaussian solver grid through the spectrum.
Wind components are carried as ``u cos(lat)`` and ``v cos(lat)``, which are smooth
and vanish at the poles; the components themselves are not band limited on a
sphere and lose several digits if transformed directly.
"""
from __future__ import annotations

import numpy as np

from .levels import G, RD, LevelSet
from .mirror import mirror_even, mirror_odd, mirrored_latitudes
from .passab import DiagnosedState, diagnose
from .sht import SHT, Grid, grid_from_axes
from .sphere import SphereOps


def fill_below_ground(
    height: np.ndarray,
    temperature: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    p_hpa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill levels that lie below the surface, bottom-up arrays.

    Temperature and wind are carried downward unchanged; the height is continued
    hydrostatically, ``z_k = z_{k+1} + (Rd Tbar/g) ln(p_{k+1}/p_k)``.  A field with
    no missing values is returned untouched, so a dataset that never goes below
    ground is bit-identical either way.
    """
    if not (
        np.isnan(height).any()
        or np.isnan(temperature).any()
        or np.isnan(u).any()
        or np.isnan(v).any()
    ):
        return height, temperature, u, v

    height = np.array(height, dtype=float, copy=True)
    temperature = np.array(temperature, dtype=float, copy=True)
    u = np.array(u, dtype=float, copy=True)
    v = np.array(v, dtype=float, copy=True)
    p_pa = np.asarray(p_hpa, dtype=float) * 100.0

    for k in range(len(p_pa) - 2, -1, -1):
        for field in (temperature, u, v):
            gap = np.isnan(field[k])
            field[k][gap] = field[k + 1][gap]
        gap = np.isnan(height[k])
        if gap.any():
            t_bar = 0.5 * (temperature[k] + temperature[k + 1])
            height[k][gap] = (
                height[k + 1][gap]
                + (RD * t_bar[gap] / G) * np.log(p_pa[k + 1] / p_pa[k])
            )
    if np.isnan(height).any():
        raise ValueError(
            "levels remain unfilled after the below-ground sweep; the topmost "
            "level cannot be missing"
        )
    return height, temperature, u, v


def solver_grid_is_symmetric(grid: Grid) -> bool:
    """True when the grid's latitudes come in north-south pairs."""
    return bool(
        grid.nlat % 2 == 0
        and np.allclose(grid.lat[: grid.nlat // 2], -grid.lat[::-1][: grid.nlat // 2])
    )


def symmetrize_even(field: np.ndarray) -> np.ndarray:
    """Replace the southern half of a field by the mirror of its northern half.

    The latitude axis is ``-2`` and must be symmetric about the equator with an
    even number of rows -- true of the Gaussian solver grid by construction.
    """
    nlat = field.shape[-2]
    if nlat % 2:
        raise ValueError(
            f"even symmetrisation needs an even number of latitude rows, got {nlat}"
        )
    north = field[..., nlat // 2 :, :]
    return np.concatenate([north[..., ::-1, :], north], axis=-2)


def prepare_state(
    height: np.ndarray,
    temperature: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    lat_nh: np.ndarray,
    lon: np.ndarray,
    levels: LevelSet,
    ops: SphereOps,
    f_floor_deg: float = 12.0,
    pv_source: str = "operator",
    data_sht: SHT | None = None,
) -> DiagnosedState:
    """Diagnose one northern-hemisphere state on the global solver grid.

    Args:
        height: Geopotential height [m], bottom-up, ``(nlev, nlat_nh, nlon)``.
        temperature: Temperature [K] -- not potential temperature.
        u, v: Eastward and northward wind [m s^-1].
        lat_nh: Northern latitudes, ascending from the equator.
        lon: Longitudes in ``[0, 360)``.
        levels: Vertical levels.
        ops: Operators on the Gaussian solver grid, which must be symmetric about
            the equator.
        f_floor_deg: Latitude scale of the smoothed Coriolis floor.
        pv_source: Passed to :func:`pvinv_sph.passab.diagnose`.
        data_sht: Transform on the mirrored data grid, if the caller already
            holds one at the solver's truncation and radius; built here
            otherwise.  A batch over one archive reuses the same tables for
            every state rather than rebuilding them twice per event.

    Returns:
        A :class:`~pvinv_sph.passab.DiagnosedState` whose fields are even about
        the equator.
    """
    if not solver_grid_is_symmetric(ops.grid):
        raise ValueError(
            "the solver grid must be symmetric about the equator for the mirror "
            "to be exact"
        )
    lat_nh = np.asarray(lat_nh, dtype=float)
    if lat_nh[0] > lat_nh[-1]:
        raise ValueError("northern latitudes must ascend towards the pole")

    height, temperature, u, v = fill_below_ground(
        height, temperature, u, v, levels.p_hpa
    )

    # Physical mirror first: the wind is a vector, so the northward component
    # changes sign.  This is only scaffolding for the vorticity.
    lat_global = mirrored_latitudes(lat_nh)
    if data_sht is None:
        data_grid = grid_from_axes(lat_global, lon)
        data_sht = SHT(data_grid, lmax=ops.sht.lmax, radius=ops.sht.radius)
    elif (
        data_sht.lmax != ops.sht.lmax
        or data_sht.radius != ops.sht.radius
        or data_sht.grid.lat.shape != lat_global.shape
        or not np.allclose(data_sht.grid.lat, lat_global)
        or not np.allclose(data_sht.grid.lon, np.asarray(lon, dtype=float))
    ):
        raise ValueError(
            "data_sht must be on the mirrored data grid at the solver's "
            "truncation and radius"
        )

    cos_data = np.cos(np.radians(lat_global))[:, None]
    u_global = mirror_even(u, lat_nh)
    v_global = mirror_odd(v, lat_nh)

    # Carry the wind as u cos(lat) and v cos(lat) across the regrid: the bare
    # components are not band limited on a sphere, and transforming them directly
    # costs several digits.
    cos_solver = ops.grid.cos_lat[:, None]
    u_solver = (
        data_sht.regrid_to(ops.sht, u_global * cos_data) / cos_solver
    )
    v_solver = (
        data_sht.regrid_to(ops.sht, v_global * cos_data) / cos_solver
    )

    zeta = np.stack(
        [
            ops.synth(ops.sht.vorticity(u_solver[k], v_solver[k]))
            for k in range(levels.nlev)
        ]
    )
    zeta = symmetrize_even(zeta)
    psi_spec = np.stack(
        [ops.inv_lap(ops.analyze(zeta[k])) for k in range(levels.nlev)]
    )
    u_even = np.empty_like(zeta)
    v_even = np.empty_like(zeta)
    for k in range(levels.nlev):
        u_even[k], v_even[k] = ops.rotational_wind(psi_spec[k])

    # Scalars are mirrored evenly from the outset.
    height_solver = data_sht.regrid_to(ops.sht, mirror_even(height, lat_nh))
    temp_solver = data_sht.regrid_to(ops.sht, mirror_even(temperature, lat_nh))
    height_solver = symmetrize_even(height_solver)
    temp_solver = symmetrize_even(temp_solver)

    return diagnose(
        ops,
        levels,
        G * height_solver,
        temp_solver,
        u_even,
        v_even,
        f_floor_deg=f_floor_deg,
        pv_source=pv_source,
    )
