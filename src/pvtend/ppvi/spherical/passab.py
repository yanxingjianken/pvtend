"""Pass A/B: potential vorticity and streamfunction from the observed state.

The streamfunction is the exact inverse Laplacian of the relative vorticity, up
to the constant that carries no wind.  On the closed sphere the Poisson problem
needs no boundary values, so this is one spectral division rather than an
iteration, and nothing in the answer depends on how an edge was seeded or
corrected.

The potential vorticity is written in Exner coordinates and scaled to the
right-hand side of the inversion:

``q_hat = (f + zeta) Phi_PiPi - grad(psi_Pi) . grad(Phi_Pi)``

which is the Ertel potential vorticity ``q_SI`` times ``p/(g kappa Pi)``.  It is
assembled from the observed state, never from the balanced fields, so the
inversion is driven by the data rather than by its own solution; by default with
the inversion's own stencils (see ``pv_source``), so that the observed state
satisfies the potential-vorticity row exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .levels import CP, KAPPA, LevelSet, P0
from .mirror import coriolis_star
from .sphere import SphereOps
from .vertical import VerticalOperator, streamfunction_ghost


@dataclass
class DiagnosedState:
    """What the inversion needs to know about one atmospheric state.

    Attributes:
        psi_spec, phi_spec: Spectra on **all** levels [m^2 s^-1] and [m^2 s^-2].
        q_hat: Right-hand side of the potential-vorticity equation on interior
            levels, on the solver grid.
        theta_bot, theta_top: Boundary potential temperature [K] on the grid.
        theta: Potential temperature on all levels, kept for diagnostics.
        zeta: Relative vorticity on all levels.
    """

    psi_spec: np.ndarray
    phi_spec: np.ndarray
    q_hat: np.ndarray
    theta_bot: np.ndarray
    theta_top: np.ndarray
    theta: np.ndarray
    zeta: np.ndarray
    #: Area mean of the boundary temperature the geopotential implies
    #: hydrostatically, over the one the temperature gives, at the bottom and
    #: the top.  Near one for data that are hydrostatically consistent; a
    #: potential temperature passed as temperature, or a height passed as
    #: geopotential, moves the top value far from it while every other check
    #: still passes.
    boundary_theta_ratio: tuple[float, float] = (1.0, 1.0)


def potential_temperature(temperature: np.ndarray, p_hpa: np.ndarray) -> np.ndarray:
    """``theta = T (p0/p)**kappa``, with pressure varying along the first axis."""
    p_pa = np.asarray(p_hpa, dtype=float) * 100.0
    factor = (P0 / p_pa) ** KAPPA
    return temperature * factor.reshape((-1,) + (1,) * (temperature.ndim - 1))


def diagnose(
    ops: SphereOps,
    levels: LevelSet,
    geopotential: np.ndarray,
    temperature: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    f_floor_deg: float = 12.0,
    pv_source: str = "operator",
) -> DiagnosedState:
    """Diagnose streamfunction, boundary temperature and potential vorticity.

    Args:
        ops: Operators on the Gaussian solver grid.
        levels: Vertical levels; all arrays are ordered bottom-up along axis 0.
        geopotential: Geopotential [m^2 s^-2] (multiply a height by ``g``).
        temperature: Temperature [K] -- not potential temperature.  The Exner
            factor is applied here, so handing in theta applies it twice and
            costs about a factor of three in the potential vorticity.
        u, v: Eastward and northward wind [m s^-1].
        f_floor_deg: Latitude scale of the smoothed Coriolis floor.
        pv_source: ``"operator"`` evaluates the potential vorticity with the
            inversion's own discretisation -- the three-point non-uniform
            second difference of the geopotential for the stratification, the
            centred first differences with the hydrostatic ghosts for the shear
            terms, and the boundary temperature as the hydrostatic difference
            of the geopotential across the half level -- so that the data pair
            satisfies the potential-vorticity row of the system exactly.
            ``"data"`` is the limited-area convention: stratification and
            shear from plain centred differences of the temperature and the
            wind, and the boundary temperature as the mean of the two levels
            beside it.  On a stretched Exner grid the two differ by 5 to 30
            percent level by level (5 to 14 on the ERA5 walkthrough case, up
            to 29 at 850 hPa on a CESM winter state), and the total inversion has to move the
            geopotential to absorb the difference: on one winter event the
            balanced rotational wind missed the observed one at 250 hPa by
            6.4 m/s rms with the data source and 5.3 with this one.  Note
            that the shear terms at the first and last interior level then
            use the thermal-wind ghost in place of the observed 1000 and
            100 hPa streamfunction, as the operator does.

    Returns:
        A :class:`DiagnosedState` on the solver grid.
    """
    if pv_source not in ("operator", "data"):
        raise ValueError(f"pv_source must be 'operator' or 'data', got {pv_source!r}")
    nlev = levels.nlev
    for name, arr in (
        ("geopotential", geopotential),
        ("temperature", temperature),
        ("u", u),
        ("v", v),
    ):
        if arr.shape[0] != nlev:
            raise ValueError(
                f"{name} has {arr.shape[0]} levels, expected {nlev} for "
                f"{levels.name}"
            )
        if arr.shape[1:] != (ops.grid.nlat, ops.grid.nlon):
            raise ValueError(
                f"{name} has grid shape {arr.shape[1:]}, expected "
                f"{(ops.grid.nlat, ops.grid.nlon)}"
            )

    # Orientation guard.  Geopotential rises with height at every point, whatever
    # the weather, so this catches a top-down level axis -- which static stability
    # alone does not, since a reversed profile can still look stratified.
    descending = float(np.mean(np.diff(geopotential, axis=0) <= 0))
    if descending > 0.01:
        raise ValueError(
            f"geopotential decreases with height at {descending:.1%} of points; "
            f"the level axis must run bottom-up (1000 hPa first)"
        )

    theta = potential_temperature(temperature, levels.p_hpa)
    unstable = float(np.mean(np.diff(theta, axis=0) <= 0))
    if unstable > 0.05:
        # Local instability is real and the clamps handle it; this much of it is
        # a sign the input is not what it claims to be.
        raise ValueError(
            f"potential temperature decreases with height at {unstable:.1%} of "
            f"points; check that temperature, not theta, was supplied"
        )

    zeta = np.empty_like(u)
    psi_spec = np.empty((nlev, ops.sht.lmax + 1, ops.sht.lmax + 1), dtype=complex)
    for k in range(nlev):
        zeta_spec = ops.sht.vorticity(u[k], v[k])
        psi_spec[k] = ops.inv_lap(zeta_spec)
        zeta[k] = ops.synth(zeta_spec)

    phi_spec = np.stack([ops.analyze(geopotential[k]) for k in range(nlev)])

    lat = ops.grid.lat[:, None]
    f = np.broadcast_to(
        coriolis_star(lat, f_floor_deg), (ops.grid.nlat, ops.grid.nlon)
    )

    interior = levels.interior
    q_hat = np.empty((interior.size, ops.grid.nlat, ops.grid.nlon))
    if pv_source == "operator":
        vert = VerticalOperator(levels)
        bot_spec, top_spec = theta_from_geopotential(phi_spec, levels)
        theta_bot = ops.synth(bot_spec)
        theta_top = ops.synth(top_spec)
        phi_grid = np.stack([ops.synth(phi_spec[k]) for k in range(nlev)])
        d2phi = vert.d2(phi_grid[interior]) + vert.theta_forcing(theta_bot, theta_top)
        dphi_dpi = vert.d_dpi(phi_spec[interior], bot_spec, top_spec)
        weights = ops.grid.weights
        psi_bot = ops.analyze(streamfunction_ghost(theta_bot, f, weights))
        psi_top = ops.analyze(streamfunction_ghost(theta_top, f, weights))
        dpsi_dpi = vert.d_dpi(psi_spec[interior], psi_bot, psi_top)
        for i, k in enumerate(interior):
            px, py = ops.grad(dphi_dpi[i])
            sx, sy = ops.grad(dpsi_dpi[i])
            q_hat[i] = (f + zeta[k]) * d2phi[i] - (sx * px + sy * py)
    else:
        theta_spec = np.stack([ops.analyze(theta[k]) for k in range(nlev)])
        for i, k in enumerate(interior):
            two_dpi = 2.0 * levels.dpi2[k]
            dtheta_dpi = (theta[k + 1] - theta[k - 1]) / two_dpi
            du_dpi = (u[k + 1] - u[k - 1]) / two_dpi
            dv_dpi = (v[k + 1] - v[k - 1]) / two_dpi
            dtheta_dx, dtheta_dy = ops.grad(theta_spec[k])
            q_hat[i] = -(
                (f + zeta[k]) * dtheta_dpi + du_dpi * dtheta_dy - dv_dpi * dtheta_dx
            )
        theta_bot = 0.5 * (theta[0] + theta[1])
        theta_top = 0.5 * (theta[nlev - 2] + theta[nlev - 1])

    implied_bot, implied_top = theta_from_geopotential(phi_spec, levels)
    ratio = tuple(
        float(implied[0, 0].real / ops.analyze(given)[0, 0].real)
        for implied, given in (
            (implied_bot, 0.5 * (theta[0] + theta[1])),
            (implied_top, 0.5 * (theta[nlev - 2] + theta[nlev - 1])),
        )
    )
    return DiagnosedState(
        psi_spec=psi_spec,
        phi_spec=phi_spec,
        q_hat=q_hat,
        theta_bot=theta_bot,
        theta_top=theta_top,
        theta=theta,
        zeta=zeta,
        boundary_theta_ratio=ratio,
    )


def ertel_pv_si(q_hat: np.ndarray, levels: LevelSet) -> np.ndarray:
    """Convert the inversion's right-hand side back to Ertel PV in SI units."""
    from .levels import pv_rhs_scale

    scale = pv_rhs_scale(levels.p_hpa[levels.interior])
    return q_hat / scale.reshape((-1,) + (1,) * (q_hat.ndim - 1))


def ertel_pv_pvu(q_hat: np.ndarray, levels: LevelSet) -> np.ndarray:
    """Ertel potential vorticity in PVU (1 PVU = 1e-6 K m^2 kg^-1 s^-1)."""
    return ertel_pv_si(q_hat, levels) * 1.0e6


def theta_from_geopotential(phi_spec_or_grid: np.ndarray, levels: LevelSet) -> np.ndarray:
    """Boundary potential temperature implied hydrostatically by a geopotential.

    ``theta = -dPhi/dPi``, evaluated at the two half levels the inversion uses.
    This is the boundary temperature of the ``"operator"`` potential-vorticity
    source, and the check that a balanced state stayed consistent with the
    boundary data it was given.
    """
    pi = levels.pi
    bot = -(phi_spec_or_grid[1] - phi_spec_or_grid[0]) / (pi[1] - pi[0])
    top = -(phi_spec_or_grid[-1] - phi_spec_or_grid[-2]) / (pi[-1] - pi[-2])
    return bot, top


def check_pv_units(q_hat: np.ndarray, levels: LevelSet) -> dict[str, float]:
    """Summarise the potential vorticity in PVU, for a quick physical sanity read.

    Tropospheric values run a few tenths of a PVU near the surface to a few PVU at
    the tropopause; a summary that lands orders of magnitude away means a unit or
    a level-ordering mistake upstream, not an unusual atmosphere.
    """
    pvu = ertel_pv_pvu(q_hat, levels)
    return {
        "min": float(np.nanmin(pvu)),
        "median": float(np.nanmedian(pvu)),
        "max": float(np.nanmax(pvu)),
        "top_level_median": float(np.nanmedian(pvu[-1])),
    }
