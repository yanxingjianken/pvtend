"""Pass C: the total balanced state, by Newton-Krylov.

Both nonlinearities are quadratic, so the Jacobian of the residual is *exactly*
the linearised operator of the piecewise pass evaluated at the current iterate.
Pass C therefore needs no solver of its own -- it reuses
:class:`~pvinv_sph.operator.PiecewiseOperator` and its preconditioner, and each
Newton step is one linear solve.

Newton, rather than alternation, is what makes that work.  Relaxing the two
fields in turn -- the streamfunction with the geopotential held fixed, then the
geopotential with the streamfunction held fixed, with the vertical coupling
lagged between sweeps -- does not converge on this system: the vertical stencil
has vanishing row sums, so nothing damps a lagged column, and the two fields
drift together at a rate no under-relaxation below about 0.07 could contain.
Newton sees the full off-diagonal blocks at once, and the vertical coupling is
implicit in both the operator and the preconditioner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ClampConfig, KrylovConfig, MirrorConfig, NewtonConfig
from .krylov import solve
from .levels import LevelSet
from .operator import FrozenState, PiecewiseOperator
from .precond import SeparablePreconditioner
from .sphere import SphereOps
from .vertical import VerticalOperator, streamfunction_ghost


@dataclass
class NewtonReport:
    """Convergence history of one nonlinear solve."""

    steps: int
    converged: bool
    increments: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    linear_iterations: list[int] = field(default_factory=list)
    backtracks: list[int] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - human-facing
        state = "converged" if self.converged else "NOT converged"
        last = self.increments[-1] if self.increments else float("nan")
        return (
            f"{state} in {self.steps} Newton steps; final geopotential increment "
            f"{last:.3g} m2 s-2, linear iterations {self.linear_iterations}"
        )


class BalancedInversion:
    """Solve the nonlinear balance and potential-vorticity system for one state."""

    def __init__(
        self,
        ops: SphereOps,
        levels: LevelSet,
        clamps: ClampConfig | None = None,
        mirror: MirrorConfig | None = None,
        krylov: KrylovConfig | None = None,
        newton: NewtonConfig | None = None,
    ):
        self.ops = ops
        self.levels = levels
        self.vert = VerticalOperator(levels)
        self.clamps = clamps or ClampConfig()
        self.mirror = mirror or MirrorConfig()
        self.krylov = krylov or KrylovConfig()
        self.newton = newton or NewtonConfig()

    # -- residual -----------------------------------------------------------

    def residual_via_operator(
        self,
        psi_spec: np.ndarray,
        phi_spec: np.ndarray,
        q_hat: np.ndarray,
        theta_bot: np.ndarray,
        theta_top: np.ndarray,
    ) -> tuple[np.ndarray, PiecewiseOperator, SeparablePreconditioner]:
        """Residual evaluated through the same path as the Jacobian.

        Both equations are quadratic, so for any state ``x``

            ``F(x) = J_{x/2}[x] + F(0)``

        exactly -- the derivative taken at the midpoint reproduces a finite
        increment.  Building the residual by applying the operator whose reference
        state is ``x/2`` therefore gives the true residual *and* guarantees that
        the Jacobian describes the same system, clamps and equatorial blending
        included.

        Evaluating the residual from the pristine equations instead leaves the two
        inconsistent wherever a coefficient was blended or floored, and Newton
        then stalls after roughly halving the residual: the step solves one system
        while the residual measures another, and the difference between them is
        exactly what the iteration cannot remove.

        Returns:
            The packed residual, the operator built at the current state (the
            Jacobian), and its preconditioner.
        """
        interior = self.levels.interior
        half = PiecewiseOperator(
            FrozenState(
                self.ops,
                self.levels,
                0.5 * psi_spec,
                0.5 * phi_spec,
                clamps=self.clamps,
                mirror=self.mirror,
            )
        )
        r1, r2 = half.apply(phi_spec[interior], psi_spec[interior])
        rhs1, rhs2 = half.rhs_rows(q_hat, theta_bot, theta_top)

        jacobian = PiecewiseOperator(
            FrozenState(
                self.ops,
                self.levels,
                psi_spec,
                phi_spec,
                clamps=self.clamps,
                mirror=self.mirror,
            )
        )
        # Scaled with the Jacobian's own row scaling, so the Krylov tolerance
        # means the same thing on both sides of the step.
        residual = jacobian.pack_rows(r1 - rhs1, r2 - rhs2)
        return residual, jacobian, SeparablePreconditioner(jacobian)

    def residual(
        self,
        psi_spec: np.ndarray,
        phi_spec: np.ndarray,
        q_hat: np.ndarray,
        theta_bot: np.ndarray,
        theta_top: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Residuals of the two equations from their definitions, for reporting.

        Written straight from the physics rather than through the operator, with
        no clamping or blending, so it measures how well the returned state
        satisfies the equations as posed.  :meth:`residual_via_operator` is what
        the iteration itself uses.
        """
        ops, lev = self.ops, self.levels
        interior = lev.interior
        f = self.frozen_f()

        phi_grid_all = np.stack([ops.synth(phi_spec[k]) for k in range(lev.nlev)])
        d2phi = self.vert.d2(phi_grid_all[interior]) + self.vert.theta_forcing(
            theta_bot, theta_top
        )
        # The vertical differences act on spectra here, so the boundary ghosts
        # have to be spectra too -- mixing the two representations is a silent
        # broadcast error waiting to happen.
        bot_spec = ops.analyze(theta_bot)
        top_spec = ops.analyze(theta_top)
        dphi_dpi = self.vert.d_dpi(phi_spec[interior], bot_spec, top_spec)
        # The streamfunction takes the same temperature divided by f; see
        # :func:`pvinv_sph.vertical.streamfunction_ghost`.  Divided in grid space,
        # because f varies with latitude and a spectrum cannot be divided by it.
        psi_bot = ops.analyze(streamfunction_ghost(theta_bot, f, ops.grid.weights))
        psi_top = ops.analyze(streamfunction_ghost(theta_top, f, ops.grid.weights))
        dpsi_dpi = self.vert.d_dpi(psi_spec[interior], psi_bot, psi_top)

        r1 = np.empty((interior.size, ops.sht.lmax + 1, ops.sht.lmax + 1), complex)
        r2 = np.empty_like(r1)
        for i, k in enumerate(interior):
            zeta = ops.synth(ops.lap(psi_spec[k]))
            r1[i] = ops.lap(phi_spec[k]) - ops.balance_nonlinear(psi_spec[k], f)
            px, py = ops.grad(dphi_dpi[i])
            sx, sy = ops.grad(dpsi_dpi[i])
            field = (f + zeta) * d2phi[i] - (sx * px + sy * py) - q_hat[i]
            r2[i] = ops.analyze(field)
        return r1, r2

    def residual_norms(
        self,
        psi_spec: np.ndarray,
        phi_spec: np.ndarray,
        q_hat: np.ndarray,
        theta_bot: np.ndarray,
        theta_top: np.ndarray,
    ) -> dict[str, float]:
        """How badly a state misses each equation, in units one can judge.

        The two equations have different physical dimensions, so a single norm
        over both is not a number anyone can interpret -- and it is dominated by
        whichever row happens to carry larger figures.  Reported instead:

        ``balance_m``
            The balance residual inverted back to a geopotential and divided by
            ``g``, so it is a height error in metres and directly comparable to
            the 0.1 m convergence threshold of
            :class:`~pvinv_sph.config.NewtonConfig`.
        ``pv_pvu``
            The potential-vorticity residual in PVU.
        """
        from .levels import G, pv_rhs_scale

        r1, r2 = self.residual(psi_spec, phi_spec, q_hat, theta_bot, theta_top)
        height = np.stack(
            [self.ops.synth(self.ops.inv_lap(r1[k])) for k in range(r1.shape[0])]
        )
        scale = pv_rhs_scale(self.levels.p_hpa[self.levels.interior])
        pv = np.stack(
            [self.ops.synth(r2[k]) / scale[k] for k in range(r2.shape[0])]
        )
        return {
            "balance_m": float(np.abs(height).max() / G),
            "pv_pvu": float(np.abs(pv).max() * 1.0e6),
        }

    def frozen_f(self) -> np.ndarray:
        from .mirror import coriolis_star

        lat = self.ops.grid.lat[:, None]
        return np.broadcast_to(
            coriolis_star(lat, self.mirror.f_floor_deg),
            (self.ops.grid.nlat, self.ops.grid.nlon),
        )

    # -- solve --------------------------------------------------------------

    def solve(
        self,
        psi_spec: np.ndarray,
        phi_spec: np.ndarray,
        q_hat: np.ndarray,
        theta_bot: np.ndarray,
        theta_top: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, NewtonReport]:
        """Balance an observed state.

        Args:
            psi_spec, phi_spec: First guess on all levels -- the observed fields.
            q_hat: Observed potential vorticity on interior levels, solver grid.
            theta_bot, theta_top: Observed boundary potential temperature.

        Returns:
            Balanced ``(psi_spec, phi_spec)`` on all levels, and the report.  The
            boundary levels are rebuilt hydrostatically from the interior and the
            supplied boundary temperature, so the returned state satisfies the
            same ghost relation the operator assumed.
        """
        lev = self.levels
        interior = lev.interior
        psi = psi_spec.copy()
        phi = phi_spec.copy()
        report = NewtonReport(steps=0, converged=False)

        for step in range(self.newton.max_steps):
            resid, op, pre = self.residual_via_operator(
                psi, phi, q_hat, theta_bot, theta_top
            )
            rhs = -resid
            norm = float(np.linalg.norm(rhs))
            report.residuals.append(norm)

            cfg = self.krylov
            if self.newton.eisenstat_walker and report.residuals:
                ratio = norm / report.residuals[0]
                cfg = KrylovConfig(
                    rtol=float(min(0.1, max(self.krylov.rtol, 0.1 * ratio))),
                    maxiter=self.krylov.maxiter,
                    restart=self.krylov.restart,
                    method=self.krylov.method,
                )
            delta, lin = solve(op.matvec, rhs, preconditioner=pre.apply, cfg=cfg)
            report.linear_iterations.append(lin.iterations)
            dphi, dpsi = op.unpack_state(delta)

            step_size, backtracks = self._line_search(
                psi, phi, dpsi, dphi, q_hat, theta_bot, theta_top, norm
            )
            report.backtracks.append(backtracks)
            psi[interior] += step_size * dpsi
            phi[interior] += step_size * dphi
            psi, phi = self._rebuild_boundaries(psi, phi, theta_bot, theta_top)

            increment = float(np.abs(self.ops.synth(step_size * dphi)).max())
            report.increments.append(increment)
            report.steps = step + 1
            if increment < self.newton.phi_tol:
                report.converged = True
                break

        return psi, phi, report

    def _line_search(
        self, psi, phi, dpsi, dphi, q_hat, theta_bot, theta_top, norm0
    ) -> tuple[float, int]:
        """Backtrack until the residual actually falls.

        Newton's first step from observed fields can overshoot where the clamps
        are active: a clamped coefficient makes the Jacobian a poorer model of the
        residual there than it is elsewhere, so a full step can land with a larger
        residual than it started from.
        """
        interior = self.levels.interior
        step = 1.0
        for attempt in range(self.newton.max_backtracks):
            trial_psi = psi.copy()
            trial_phi = phi.copy()
            trial_psi[interior] += step * dpsi
            trial_phi[interior] += step * dphi
            trial_psi, trial_phi = self._rebuild_boundaries(
                trial_psi, trial_phi, theta_bot, theta_top
            )
            resid, _, _ = self.residual_via_operator(
                trial_psi, trial_phi, q_hat, theta_bot, theta_top
            )
            if float(np.linalg.norm(resid)) <= (
                1.0 - self.newton.armijo * step
            ) * norm0:
                return step, attempt
            step *= 0.5
        return step, self.newton.max_backtracks

    def _rebuild_boundaries(self, psi, phi, theta_bot, theta_top):
        """Reset the two boundary levels to their hydrostatic ghost values."""
        pi = self.levels.pi
        nlev = self.levels.nlev
        f = self.frozen_f()
        bot_spec = self.ops.analyze(theta_bot)
        top_spec = self.ops.analyze(theta_top)
        psi_bot = self.ops.analyze(streamfunction_ghost(theta_bot, f, self.ops.grid.weights))
        psi_top = self.ops.analyze(streamfunction_ghost(theta_top, f, self.ops.grid.weights))
        phi[0] = phi[1] + bot_spec * (pi[1] - pi[0])
        phi[nlev - 1] = phi[nlev - 2] - top_spec * (pi[nlev - 1] - pi[nlev - 2])
        psi[0] = psi[1] + psi_bot * (pi[1] - pi[0])
        psi[nlev - 1] = psi[nlev - 2] - psi_top * (pi[nlev - 1] - pi[nlev - 2])
        return psi, phi
