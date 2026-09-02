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
    #: Whether each inner linear solve met its tolerance.  A Newton step built
    #: on a solve that did not is a step in an approximate direction, and the
    #: outer iteration then stalls with heavy backtracking rather than failing
    #: outright -- which reads from outside like a nonlinear problem and is not
    #: one.  Recorded so that the distinction is a number in the output.
    linear_converged: list[bool] = field(default_factory=list)
    linear_residuals: list[float] = field(default_factory=list)
    #: Residuals of the equations as posed (no taper, no floors, no limiter) at
    #: the returned state: the balance row as metres of geopotential height and
    #: the potential-vorticity row in PVU, over the whole grid and poleward of
    #: the taper band.  Where the deformation limiter acted, the balance residual
    #: measures the limiter's own modification of that row as well, so it is
    #: large exactly where the balance equation as posed had no elliptic
    #: solution; the potential-vorticity residual is unaffected by it.
    final_norms: dict = field(default_factory=dict)
    #: Area fraction per interior level where the deformation limiter acted.
    deformation_fraction: list[float] = field(default_factory=list)
    #: The limiter itself, one factor per interior level on the grid, so the
    #: piecewise pass can freeze the same regularised system.
    deformation_limit: np.ndarray | None = field(default=None, repr=False)
    #: How many times the limiter was brought in or tightened.
    limiter_refreshes: int = 0
    #: Per interior level, the area fraction where the linearised balance row
    #: is not elliptic at the returned state under the limiter used.
    final_nonelliptic_fraction: list[float] = field(default_factory=list)

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
        deformation_limit: np.ndarray | None = None,
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

        The deformation limiter is handed in, not recomputed: it is a
        function of the state, so recomputing it at every evaluation would make
        the residual a different function from the one the operator is the
        derivative of.  With one limiter shared by the residual, the Jacobian
        and the half-state operator, the residual is exactly quadratic and the
        Newton step is a Newton step.  Left as ``None``, the limiter is taken
        from the state itself, which is the right thing for a one-off
        evaluation and the wrong thing inside an iteration.

        Returns:
            The packed residual, the operator built at the current state (the
            Jacobian), and its preconditioner.
        """
        interior = self.levels.interior
        jacobian = PiecewiseOperator(
            FrozenState(
                self.ops,
                self.levels,
                psi_spec,
                phi_spec,
                clamps=self.clamps,
                mirror=self.mirror,
                deformation_limit=deformation_limit,
            )
        )
        half = PiecewiseOperator(
            FrozenState(
                self.ops,
                self.levels,
                0.5 * psi_spec,
                0.5 * phi_spec,
                clamps=self.clamps,
                mirror=self.mirror,
                deformation_limit=jacobian.frozen.deform_limit,
            )
        )
        r1, r2 = half.apply(phi_spec[interior], psi_spec[interior])
        rhs1, rhs2 = half.rhs_rows(q_hat, theta_bot, theta_top)
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
        outside = np.abs(self.ops.grid.lat) >= self.mirror.blend_north
        return {
            "balance_m": float(np.abs(height).max() / G),
            "pv_pvu": float(np.abs(pv).max() * 1.0e6),
            "balance_m_extratropics": float(np.abs(height[:, outside]).max() / G),
            "pv_pvu_extratropics": float(np.abs(pv[:, outside]).max() * 1.0e6),
            "pv_pvu_rms_extratropics": float(
                np.sqrt(np.mean(pv[:, outside] ** 2)) * 1.0e6
            ),
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
        # The gauge pins the level-mean geopotential to zero; an observed state
        # carries tens of kilometres squared per second squared of it.  Removing
        # that constant first is a pure gauge shift, carries no wind, and keeps
        # the first residual and line search about the physics rather than about
        # the gauge.
        phi[:, 0, 0] -= np.mean(phi[interior, 0, 0].real)
        report = NewtonReport(steps=0, converged=False)

        # The deformation limiter is a frozen field: while it does not change,
        # the residual is one quadratic system and every Newton step is exact.
        # Under the adaptive policy it starts as one everywhere -- the balance
        # equation as posed -- and is brought in from the current iterate only
        # when an inner solve or a line search fails, the two signs that the
        # linearised balance row has lost ellipticity; the step is then retaken
        # on the regularised system.  Refreshes only ever lower it.
        policy = self.newton.deformation_limiter
        observed_limit = FrozenState(
            self.ops, lev, psi, phi, clamps=self.clamps, mirror=self.mirror
        ).deform_limit
        if policy == "observed":
            limiter = observed_limit
        else:
            limiter = np.ones((interior.size, self.ops.grid.nlat, self.ops.grid.nlon))

        def refreshed(current_psi, current_phi):
            """The limiter tightened to the observed state's and the iterate's.

            Every refresh takes the pointwise minimum of the limiter in force,
            the observed state's limiter and the current iterate's own.  The
            observed state's covers every strain-dominated region the data
            hold, which is where an iterate is about to fold; the iterate's
            adds whatever the balancing has grown since.
            """
            if policy == "off" or report.limiter_refreshes >= self.newton.max_limiter_refreshes:
                return None
            own = FrozenState(
                self.ops, lev, current_psi, current_phi,
                clamps=self.clamps, mirror=self.mirror,
            ).deform_limit
            candidate = np.minimum(limiter, np.minimum(own, observed_limit))
            if not np.any(candidate < limiter):
                return None
            return candidate

        for step in range(self.newton.max_steps):
            resid, op, pre = self.residual_via_operator(
                psi, phi, q_hat, theta_bot, theta_top, deformation_limit=limiter
            )
            rhs = -resid
            norm = float(np.linalg.norm(rhs))
            report.residuals.append(norm)

            # Stagnation at a fold announces itself before the line search
            # fails outright: several halvings and a residual that no longer
            # falls.  Bringing the limiter in at that point saves the dozen
            # steps of shrinking half-hearted progress that would otherwise
            # precede the failure.
            if (
                step > 0
                and report.backtracks[-1] >= 3
                and norm > 0.5 * report.residuals[-2]
                and norm > 1.0e-8 * report.residuals[0]
            ):
                update = refreshed(psi, phi)
                if update is not None:
                    limiter = update
                    report.limiter_refreshes += 1
                    resid, op, pre = self.residual_via_operator(
                        psi, phi, q_hat, theta_bot, theta_top, deformation_limit=limiter
                    )
                    rhs = -resid
                    norm = float(np.linalg.norm(rhs))
                    report.residuals[-1] = norm

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
            report.linear_converged.append(bool(lin.converged))
            report.linear_residuals.append(float(lin.residual))
            if not lin.converged:
                update = refreshed(psi, phi)
                if update is not None:
                    limiter = update
                    report.limiter_refreshes += 1
                    report.backtracks.append(0)
                    report.increments.append(float("inf"))
                    report.steps = step + 1
                    continue
            dphi, dpsi = op.unpack_state(delta)

            step_size, backtracks = self._line_search(
                psi, phi, dpsi, dphi, q_hat, theta_bot, theta_top, norm, limiter
            )
            report.backtracks.append(backtracks)
            if (
                backtracks >= self.newton.max_backtracks
                and norm > 1.0e-8 * report.residuals[0]
            ):
                # A failed line search far from convergence is the sign of a
                # fold; one at rounding level is not, and must not tighten the
                # limiter.
                update = refreshed(psi, phi)
                if update is not None:
                    limiter = update
                    report.limiter_refreshes += 1
                    step_size = 0.0
            psi[interior] += step_size * dpsi
            phi[interior] += step_size * dphi
            psi, phi = self._rebuild_boundaries(psi, phi, theta_bot, theta_top)

            increment = float(np.abs(self.ops.synth(step_size * dphi)).max())
            report.increments.append(increment)
            report.steps = step + 1
            report.deformation_fraction = [
                float(v) for v in np.mean(limiter < 1.0, axis=(1, 2))
            ]
            report.deformation_limit = limiter
            # A small increment reached only because the line search gave up is
            # a stall, not convergence: the test asks for both.
            if (
                increment < self.newton.phi_tol
                and backtracks < self.newton.max_backtracks
            ):
                report.converged = True
                break

        report.final_norms = self.residual_norms(
            psi, phi, q_hat, theta_bot, theta_top
        )
        # Ellipticity of the linearised balance row at the returned state under
        # the limiter it was solved with: the fraction of each level where the
        # symbol's smaller eigenvalue is negative.  Zero means the returned
        # state's tangent operator is definite everywhere; where it is not, the
        # Krylov solves of the piecewise pass may still converge, but the row
        # there is the tangent of a system with no elliptic solution.
        final = FrozenState(
            self.ops, lev, psi, phi, clamps=self.clamps, mirror=self.mirror
        )
        deform = np.stack(
            [final._deformation_magnitude(psi[k]) for k in interior]
        )
        smallest = final.avo - limiter * final.weight[None] * deform
        report.final_nonelliptic_fraction = [
            float(v) for v in np.mean(smallest < 0.0, axis=(1, 2))
        ]
        return psi, phi, report

    def _line_search(
        self, psi, phi, dpsi, dphi, q_hat, theta_bot, theta_top, norm0, limiter=None
    ) -> tuple[float, int]:
        """Backtrack until the residual actually falls.

        Newton's first step from observed fields can overshoot where the clamps
        are active: a clamped coefficient makes the Jacobian a poorer model of the
        residual there than it is elsewhere, so a full step can land with a larger
        residual than it started from.
        """
        interior = self.levels.interior
        step = 1.0
        best_step, best_norm = 0.0, norm0
        for attempt in range(self.newton.max_backtracks):
            trial_psi = psi.copy()
            trial_phi = phi.copy()
            trial_psi[interior] += step * dpsi
            trial_phi[interior] += step * dphi
            trial_psi, trial_phi = self._rebuild_boundaries(
                trial_psi, trial_phi, theta_bot, theta_top
            )
            resid, _, _ = self.residual_via_operator(
                trial_psi, trial_phi, q_hat, theta_bot, theta_top, limiter
            )
            norm = float(np.linalg.norm(resid))
            if norm <= (1.0 - self.newton.armijo * step) * norm0:
                return step, attempt
            if norm < best_norm:
                best_step, best_norm = step, norm
            step *= 0.5
        # No step met the sufficient-decrease test.  The best of those tried is
        # applied -- possibly none at all -- rather than an untested halving.
        return best_step, self.newton.max_backtracks

    def _rebuild_boundaries(self, psi, phi, theta_bot, theta_top):
        """Reset the two boundary levels to their hydrostatic ghost values."""
        return rebuild_boundary_levels(
            self.ops, self.levels, psi, phi, theta_bot, theta_top, self.frozen_f()
        )


def rebuild_boundary_levels(
    ops: SphereOps,
    levels: LevelSet,
    psi: np.ndarray,
    phi: np.ndarray,
    theta_bot: np.ndarray,
    theta_top: np.ndarray,
    f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Set levels ``0`` and ``NL-1`` of a state to the operator's ghost values.

    The geopotential's ghost is hydrostatic in the boundary temperature; the
    streamfunction's is the same temperature, less its area mean, over the
    Coriolis parameter -- see :func:`pvinv_sph.vertical.streamfunction_ghost`.
    Modifies ``psi`` and ``phi`` in place and returns them.
    """
    pi = levels.pi
    nlev = levels.nlev
    bot_spec = ops.analyze(theta_bot)
    top_spec = ops.analyze(theta_top)
    # The streamfunction ghost is built from the resolved temperature -- the
    # one the geopotential ghost above carries -- so that a temperature with
    # power beyond the truncation gives the two ghosts the same field.
    psi_bot = ops.analyze(
        streamfunction_ghost(ops.synth(bot_spec), f, ops.grid.weights)
    )
    psi_top = ops.analyze(
        streamfunction_ghost(ops.synth(top_spec), f, ops.grid.weights)
    )
    phi[0] = phi[1] + bot_spec * (pi[1] - pi[0])
    phi[nlev - 1] = phi[nlev - 2] - top_spec * (pi[nlev - 1] - pi[nlev - 2])
    psi[0] = psi[1] + psi_bot * (pi[1] - pi[0])
    psi[nlev - 1] = psi[nlev - 2] - psi_top * (pi[nlev - 1] - pi[nlev - 2])
    return psi, phi
