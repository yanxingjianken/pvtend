"""Pass D: the piecewise inversion.

Every piece is one linear solve against the *same* operator; only the source
changes.  That is what makes the pieces add up: the operator is frozen once, at
the mean plus half the perturbation, and the linearisation of a quadratic system
about its midpoint is exact, so the sum of the pieces is the all-sources solution
rather than an approximation to it.

The decomposition is exactly the sources: interior potential vorticity level by
level, plus the two boundary temperatures.  Nothing else enters it and nothing
has to be chosen.  On a bounded domain the elliptic problem would need data
imposed on its edge, and the response to that data -- eight to twenty-four
percent of the answer, measured on a bounded version of this problem -- would
then have to be attributed to one of the sources or carried as a piece of its
own.  On the closed sphere there is no edge, so the question does not arise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import InversionConfig
from .krylov import SolveReport, solve
from .levels import LevelSet
from .operator import FrozenState, PiecewiseOperator
from .passab import DiagnosedState
from .passc import BalancedInversion, rebuild_boundary_levels
from .precond import SeparablePreconditioner
from .qmin import FloorReport, floor_pv
from .sphere import SphereOps
from .vertical import VerticalOperator, streamfunction_ghost


@dataclass
class Piece:
    """One piece of the decomposition.

    Attributes:
        name: Level label -- a pressure in hPa for an interior piece, and the
            bottom and top pressures for the two boundary-temperature pieces,
            matching the naming the batch outputs use.
        levels: 1-based indices of the levels this piece draws its source from,
            counted upward from the bottom, so level 1 is the surface.
        psi_spec, phi_spec: The induced perturbation, on all levels.
        report: Convergence of its linear solve.
    """

    name: str
    levels: list[int]
    psi_spec: np.ndarray
    phi_spec: np.ndarray
    report: SolveReport


@dataclass
class PiecewiseResult:
    """Everything one event's inversion produced."""

    pieces: dict[str, Piece]
    total_psi_spec: np.ndarray
    total_phi_spec: np.ndarray
    psi_perturbation: np.ndarray
    phi_perturbation: np.ndarray
    floor_event: FloorReport
    floor_mean: FloorReport
    clamp_worst: float
    newton_steps: int
    diagnostics: dict = field(default_factory=dict)

    def piece_names(self) -> list[str]:
        return list(self.pieces)

    def summed_psi(self) -> np.ndarray:
        return sum(p.psi_spec for p in self.pieces.values())


def default_pieces(levels: LevelSet) -> dict[str, list[int]]:
    """One piece per source: each interior level, plus the two boundaries.

    Keys are pressures in hPa, so ``"1000"`` is the surface potential-temperature
    piece, ``"850"`` to ``"200"`` are the interior potential-vorticity levels and
    ``"100"`` is the top boundary -- the labelling downstream consumers read the
    per-level outputs by.
    """
    return {f"{int(p)}": [i + 1] for i, p in enumerate(levels.p_hpa)}


def invert_pieces(
    ops: SphereOps,
    levels: LevelSet,
    mean: DiagnosedState,
    event: DiagnosedState,
    cfg: InversionConfig | None = None,
    pieces: dict[str, list[int]] | None = None,
    qp_overrides: dict[str, np.ndarray] | None = None,
    th_overrides: dict[str, tuple[np.ndarray | None, np.ndarray | None]] | None = None,
) -> PiecewiseResult:
    """Decompose an event's balanced perturbation into per-source pieces.

    Args:
        ops: Operators on the Gaussian solver grid.
        levels: Vertical level set.
        mean: Diagnosed mean state (the climatology).  It enters as observed and
            is not itself balanced first, so the perturbation is measured against
            the state the data held rather than against a second inversion's
            answer.
        event: Diagnosed event state.
        cfg: Solver configuration.
        pieces: Mapping of name to 1-based level indices; defaults to one piece
            per source.
        qp_overrides: Optional per-piece replacement for that piece's interior
            potential-vorticity anomaly, on the solver grid.  Additive splits
            (planetary against eddy, say) go through here rather than through the
            level list, because the pass is linear in its source and so additive
            parts sum exactly, while a multiplicative mask does not.
        th_overrides: The same for the two boundary temperatures, as
            ``(bottom, top)`` with ``None`` for a boundary the piece does not
            carry.  A scale split of an upper piece has to divide the top boundary
            temperature along with the interior potential vorticity, or the two
            parts no longer sum to the piece they came from.

    Returns:
        A :class:`PiecewiseResult`.
    """
    cfg = cfg or InversionConfig()
    vert = VerticalOperator(levels)
    pieces = pieces or default_pieces(levels)
    qp_overrides = qp_overrides or {}
    th_overrides = th_overrides or {}
    weights = ops.grid.weights

    # Floor the two states separately before differencing: a floor applied to the
    # difference would let a clamp on one state leak into the other's source.
    q_event, floor_event = floor_pv(
        event.q_hat, levels, weights, cfg.pv_floor.qmin_pieces, cfg.clamps.mode
    )
    q_mean, floor_mean = floor_pv(
        mean.q_hat, levels, weights, cfg.pv_floor.qmin_pieces, cfg.clamps.mode
    )
    q_anom = q_event - q_mean
    theta_bot_anom = event.theta_bot - mean.theta_bot
    theta_top_anom = event.theta_top - mean.theta_top

    # The total balanced state, and the perturbation about the observed mean.
    balancer = BalancedInversion(
        ops,
        levels,
        clamps=cfg.clamps,
        mirror=cfg.mirror,
        krylov=cfg.krylov,
        newton=cfg.newton,
    )
    q_total, _ = floor_pv(
        event.q_hat, levels, weights, cfg.pv_floor.qmin_total, cfg.clamps.mode
    )
    psi_bal, phi_bal, newton = balancer.solve(
        event.psi_spec, event.phi_spec, q_total, event.theta_bot, event.theta_top
    )
    psi_pert = psi_bal - mean.psi_spec
    phi_pert = phi_bal - mean.phi_spec

    # One frozen operator for every piece: the mean plus half the perturbation,
    # which makes the linearisation exact and the pieces additive.  Its two
    # boundary levels are set to the ghost values the operator assumes for the
    # midpoint temperature: the balanced state carries those already, the
    # observed mean carries the data's own 1000 and 100 hPa fields, and a
    # reference mixing the two conventions makes the stability and cross-term
    # coefficients at the first and last interior levels the tangent of neither.
    psi_ref = mean.psi_spec + 0.5 * psi_pert
    phi_ref = mean.phi_spec + 0.5 * phi_pert
    psi_ref, phi_ref = rebuild_boundary_levels(
        ops,
        levels,
        psi_ref.copy(),
        phi_ref.copy(),
        mean.theta_bot + 0.5 * theta_bot_anom,
        mean.theta_top + 0.5 * theta_top_anom,
        balancer.frozen_f(),
    )
    # The pieces take the limiter the total inversion converged with, so their
    # operator is the tangent of the same regularised system at its midpoint.
    frozen = FrozenState(
        ops,
        levels,
        psi_ref,
        phi_ref,
        clamps=cfg.clamps,
        mirror=cfg.mirror,
        deformation_limit=newton.deformation_limit,
    )
    op = PiecewiseOperator(frozen)
    pre = SeparablePreconditioner(op)

    interior = levels.interior
    nlev = levels.nlev
    zeros_grid = np.zeros((ops.grid.nlat, ops.grid.nlon))
    out: dict[str, Piece] = {}

    for name, level_list in pieces.items():
        source = np.zeros_like(q_anom)
        theta_b = None
        theta_t = None
        for one_based in level_list:
            index = one_based - 1
            if index == 0:
                theta_b = theta_bot_anom
            elif index == nlev - 1:
                theta_t = theta_top_anom
            else:
                position = int(np.where(interior == index)[0][0])
                source[position] = q_anom[position]
        if name in qp_overrides:
            source = np.asarray(qp_overrides[name], dtype=float)
            if source.shape != q_anom.shape:
                raise ValueError(
                    f"qp_overrides[{name!r}] has shape {source.shape}, expected "
                    f"{q_anom.shape}"
                )
        if name in th_overrides:
            theta_b, theta_t = th_overrides[name]

        rhs = op.rhs(source, theta_b, theta_t)
        vec, report = solve(op.matvec, rhs, preconditioner=pre.apply, cfg=cfg.krylov)
        phi_int, psi_int = op.unpack_state(vec)
        tb = theta_b if theta_b is not None else zeros_grid
        tt = theta_t if theta_t is not None else zeros_grid
        # The streamfunction's boundary levels take the thermal-wind ghost, the
        # temperature over f with its area mean removed -- the same relation the
        # operator's cross terms and the balanced total state use.  Handing it
        # the geopotential's ghost instead adds a few hundred m^2 s^-2 to a
        # field of order 1e7 m^2 s^-1: no error is raised, and the delivered
        # 1000 and 100 hPa winds of every piece are then those of the level
        # next to them.
        out[name] = Piece(
            name=name,
            levels=list(level_list),
            psi_spec=vert.extend(
                psi_int,
                ops.analyze(streamfunction_ghost(tb, frozen.f_grid, weights)),
                ops.analyze(streamfunction_ghost(tt, frozen.f_grid, weights)),
            ),
            phi_spec=vert.extend(phi_int, ops.analyze(tb), ops.analyze(tt)),
            report=report,
        )

    return PiecewiseResult(
        pieces=out,
        total_psi_spec=psi_bal,
        total_phi_spec=phi_bal,
        psi_perturbation=psi_pert,
        phi_perturbation=phi_pert,
        floor_event=floor_event,
        floor_mean=floor_mean,
        clamp_worst=frozen.report.worst(),
        newton_steps=newton.steps,
        diagnostics={
            "newton_converged": newton.converged,
            # How far short a run that hit the step cap actually stopped.  A
            # boolean cannot distinguish an iteration asymptoting a hair above the
            # tolerance from one stalled an order of magnitude above it, and the
            # difference decides whether the result is usable.
            "newton_final_increment_m": (
                newton.increments[-1] / 9.81 if newton.increments else float("nan")
            ),
            "newton_residuals": list(newton.residuals),
            # Whether every inner linear solve met its tolerance.  When it did
            # not, the Newton steps were taken in approximate directions, and a
            # stalled outer iteration is a statement about the preconditioner
            # rather than about the nonlinearity.
            "inner_solves_converged": bool(all(newton.linear_converged)),
            "inner_solves_unconverged": int(
                sum(1 for c in newton.linear_converged if not c)
            ),
            "newton_increments": list(newton.increments),
            "newton_final_norms": dict(newton.final_norms),
            "newton_deformation_fraction": list(newton.deformation_fraction),
            "newton_limiter_refreshes": int(newton.limiter_refreshes),
            "newton_final_nonelliptic_fraction": list(newton.final_nonelliptic_fraction),
            "piece_deformation_fraction": [float(v) for v in frozen.deform_fraction],
            "linear_iterations": {n: p.report.iterations for n, p in out.items()},
        },
    )


def all_sources_inversion(
    ops: SphereOps,
    levels: LevelSet,
    mean: DiagnosedState,
    event: DiagnosedState,
    cfg: InversionConfig | None = None,
) -> tuple[np.ndarray, SolveReport]:
    """Invert every source at once, for the closure check.

    The sum of the pieces must reproduce this exactly -- not the raw difference
    between the balanced and mean states, which also contains whatever the balance
    equations cannot represent.
    """
    cfg = cfg or InversionConfig()
    result = invert_pieces(
        ops, levels, mean, event, cfg=cfg, pieces={"all": list(range(1, levels.nlev + 1))}
    )
    piece = result.pieces["all"]
    return piece.psi_spec, piece.report
