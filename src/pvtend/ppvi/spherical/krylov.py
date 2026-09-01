"""Krylov driver for the coupled linear system, with iteration telemetry.

The counts matter as much as the answer: the preconditioner freezes the two
coefficients at their level means, so how far the true coefficients wander from
those means is what sets the iteration count.  Recording it per solve is how a
degradation shows up as a number rather than as a slow batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres

from .config import KrylovConfig


@dataclass
class SolveReport:
    """What one linear solve cost and how well it converged."""

    iterations: int
    residual: float
    converged: bool
    history: list[float] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - human-facing
        state = "converged" if self.converged else "NOT converged"
        return f"{state} in {self.iterations} iterations, relative residual {self.residual:.2e}"


def solve(
    matvec,
    rhs: np.ndarray,
    preconditioner=None,
    cfg: KrylovConfig | None = None,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, SolveReport]:
    """Solve ``M x = b`` and report the iteration history.

    Args:
        matvec: Callable applying the operator to a flat real vector.
        rhs: Right-hand side, flat and real.
        preconditioner: Callable approximating the inverse, or ``None``.
        cfg: Stopping rules.
        x0: Optional initial guess.

    Returns:
        The solution and a :class:`SolveReport`.  A solve that hits the iteration
        cap returns its best iterate with ``converged`` false rather than raising,
        so a batch can record the failure per event instead of dying.
    """
    cfg = cfg or KrylovConfig()
    n = rhs.size
    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    pre = (
        LinearOperator((n, n), matvec=preconditioner, dtype=np.float64)
        if preconditioner is not None
        else None
    )

    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        return np.zeros_like(rhs), SolveReport(0, 0.0, True, [0.0])

    history: list[float] = []

    def track(residual):
        # GMRES reports the preconditioned residual norm, BiCGStab the current
        # iterate; normalise whichever arrives so the history is comparable.
        value = residual if np.isscalar(residual) else np.linalg.norm(rhs - matvec(residual))
        history.append(float(value) / rhs_norm)

    if cfg.method == "gmres":
        # SciPy counts restart cycles, not iterations, so a plain maxiter would
        # buy `restart` times more work than asked for -- minutes instead of
        # seconds on an unpreconditioned solve.  Convert to cycles here so
        # `maxiter` means what it says: a budget of operator applications.
        cycles = max(1, -(-cfg.maxiter // cfg.restart))
        x, info = gmres(
            op,
            rhs,
            restart=cfg.restart,
            maxiter=cycles,
            callback_type="pr_norm",
            rtol=cfg.rtol,
            M=pre,
            x0=x0,
            callback=track,
        )
    else:
        x, info = bicgstab(
            op, rhs, rtol=cfg.rtol, maxiter=cfg.maxiter, M=pre, x0=x0, callback=track
        )

    residual = float(np.linalg.norm(rhs - matvec(x)) / rhs_norm)
    return x, SolveReport(
        iterations=len(history),
        residual=residual,
        converged=bool(info == 0) and residual <= max(cfg.rtol * 10, 1e-10),
        history=history,
    )
