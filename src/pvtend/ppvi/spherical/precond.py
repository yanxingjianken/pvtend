"""Separable preconditioner: one small vertical solve per spherical harmonic.

Freezing the two coefficients at their level means and dropping the cross and
deformation terms leaves a system that spherical harmonics diagonalise
horizontally, so with ``lam = -n(n+1)/a^2`` each coefficient obeys

``lam Phi - A lam psi = r1``
``A (T Phi) + S lam psi = r2``

where ``T`` is the folded vertical second difference.  Eliminating ``psi``
exactly gives a single tridiagonal problem in the column,

``diag(A^2/S) T Phi + lam Phi = (A/S) r2 + r1``,

which is negative definite for every ``n >= 1`` because ``T`` is negative
semi-definite and ``lam`` is negative -- so there is no wavenumber at which the
preconditioner degrades.  This is the exact block factorisation of the
approximate operator, which costs the same as a block-diagonal approximation and
is strictly stronger.

``n = 0`` is the gauge slot: the balance row there was replaced by the condition
that each level's mean streamfunction vanish, and the geopotential column is made
solvable by the operator's rank-one term.
"""
from __future__ import annotations

import numpy as np

from .operator import PiecewiseOperator


class SeparablePreconditioner:
    """Approximate inverse of :class:`~pvinv_sph.operator.PiecewiseOperator`."""

    def __init__(self, op: PiecewiseOperator):
        self.op = op
        self.packer = op.packer
        vert = op.vert
        nint = vert.nint
        lmax = op.lmax

        a = op.frozen.avo_mean.astype(float)
        s = op.frozen.stb_mean.astype(float)
        if np.any(a <= 0) or np.any(s <= 0):
            raise ValueError(
                "level-mean absolute vorticity and static stability must be "
                f"positive for the preconditioner; got A={a}, S={s}"
            )
        self.a_level = a
        self.s_level = s
        # The operator hands over scaled rows and a scaled streamfunction column;
        # these undo exactly that, so the tridiagonal below is the same matrix as
        # for the unscaled system.
        self.f0 = op.F0
        self._row_ratio = a / self.f0
        self._row_unscale = s / self.f0

        t_full = np.zeros((nint, nint))
        lower, diag, upper = vert.tridiagonal(np.ones(nint))
        for k in range(nint):
            t_full[k, k] = diag[k]
            if k > 0:
                t_full[k, k - 1] = lower[k]
            if k < nint - 1:
                t_full[k, k + 1] = upper[k]
        self._t = t_full

        bar = a * a / s
        eig = op.ops.sht.laplacian_eigen  # length lmax+1, entry 0 is zero
        self._inv = np.empty((lmax + 1, nint, nint))
        for n in range(1, lmax + 1):
            self._inv[n] = np.linalg.inv(bar[:, None] * t_full + eig[n] * np.eye(nint))
        gauge = op.gauge_scale / nint
        self._inv[0] = np.linalg.inv(
            a[:, None] * t_full + gauge * np.ones((nint, nint))
        )
        self._eig = eig

    def apply(self, vec: np.ndarray) -> np.ndarray:
        r1, r2 = self.packer.unpack(vec)
        phi = np.zeros_like(r1)
        psi = np.zeros_like(r2)

        # n = 0: the gauge rows.  The streamfunction mean is read straight off the
        # balance row, and the geopotential column uses the rank-one-corrected
        # operator, so nothing here is left to the Krylov iteration to discover.
        phi[:, 0, 0] = self._inv[0] @ (self._row_unscale * r2[:, 0, 0])
        psi[:, 0, 0] = self.f0 * r1[:, 0, 0]

        for n in range(1, self.op.lmax + 1):
            r1n = r1[:, : n + 1, n]
            r2n = r2[:, : n + 1, n]
            rhs = self._row_ratio[:, None] * r2n + r1n
            phin = self._inv[n] @ rhs
            phi[:, : n + 1, n] = phin
            psi[:, : n + 1, n] = self.f0 * (
                phin / self.a_level[:, None]
                - r1n / (self.a_level[:, None] * self._eig[n])
            )
        return self.packer.pack(phi, psi)
