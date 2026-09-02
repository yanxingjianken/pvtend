"""The vertical half of the operator: Exner second differences and their ghosts.

Interior potential vorticity lives on levels ``1 .. NL-2``; the top and bottom
levels carry boundary potential temperature instead.  Hydrostatic balance turns
that boundary temperature into ghost values,

``Phi_0 = Phi_1 + theta_B (Pi_1 - Pi_0)``,
``Phi_{NL-1} = Phi_{NL-2} - theta_T (Pi_{NL-1} - Pi_{NL-2})``,

and substituting them into the end rows both folds a coupling weight onto the
diagonal and moves the temperature onto the right-hand side, divided by ``dpi2``.
Keeping the homogeneous folding and the source separate is what lets a piece be
driven by one boundary temperature alone -- the operator is shared, only the
source changes.

The same ghost rule applies to the vertical differences inside the cross terms,
which is easy to miss: a first difference at the first interior level reaches the
ghost too.
"""
from __future__ import annotations

import numpy as np

from .levels import LevelSet


def streamfunction_ghost(
    theta: np.ndarray, f: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """The boundary temperature as it acts on the streamfunction.

    Hydrostatic balance fixes the *geopotential*: ``dPhi/dPi = -theta``.  The
    streamfunction has no such relation of its own, so what its ghost level should
    be is a statement about the flow just outside the domain, and the defensible
    one is that the flow is balanced there.  To leading order ``Phi = f psi``, so

        dpsi/dPi = -(theta - <theta>) / f

    with the area mean removed.  Two things had to be got right here, and each was
    got wrong first:

    *The division by f.*  Without it the expression is not even dimensionally a
    streamfunction -- it adds a geopotential to one.  Its magnitude then happens to
    be small enough to look like no boundary condition at all, which is why leaving
    it out passes unnoticed.

    *The area mean.*  The horizontally uniform part of the temperature carries no
    flow at all, so it must not reach the streamfunction: with the full field
    divided by f the ghost is of order 1e8 m^2/s, larger than the streamfunction
    itself.  The subtraction has to be made in the total and the perturbation
    inversion alike.  Making it in one and not the other leaves the two solving
    different boundary conditions, and the piece sum then falls short of the total
    for a reason that presents as a solver fault rather than as an inconsistent
    ghost.

    This lives in one place because three parts of the solver need the same
    convention -- the operator's right-hand side, the nonlinear residual, and the
    reconstruction of the boundary levels -- and any two of them disagreeing leaves
    a balanced state failing its own equations.

    ``theta`` may carry leading axes; each of its grids then gets its own area
    mean, which is what a column of ghosts needs.
    """
    zonal = theta.mean(axis=-1)
    mean = np.sum(weights * zonal, axis=-1) / np.sum(weights)
    return (theta - mean[..., None, None]) / f


class VerticalOperator:
    """Vertical operators on the interior levels of one :class:`LevelSet`."""

    def __init__(self, levels: LevelSet):
        self.levels = levels
        self.interior = levels.interior
        self.nint = int(self.interior.size)
        if self.nint < 2:
            raise ValueError("need at least two interior levels")

        k = self.interior
        self._bb = levels.bb[k].copy()
        self._bh = levels.bh[k].copy()
        self._bl = levels.bl[k].copy()
        self._dpi2 = levels.dpi2[k].copy()
        #: Diagonal after folding the hydrostatic ghosts into the end rows.
        self.diag = self._bb.copy()
        self.diag[0] += self._bl[0]
        self.diag[-1] += self._bh[-1]
        #: Sub- and super-diagonals of the folded second-difference operator.
        self.lower = self._bl[1:].copy()
        self.upper = self._bh[:-1].copy()

    # -- second derivative --------------------------------------------------

    def d2(self, field: np.ndarray) -> np.ndarray:
        """Folded ``d2/dPi2`` on interior levels; the homogeneous part only.

        Args:
            field: Array with the interior level axis first.
        """
        out = self.diag.reshape((-1,) + (1,) * (field.ndim - 1)) * field
        out[:-1] += self.upper.reshape((-1,) + (1,) * (field.ndim - 1)) * field[1:]
        out[1:] += self.lower.reshape((-1,) + (1,) * (field.ndim - 1)) * field[:-1]
        return out

    def theta_forcing(self, theta_bot: np.ndarray, theta_top: np.ndarray) -> np.ndarray:
        """Boundary-temperature contribution to ``d2/dPi2``, on interior levels.

        The sign follows the ghost substitution: a warm lower boundary raises the
        static-stability term at the level above it, a warm upper boundary lowers
        it at the level below.
        """
        shape = (self.nint,) + np.shape(theta_bot)
        out = np.zeros(shape, dtype=np.result_type(theta_bot, theta_top, np.float64))
        out[0] = theta_bot / self._dpi2[0]
        out[-1] = -theta_top / self._dpi2[-1]
        return out

    def tridiagonal(self, coeff: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``coeff * d2/dPi2`` as banded arrays, for the preconditioner.

        Args:
            coeff: One value per interior level.

        Returns:
            ``(lower, diag, upper)``, each an array over interior levels with
            ``lower[0]`` and ``upper[-1]`` unused and set to zero.
        """
        coeff = np.asarray(coeff, dtype=float).reshape(self.nint)
        lower = np.zeros(self.nint)
        upper = np.zeros(self.nint)
        lower[1:] = coeff[1:] * self.lower
        upper[:-1] = coeff[:-1] * self.upper
        return lower, coeff * self.diag, upper

    # -- first derivative ---------------------------------------------------

    def d_dpi(
        self,
        field: np.ndarray,
        theta_bot: np.ndarray | None = None,
        theta_top: np.ndarray | None = None,
    ) -> np.ndarray:
        """Centred ``d/dPi`` on interior levels, using the hydrostatic ghosts.

        With no boundary temperature supplied the ghosts are homogeneous, which is
        the right choice for a piece not driven by that boundary.
        """
        below = np.empty_like(field)
        above = np.empty_like(field)
        below[1:] = field[:-1]
        above[:-1] = field[1:]
        pi = self.levels.pi
        k0, k1 = self.interior[0], self.interior[-1]
        below[0] = field[0]
        above[-1] = field[-1]
        if theta_bot is not None:
            below[0] = below[0] + theta_bot * (pi[k0] - pi[k0 - 1])
        if theta_top is not None:
            above[-1] = above[-1] - theta_top * (pi[k1 + 1] - pi[k1])
        scale = (2.0 * self._dpi2).reshape((-1,) + (1,) * (field.ndim - 1))
        return (above - below) / scale

    # -- hydrostatic extension ---------------------------------------------

    def extend(
        self, field: np.ndarray, theta_bot: np.ndarray, theta_top: np.ndarray
    ) -> np.ndarray:
        """Add the two boundary levels back, hydrostatically.

        Args:
            field: Interior levels, level axis first.
            theta_bot, theta_top: Boundary potential temperature anomalies.

        Returns:
            Array over all ``NL`` levels.
        """
        pi = self.levels.pi
        nlev = self.levels.nlev
        out = np.empty((nlev,) + field.shape[1:], dtype=field.dtype)
        out[1:-1] = field
        out[0] = field[0] + theta_bot * (pi[1] - pi[0])
        out[-1] = field[-1] - theta_top * (pi[nlev - 1] - pi[nlev - 2])
        return out
