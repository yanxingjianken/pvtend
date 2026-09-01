"""Invariant horizontal operators for the balance and potential-vorticity terms.

Every term of the Davis & Emanuel system is assembled here in a form that carries
no explicit metric factor, because on a sphere the metric terms are exactly what
gets dropped by accident.  Two devices do the work:

*The nonlinear balance term is written as a divergence.*  Taking the divergence
of the vector-invariant momentum equation for a nondivergent wind
``V = k x grad psi`` and setting the divergence tendency to zero gives, exactly
and on any surface,

``lap(Phi) = div((f + zeta) grad psi) - (1/2) lap(|grad psi|^2)``.

That is the form used here.  It needs only gradients, a divergence and a
Laplacian -- all spectral -- so the ``tan(lat)`` Christoffel terms live inside the
transforms and cannot be forgotten.

It is *not* algebraically identical to the textbook
``div(f grad psi) + 2(psi_xx psi_yy - psi_xy^2)``.  On a curved surface the
Laplacian does not commute with the gradient, and the Bochner identity leaves a
curvature term:

``div(zeta grad psi) - (1/2) lap(|grad psi|^2) = 2 det(Hess psi) - |grad psi|^2 / a^2``.

The Cartesian form is therefore the plane approximation of the balance equation.
The difference is of order ``|V|^2/a^2``, a couple of percent of the balance
terms in a midlatitude jet, so it is neither negligible nor hidden: the
deformation identity ``4 det(Hess psi) = zeta^2 - D1^2 - D2^2`` is kept as an
independent check on the production path, together with that curvature term.

*Cross terms are dot products of gradients.*  ``grad(a_Pi) . grad(b_Pi)`` is an
invariant scalar, evaluated as ``ax bx + ay by`` from spectral gradients of the
vertical differences.

All of this runs on the pole-free Gaussian solver grid; see :mod:`pvinv_sph.sht`.
"""
from __future__ import annotations

import numpy as np

from .sht import SHT


class SphereOps:
    """Horizontal calculus on one Gaussian solver grid.

    Args:
        sht: Transform bound to a Gaussian grid (exact quadrature, no pole row).
    """

    def __init__(self, sht: SHT):
        if sht.grid.weights is None:
            raise ValueError(
                "SphereOps needs a Gaussian grid: products and derivatives are "
                "collocated there, and its quadrature is what makes the "
                "divergence exact"
            )
        self.sht = sht

    # -- plumbing -----------------------------------------------------------

    @property
    def grid(self):
        return self.sht.grid

    def analyze(self, field: np.ndarray) -> np.ndarray:
        return self.sht.analyze(field)

    def synth(self, spec: np.ndarray) -> np.ndarray:
        return self.sht.synthesize(spec)

    def grad(self, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Eastward and northward derivatives of a scalar, on the grid."""
        return self.sht.gradient(spec)

    def lap(self, spec: np.ndarray) -> np.ndarray:
        return self.sht.laplacian_spec(spec)

    def inv_lap(self, spec: np.ndarray) -> np.ndarray:
        return self.sht.invert_laplacian_spec(spec)

    def div(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        return self.sht.divergence(fx, fy)

    # -- composite terms ----------------------------------------------------

    def dot_grad(self, spec_a: np.ndarray, spec_b: np.ndarray) -> np.ndarray:
        """``grad(a) . grad(b)`` on the grid -- an invariant, so pole-regular."""
        ax, ay = self.grad(spec_a)
        bx, by = self.grad(spec_b)
        return ax * bx + ay * by

    def div_c_grad(self, c_grid: np.ndarray, spec: np.ndarray) -> np.ndarray:
        """Spectrum of ``div(c grad a)`` for a gridded coefficient ``c``."""
        ax, ay = self.grad(spec)
        return self.div(c_grid * ax, c_grid * ay)

    def balance_nonlinear(
        self, psi_spec: np.ndarray, f_grid: np.ndarray
    ) -> np.ndarray:
        """Spectrum of the right-hand side of the nonlinear balance equation.

        ``N(psi) = div((f + zeta) grad psi) - (1/2) lap(|grad psi|^2)`` -- the
        exact spherical form, which exceeds the Cartesian
        ``div(f grad psi) + 2 det(Hess psi)`` by the curvature term
        ``-|grad psi|^2 / a^2``.
        """
        zeta = self.synth(self.lap(psi_spec))
        px, py = self.grad(psi_spec)
        first = self.div((f_grid + zeta) * px, (f_grid + zeta) * py)
        second = self.lap(self.analyze(px * px + py * py))
        return first - 0.5 * second

    def balance_nonlinear_tangent(
        self,
        psi_ref_spec: np.ndarray,
        psi_prime_spec: np.ndarray,
        f_grid: np.ndarray,
        zeta_ref_grid: np.ndarray | None = None,
    ) -> np.ndarray:
        """Directional derivative ``DN(psi_ref)[psi_prime]``.

        ``N`` is quadratic in ``psi`` apart from the exactly linear Coriolis term,
        so evaluating this at ``psi_ref = mean + half the perturbation`` reproduces
        ``N(mean + perturbation) - N(mean)`` exactly.  That midpoint choice is why
        pieces sharing this frozen operator sum to the all-sources solution rather
        than merely approximating it.

        Args:
            psi_ref_spec: The frozen reference streamfunction.
            psi_prime_spec: The perturbation the derivative acts on.
            f_grid: Coriolis parameter on the grid.
            zeta_ref_grid: Optional precomputed ``lap(psi_ref)`` on the grid.
        """
        if zeta_ref_grid is None:
            zeta_ref_grid = self.synth(self.lap(psi_ref_spec))
        zeta_p = self.synth(self.lap(psi_prime_spec))
        rx, ry = self.grad(psi_ref_spec)
        px, py = self.grad(psi_prime_spec)

        term_a = self.div((f_grid + zeta_ref_grid) * px, (f_grid + zeta_ref_grid) * py)
        term_b = self.div(zeta_p * rx, zeta_p * ry)
        term_c = self.lap(self.analyze(rx * px + ry * py))
        return term_a + term_b - term_c

    # -- diagnostics --------------------------------------------------------

    def deformation(self, psi_spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stretching and shearing deformation of the rotational wind.

        From the strain-rate tensor in orthogonal curvilinear coordinates
        (scale factors ``a cos(lat)`` and ``a``),

        ``D1 = u_x - v_y - v tan(lat)/a``   (stretching, ``e_11 - e_22``)
        ``D2 = v_x + u_y + u tan(lat)/a``   (shearing, ``2 e_12``)

        which pair with the familiar ``div = u_x + v_y - v tan(lat)/a`` and
        ``zeta = v_x - u_y + u tan(lat)/a``.  The ``tan(lat)`` terms come from
        differentiating a *vector* rather than a scalar; dropping them, as a flat
        finite difference does, leaves solid-body rotation with a spurious
        deformation.  They grow without bound at the poles, so this diagnostic
        degrades there -- which is exactly why :meth:`balance_nonlinear` is
        written as divergences instead and needs no metric term at all.
        """
        px, py = self.grad(psi_spec)
        u, v = -py, px
        ux, uy = self.grad(self.analyze(u))
        vx, vy = self.grad(self.analyze(v))
        metric = np.tan(np.radians(self.grid.lat))[:, None] / self.sht.radius
        return ux - vy - v * metric, vx + uy + u * metric

    def hessian_determinant_from_deformation(self, psi_spec: np.ndarray) -> np.ndarray:
        """``2 det(Hess psi)`` via ``(zeta^2 - D1^2 - D2^2)/2``, on the grid."""
        zeta = self.synth(self.lap(psi_spec))
        d1, d2 = self.deformation(psi_spec)
        return 0.5 * (zeta * zeta - d1 * d1 - d2 * d2)

    def curvature_term(self, psi_spec: np.ndarray) -> np.ndarray:
        """``|grad psi|^2 / a^2``: what separates the spherical and plane forms.

        Subtracting this from ``div(f grad psi) + 2 det(Hess psi)`` reproduces
        :meth:`balance_nonlinear`.  Exposed as a diagnostic rather than left
        implicit, because it is the whole of the difference between the spherical
        balance equation and its plane approximation, and its size on a given flow
        is what says whether that difference matters there.
        """
        px, py = self.grad(psi_spec)
        return (px * px + py * py) / self.sht.radius**2

    def rotational_wind(self, psi_spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(u, v) = (-d psi/dy, d psi/dx)`` on the grid."""
        px, py = self.grad(psi_spec)
        return -py, px
