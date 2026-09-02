"""The coupled linearised operator of the piecewise inversion.

The perturbation form of the Davis & Emanuel system is two equations for two
fields, on the interior levels:

``E1``  ``lap(Phi') - DN(psi~)[psi'] = 0``
``E2``  ``AVO Phi'_PiPi + STB lap(psi') - grad(psi~_Pi).grad(Phi'_Pi)
        - grad(Phi~_Pi).grad(psi'_Pi) = q_hat'``

with ``AVO = f* + lap(psi~)`` and ``STB = Phi~_PiPi``.  Both nonlinearities are
quadratic or bilinear, so freezing the coefficients at the mean plus half the
perturbation makes the linearisation *exact* rather than approximate, which is
the reason pieces sum to the all-sources solution.

The two equations go to the Krylov solver together, as one system, so the
vertical coupling is implicit in both the operator and its preconditioner.
Solving them by alternation instead -- sweeping one field with the other held
fixed and lagging the vertical coupling between sweeps -- does not work here: the
vertical stencil has vanishing row sums, so nothing damps a lagged column and the
two fields drift together.

Neither is one field eliminated into the other at the discrete level, which is
the usual way to accelerate such an alternation.  That elimination buries the
discrete horizontal diagonal inside the coefficients it produces, and
substituting the converged balance equation back into them collapses every such
term and returns exactly ``E1``/``E2`` above.  The coupled solve supplies the
same coupling without the detour, so the equations stay in the form they were
derived in and can be read against the physics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ClampConfig, MirrorConfig
from .levels import LevelSet
from .mirror import blend_weight, coriolis_star
from .sphere import SphereOps
from .state import SpectralPacker
from .vertical import VerticalOperator, streamfunction_ghost


@dataclass
class ClampReport:
    """Where ellipticity had to be enforced, as area fractions per level.

    Published rather than logged away: the clamps activate under strong
    anticyclones and blocks, which is exactly where the science is, so a run
    should always be able to show how much of its domain was propped up.
    """

    avo_fraction: np.ndarray
    stb_fraction: np.ndarray

    def worst(self) -> float:
        return float(max(self.avo_fraction.max(), self.stb_fraction.max()))


class FrozenState:
    """Coefficients of the linearised operator, evaluated once per inversion.

    Args:
        ops: Horizontal operators on the Gaussian solver grid.
        levels: Vertical level set.
        psi_spec: Reference streamfunction spectra on **all** levels.
        phi_spec: Reference geopotential spectra on all levels.
        clamps: Ellipticity floors.
        mirror: Equatorial blending of the coefficients.
        deformation_limit: Optional limiter for the deformation terms, one
            factor per interior level on the grid, to be used instead of the
            one this state would compute for itself.  The nonlinear pass hands
            the Jacobian's limiter to the half-state operator that evaluates
            its residual, so the two describe the same regularised system.
    """

    def __init__(
        self,
        ops: SphereOps,
        levels: LevelSet,
        psi_spec: np.ndarray,
        phi_spec: np.ndarray,
        clamps: ClampConfig | None = None,
        mirror: MirrorConfig | None = None,
        deformation_limit: np.ndarray | None = None,
    ):
        self.ops = ops
        self.levels = levels
        self.vert = VerticalOperator(levels)
        self.clamps = clamps or ClampConfig()
        self.mirror = mirror or MirrorConfig()

        nlev = levels.nlev
        if psi_spec.shape[0] != nlev or phi_spec.shape[0] != nlev:
            raise ValueError(
                f"reference state must cover all {nlev} levels, got "
                f"{psi_spec.shape[0]} and {phi_spec.shape[0]}"
            )
        self.psi_spec = psi_spec
        self.phi_spec = phi_spec
        interior = levels.interior
        lat = ops.grid.lat

        self.f_grid = np.broadcast_to(
            coriolis_star(lat, self.mirror.f_floor_deg)[:, None],
            (ops.grid.nlat, ops.grid.nlon),
        ).copy()

        zeta = ops.synth(ops.lap(psi_spec[interior]))
        phi_grid = ops.synth(phi_spec)
        below, here, above = (
            weight[interior].reshape(-1, 1, 1)
            for weight in (levels.bl, levels.bb, levels.bh)
        )
        stb = (
            below * phi_grid[interior - 1]
            + here * phi_grid[interior]
            + above * phi_grid[interior + 1]
        )

        # The equatorial taper.  Only the coefficients are tapered, never the
        # sources or the solution, so the operator stays shared between pieces.
        # The weight multiplies the *products* the quadratic terms are built
        # from, which keeps the tapered system an exact quadratic with a
        # symmetric bilinear form: the midpoint linearisation then holds with the
        # taper on, and the Jacobian is the derivative of the residual.  Tapering
        # the reference argument alone (the vorticity, or a streamfunction rebuilt
        # from it) does neither, and cost a factor of two per Newton step on every
        # event before it was found.
        weight = np.ones((ops.grid.nlat, 1))
        if self.mirror.blend:
            weight = blend_weight(
                lat, self.mirror.blend_south, self.mirror.blend_north
            ).reshape(-1, 1)
        self.weight = weight
        self.zeta_raw = zeta
        zeta = weight[None] * zeta
        self.stb_limit = self._area_mean(stb).reshape(-1, 1, 1)
        stb = self.stb_limit + weight[None] * (stb - self.stb_limit)
        self.psi_ref_spec = psi_spec[interior]

        avo = self.f_grid[None, :, :] + zeta
        self.avo, avo_hit = self._floor(avo, self.clamps.avo_min)
        self.stb, stb_hit = self._floor(stb, self.clamps.stb_min)
        self.report = ClampReport(avo_fraction=avo_hit, stb_fraction=stb_hit)
        self.zeta_ref = zeta

        # The balance row is elliptic only where the absolute vorticity exceeds
        # the deformation of the reference flow: the quadratic part of the
        # balance equation, polarised, is ``zeta_a zeta_b - D_a . D_b`` (less the
        # curvature term), so the symbol of the linearised row has eigenvalues
        # ``AVO -/+ w D``.  Where the deformation wins -- routinely on the flank
        # of a strong anticyclone, over about a sixth of the jet level -- the
        # linearised system is indefinite, no inner solver converges and Newton
        # walks into a fold.  The deformation part alone is therefore scaled by
        # ``s = min(1, (1 - margin) AVO / (w D))``, the counterpart of the
        # limited-area code's clamp on its balance-equation coefficient; the
        # vorticity part, which is what gradient-wind balance lives on, is kept.
        if deformation_limit is None:
            deform = self._deformation_magnitude(psi_spec[interior])
            margin = self.clamps.deformation_margin
            denom = weight[None] * deform
            allowed = (1.0 - margin) * self.avo
            with np.errstate(divide="ignore", invalid="ignore"):
                limit = np.where(denom > allowed, allowed / denom, 1.0)
            self.deform_limit = limit
        else:
            limit = np.asarray(deformation_limit, dtype=float)
            if limit.shape != zeta.shape:
                raise ValueError(
                    f"deformation_limit has shape {limit.shape}, expected {zeta.shape}"
                )
            self.deform_limit = limit
        self.deform_fraction = np.mean(self.deform_limit < 1.0, axis=(1, 2))

        # Cross-term coefficients: horizontal gradients of the vertical
        # derivatives of the reference state, tapered on the same band.
        dpsi = self._d_dpi_reference(psi_spec)
        dphi = self._d_dpi_reference(phi_spec)
        self.dpsi_x, self.dpsi_y = self.ops.grad(dpsi)
        self.dphi_x, self.dphi_y = self.ops.grad(dphi)
        for arr in (self.dpsi_x, self.dpsi_y, self.dphi_x, self.dphi_y):
            arr *= weight[None]

        # Level means for the separable preconditioner.
        self.avo_mean = self._area_mean(self.avo)
        self.stb_mean = self._area_mean(self.stb)

    # -- helpers ------------------------------------------------------------

    def _floor(self, field: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
        """Apply the ellipticity floor and report where it bit.

        In ``"parity"`` mode the rule is asymmetric: the test is against a tenth
        of the floor, and points that fail it are set to the floor itself, so a
        marginal point is nudged clear of the threshold rather than pinned to it.
        ``"clean"`` tests against the floor itself.

        Returns:
            The floored field, and the area fraction clamped on each level.
        """
        out = field.copy()
        threshold = 0.1 * floor if self.clamps.mode == "parity" else floor
        hit = out < threshold
        out[hit] = floor
        return out, np.mean(hit, axis=(1, 2))

    def _d_dpi_reference(self, spec: np.ndarray) -> np.ndarray:
        """Centred ``d/dPi`` of a reference field, which is known at every level."""
        interior = self.levels.interior
        two_dpi = 2.0 * self.levels.dpi2[interior]
        return (spec[interior + 1] - spec[interior - 1]) / two_dpi.reshape(
            (-1,) + (1,) * (spec.ndim - 1)
        )

    def _deformation_magnitude(self, spec: np.ndarray) -> np.ndarray:
        """``D`` of the reference flow, from the polarised balance identity.

        ``B(psi, psi) = zeta^2 - D^2 - 2 |grad psi|^2 / a^2`` with ``B`` the
        exact divergence form, so ``D`` follows from spectral operations alone
        and stays regular at the poles, where the strain-rate components -- which
        expand the wind components as scalars -- do not.  Truncation can push the
        square a little below zero; it is clipped.
        """
        ops = self.ops
        zeta = ops.synth(ops.lap(spec))
        px, py = ops.grad(spec)
        speed2 = px * px + py * py
        form = ops.synth(2.0 * ops.div(zeta * px, zeta * py) - ops.lap(ops.analyze(speed2)))
        d2 = zeta * zeta - form - 2.0 * speed2 / ops.sht.radius**2
        return np.sqrt(np.clip(d2, 0.0, None))

    def _area_mean(self, field: np.ndarray) -> np.ndarray:
        w = self.ops.grid.weights
        return np.einsum("...ij,i->...", field, w) / (w.sum() * field.shape[-1])


class PiecewiseOperator:
    """``M x = b`` for one frozen state; the source changes from piece to piece.

    The operator is built once per event and reused for every piece, which is
    what makes the pieces additive: they differ only in ``b``.
    """

    #: Reference Coriolis parameter used to put the streamfunction on the same
    #: numerical footing as the geopotential.  Geostrophy makes ``f0 psi`` and
    #: ``Phi`` comparable, which is all this is for.
    F0 = 1.0e-4

    def __init__(self, frozen: FrozenState):
        self.frozen = frozen
        self.ops = frozen.ops
        self.levels = frozen.levels
        self.vert = frozen.vert
        self.lmax = self.ops.sht.lmax
        self.nint = self.vert.nint
        self.packer = SpectralPacker(self.nint, self.lmax)
        #: Scale for the rank-one term that removes the constant-geopotential
        #: null direction; comparable to the operator's own diagonal.  Built
        #: from the Coriolis parameter alone so that it is the same number for
        #: every reference state: a scale that moved with the reference would
        #: make the gauge row bilinear in the state and the residual's Jacobian
        #: inexact in that one slot.
        self.gauge_scale = float(
            np.mean(np.abs(np.mean(frozen.f_grid) * self.vert.diag))
        )
        # The two equations differ by many orders of magnitude in units alone, so
        # a Krylov tolerance on the combined vector is met almost entirely by the
        # potential-vorticity row while the balance row stays far from converged
        # -- worth a third of a metre of geopotential height at every latitude.
        # Scaling the rows and the streamfunction column puts both blocks on a
        # Laplacian-like footing; it is a similarity transform, so the solution is
        # unchanged.
        self.row2_scale = self.F0 / frozen.stb_mean

    # -- pieces of the operator --------------------------------------------

    def _row1(self, phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Linearised balance equation, level by level (no vertical coupling).

        ``lap(Phi') - div(f* grad psi') - T[psi']`` with the tangent of the
        quadratic part built from its polarised form,

        ``T[psi'] = w zeta~ zeta' + w s (B(psi~, psi') - zeta~ zeta')``,

        where ``B(a, b) = div(zeta_a grad b) + div(zeta_b grad a)
        - lap(grad a . grad b)`` is the exact spherical bilinear form,
        ``B - zeta zeta`` its deformation-and-curvature part, ``w`` the
        equatorial weight and ``s`` the deformation limiter, both frozen.
        Every factor is symmetric in the reference and the perturbation, so the
        row is the exact tangent of a quadratic and the midpoint linearisation
        holds wherever the floors are quiet.
        """
        ops = self.ops
        frozen = self.frozen
        zeta_ref = frozen.avo - frozen.f_grid
        linear = ops.div_c_grad(frozen.f_grid, psi)
        zeta_p = ops.synth(ops.lap(psi))
        rx, ry = ops.grad(frozen.psi_ref_spec)
        px, py = ops.grad(psi)
        full = ops.synth(
            ops.div(frozen.zeta_raw * px, frozen.zeta_raw * py)
            + ops.div(zeta_p * rx, zeta_p * ry)
            - ops.lap(ops.analyze(rx * px + ry * py))
        )
        # ``zeta_ref`` already carries the equatorial weight (and the floor
        # where it bit); the deformation part takes the weight here.
        vorticity_part = zeta_ref * zeta_p
        deformation_part = full - frozen.zeta_raw * zeta_p
        tangent = ops.analyze(
            vorticity_part + frozen.weight * frozen.deform_limit * deformation_part
        )
        return ops.lap(phi) - linear - tangent

    def _row2(self, phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Linearised potential-vorticity equation, with the column coupled."""
        phi_grid = self.ops.synth(phi)
        zeta_grid = self.ops.synth(self.ops.lap(psi))
        d2phi = self.vert.d2(phi_grid)

        dphi_dpi = self.vert.d_dpi(phi)
        dpsi_dpi = self.vert.d_dpi(psi)

        px, py = self.ops.grad(dphi_dpi)
        sx, sy = self.ops.grad(dpsi_dpi)
        cross = (
            self.frozen.dpsi_x * px
            + self.frozen.dpsi_y * py
            + self.frozen.dphi_x * sx
            + self.frozen.dphi_y * sy
        )
        field = self.frozen.avo * d2phi + self.frozen.stb * zeta_grid - cross
        # The static stability is tapered towards its level mean, an affine
        # map of the reference; the symmetric bilinear form that goes with it
        # carries the perturbation's level-mean stratification against the
        # untapered reference vorticity where the taper acts.
        field += (
            (1.0 - self.frozen.weight)
            * self.frozen.zeta_raw
            * self.frozen._area_mean(d2phi).reshape(-1, 1, 1)
        )
        return self.ops.analyze(field)

    def apply(self, phi: np.ndarray, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply the operator, with both gauge conditions folded in.

        The balance equation is a sum of divergences, so its ``n = 0`` component
        vanishes identically for any input and carries no information; that slot
        is reused to pin each level's mean streamfunction, which is precisely the
        null direction it corresponds to.  A rank-one term on the
        potential-vorticity row removes the remaining constant-geopotential
        direction.
        """
        r1 = self._row1(phi, psi)
        r2 = self._row2(phi, psi)
        r1[:, 0, 0] = psi[:, 0, 0].real
        r2[:, 0, 0] = r2[:, 0, 0] + self.gauge_scale * np.mean(phi[:, 0, 0].real)
        return r1, r2

    # -- scaled representation used by the solver ---------------------------

    def pack_state(self, phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Pack a physical state into the solver's scaled unknowns."""
        return self.packer.pack(phi, psi * self.F0)

    def unpack_state(self, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Recover the physical ``(phi, psi)`` from the solver's unknowns."""
        phi, scaled_psi = self.packer.unpack(vec)
        return phi, scaled_psi / self.F0

    def pack_rows(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """Pack two equation rows with the row scaling applied."""
        return self.packer.pack(r1, r2 * self.row2_scale[:, None, None])

    def matvec(self, vec: np.ndarray) -> np.ndarray:
        phi, psi = self.unpack_state(vec)
        r1, r2 = self.apply(phi, psi)
        return self.pack_rows(r1, r2)

    # -- right-hand side ----------------------------------------------------

    def rhs(
        self,
        q_hat: np.ndarray,
        theta_bot: np.ndarray | None = None,
        theta_top: np.ndarray | None = None,
    ) -> np.ndarray:
        """Assemble ``b`` for one piece, in the solver's scaled rows.

        See :meth:`rhs_rows` for the unscaled equation rows.
        """
        return self.pack_rows(*self.rhs_rows(q_hat, theta_bot, theta_top))

    def rhs_rows(
        self,
        q_hat: np.ndarray,
        theta_bot: np.ndarray | None = None,
        theta_top: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble ``b`` for one piece.

        Args:
            q_hat: Interior potential-vorticity anomaly of this piece, on the
                solver grid, already scaled by :func:`pvinv_sph.levels.pv_rhs_scale`
                and shaped ``(nint, nlat, nlon)``.
            theta_bot, theta_top: Boundary potential-temperature anomalies on the
                solver grid, or ``None`` for a piece that does not include them.

        The boundary temperature enters twice: through the hydrostatic ghost in
        the static-stability term, and through the same ghost inside the vertical
        differences of the cross terms.  Missing the second is an easy and silent
        error, so both are assembled here from one call to the vertical operator.
        """
        zeros = np.zeros((self.ops.grid.nlat, self.ops.grid.nlon))
        # Both boundary temperatures pass through the spectrum first, so the
        # right-hand side sees exactly the temperature the hydrostatic ghosts
        # of a reference state are built from; a temperature with power beyond
        # the truncation would otherwise enter the two sides of the same
        # bilinear form differently.
        tb = zeros if theta_bot is None else self.ops.synth(self.ops.analyze(theta_bot))
        tt = zeros if theta_top is None else self.ops.synth(self.ops.analyze(theta_top))

        b2_grid = np.asarray(q_hat, dtype=float).copy()
        forcing = self.vert.theta_forcing(tb, tt)
        b2_grid -= self.frozen.avo * forcing
        forcing_mean = self.frozen._area_mean(forcing)
        b2_grid -= (
            (1.0 - self.frozen.weight)
            * self.frozen.zeta_raw
            * forcing_mean.reshape(-1, 1, 1)
        )

        # Ghost contribution to the cross terms: d/dPi of a zero field with the
        # boundary temperature attached is exactly the part that belongs on the
        # right.
        #
        # The two cross terms need two different ghosts.  The geopotential's comes
        # straight from hydrostatic balance, dPhi/dPi = -theta.  The
        # streamfunction's is that divided by the Coriolis parameter, because what
        # balance relates to the temperature is the geopotential, and psi carries
        # a factor of f against it.  Handing the same ghost to both terms reads as
        # the natural thing to do and is wrong: it gives the streamfunction a
        # boundary condition too large by f, everywhere -- a factor of about 1e4
        # in mid-latitudes.  The way to tell is dimensional, since psi is
        # m^2 s^-1 and Phi is m^2 s^-2 and one array cannot be both.  The division
        # is done before the gradient: f varies with latitude, so grad(g/f) is not
        # grad(g)/f.
        zero_col = np.zeros((self.nint, self.ops.grid.nlat, self.ops.grid.nlon))
        ghost_phi = self.vert.d_dpi(zero_col, tb, tt)
        ghost_psi = streamfunction_ghost(
            ghost_phi, self.frozen.f_grid, self.ops.grid.weights
        )
        gx, gy = self.ops.grad(self.ops.analyze(ghost_phi))
        sx, sy = self.ops.grad(self.ops.analyze(ghost_psi))
        b2_grid += (
            self.frozen.dpsi_x * gx
            + self.frozen.dpsi_y * gy
            + self.frozen.dphi_x * sx
            + self.frozen.dphi_y * sy
        )

        b1 = np.zeros(
            (self.nint, self.lmax + 1, self.lmax + 1), dtype=np.complex128
        )
        b2 = self.ops.analyze(b2_grid)
        b1[:, 0, 0] = 0.0  # the gauge rows ask for a zero mean streamfunction
        b2[:, 0, 0] = b2[:, 0, 0].real
        return b1, b2
