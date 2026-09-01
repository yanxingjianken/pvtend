"""Vertical level presets and the Exner-coordinate tables they imply.

Single source of truth for the pressure levels and every quantity derived from
them.  Two presets are supported:

``NL9``
    1000, 850, 700, 500, 400, 300, 250, 200, 100 hPa.  The per-level outputs are
    named for these pressures, and downstream consumers expect that set.
``NL10``
    The same nine with 150 hPa added.

Interior potential vorticity lives on ``k = 1 .. NL-2`` (0-based); the first and
last levels carry boundary potential temperature instead and are reconstructed
hydrostatically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Specific heat at constant pressure [J kg^-1 K^-1].
CP = 1004.5
#: Gas constant for dry air [J kg^-1 K^-1].
RD = 287.0
#: kappa = Rd/Cp, which is exactly 2/7 for the constants above.  The exactness
#: matters: only at 2/7 does the exponent ``1 - kappa`` equal ``5 kappa / 2``,
#: which is what collapses the potential-vorticity scaling of
#: :func:`pv_rhs_scale` to a single number instead of a vertical profile.
KAPPA = 2.0 / 7.0
#: Gravity [m s^-2].
G = 9.81
#: Reference pressure [Pa].
P0 = 1.0e5
#: Earth angular velocity [s^-1].
OMEGA = 7.292115e-5
#: Earth radius [m].
R_EARTH = 6.371e6

#: Pressure levels [hPa], bottom-up, per preset.
PLEVS: dict[str, tuple[float, ...]] = {
    "NL9": (1000.0, 850.0, 700.0, 500.0, 400.0, 300.0, 250.0, 200.0, 100.0),
    "NL10": (1000.0, 850.0, 700.0, 500.0, 400.0, 300.0, 250.0, 200.0, 150.0, 100.0),
}


@dataclass(frozen=True)
class LevelSet:
    """Pressure levels and the Exner-coordinate operators built on them.

    Attributes:
        name: Preset name (``"NL9"`` / ``"NL10"``).
        p_hpa: Pressure levels [hPa], bottom-up.
        pi: Exner function ``Pi = Cp (p/p0)**kappa`` [J kg^-1 K^-1].
        bb, bh, bl: Second-derivative stencil weights in Pi, defined on interior
            levels only (entries at ``k = 0`` and ``k = NL-1`` are zero):
            ``d2f/dPi2 |_k = bl_k f_{k-1} + bb_k f_k + bh_k f_{k+1}``.
        dpi2: Centred first-difference denominator ``(Pi_{k+1} - Pi_{k-1})/2``,
            interior only.  The boundary-theta forcing enters divided by this.
    """

    name: str
    p_hpa: np.ndarray
    pi: np.ndarray
    bb: np.ndarray
    bh: np.ndarray
    bl: np.ndarray
    dpi2: np.ndarray

    @property
    def nlev(self) -> int:
        return int(self.p_hpa.size)

    @property
    def interior(self) -> np.ndarray:
        """0-based indices of the levels carrying interior PV."""
        return np.arange(1, self.nlev - 1)


def exner(p_hpa: np.ndarray) -> np.ndarray:
    """Exner function ``Cp (p/p0)**kappa`` for pressures given in hPa."""
    p_pa = np.asarray(p_hpa, dtype=float) * 100.0
    return CP * (p_pa / P0) ** KAPPA


def build_levels(name: str = "NL9") -> LevelSet:
    """Build a :class:`LevelSet` from a preset name.

    The stencil weights are the three-point second difference on a non-uniform
    grid.  With ``dp_up = Pi_{k+1} - Pi_k``, ``dp_dn = Pi_k - Pi_{k-1}`` and
    ``dp_2 = dp_up + dp_dn``, the weights ``2/(dp_dn dp_2)``, ``-2/(dp_up dp_dn)``
    and ``2/(dp_up dp_2)`` are fixed uniquely by asking that the stencil
    annihilate a constant and a linear function and return two for ``Pi**2``.
    Equal spacing collapses them to the familiar ``1, -2, 1`` over ``dp**2``, but
    the Exner levels are far from equally spaced: on ``NL9`` the ratio
    ``dp_up / dp_dn`` runs from 0.59 at 300 hPa to 2.73 at 200 hPa.  On such a
    column the equal-spacing form does not merely lose accuracy: it no longer
    annihilates a linear profile, so a geopotential linear in ``Pi`` -- a column
    of constant potential temperature -- comes back with a non-zero static
    stability.

    The table is built on ``Pi`` itself, in SI.  Measuring the coordinate in units
    of some constant instead scales every weight by the square of that constant,
    which cancels out of the solution but leaves the numbers here incomparable
    with the level spacings they came from.
    """
    try:
        p_hpa = np.asarray(PLEVS[name], dtype=float)
    except KeyError:
        raise ValueError(
            f"unknown level preset {name!r}; available: {sorted(PLEVS)}"
        ) from None

    pi = exner(p_hpa)
    nlev = pi.size
    bb = np.zeros(nlev)
    bh = np.zeros(nlev)
    bl = np.zeros(nlev)
    dpi2 = np.zeros(nlev)
    for k in range(1, nlev - 1):
        dp_up = pi[k + 1] - pi[k]
        dp_dn = pi[k] - pi[k - 1]
        dp_2 = pi[k + 1] - pi[k - 1]
        bb[k] = -2.0 / (dp_up * dp_dn)
        bh[k] = 2.0 / (dp_up * dp_2)
        bl[k] = 2.0 / (dp_dn * dp_2)
        dpi2[k] = dp_2 / 2.0

    return LevelSet(
        name=name, p_hpa=p_hpa, pi=pi, bb=bb, bh=bh, bl=bl, dpi2=dpi2
    )


def pv_rhs_scale(p_hpa: np.ndarray) -> np.ndarray:
    """Factor taking SI Ertel PV to the right-hand side of the PV equation.

    Mapping ``d/dp`` onto ``d/dPi`` (``dPi/dp = kappa Pi / p``) and using
    hydrostatic balance ``dPhi/dPi = -theta`` turns the Ertel potential vorticity
    into

    ``q_SI = g kappa (Pi/p) [ (f + zeta) Phi_PiPi - grad(psi_Pi) . grad(Phi_Pi) ]``

    so the bracket -- what the inversion actually solves for -- is
    ``q_hat = q_SI * p / (g kappa Pi) = q_SI * (p/p0)**(1-kappa) * p0/(kappa g Cp)``.

    This one factor is the whole of the conversion.  Nothing else in the solver
    is scaled: every field it holds is SI, and this is the only place a unit
    changes.  With ``kappa = 2/7`` the exponent ``1 - kappa`` equals
    ``5 kappa / 2``, so the factor is a pure power of the pressure ratio with no
    separate level-dependent constant left over -- which is why a potential
    vorticity written in a scaled system converts here by one number at every
    level rather than by a profile.

    Args:
        p_hpa: Pressure levels in hPa.

    Returns:
        Multiplicative factor per level, applied to PV in K m^2 kg^-1 s^-1.
    """
    p_pa = np.asarray(p_hpa, dtype=float) * 100.0
    return p_pa / (G * KAPPA * exner(p_hpa))
