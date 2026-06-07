"""Rotational winds from a Wu piecewise streamfunction ψ′.

The Wu inversion is solved on a **limited-area, non-periodic** regional
window, so the spectral (FFT) zonal derivative used elsewhere in pvtend
(:func:`pvtend.helmholtz.gradient`) is *not* valid here — it would wrap the
zonal boundary.  This module instead uses Wu's own plain centred
finite-difference operator on the regional grid, matching the validated
standalone pipeline (``08_parse_outputs/parse_and_pv.py``):

.. code-block:: text

    RE_WU = 2.0e7 / pi                 # Wu Earth radius [m]
    DP    = RE_WU * radians(dlat)
    DL    = RE_WU * radians(dlon)
    AP    = cos(radians(lat))
    u_rot = (psi[j+1] - psi[j-1]) / (2*DP)            # = -dψ/dy
    v_rot = (psi[:,i+1] - psi[:,i-1]) / (2*DL*cosφ)   # = +dψ/dx

Latitude rows run **north → south** (j increasing southward); the centred
difference therefore yields ``u_rot = -∂ψ/∂y`` with y pointing north.
Domain-edge rows/columns are returned as NaN (no one-sided closure), to be
cropped away before use.
"""
from __future__ import annotations

import numpy as np

#: Wu Earth radius [m] (2·10⁷ / π — gives 1° ≈ DP at the equator).
RE_WU: float = 2.0e7 / np.pi


def psi_to_winds(
    psi: np.ndarray,
    lats: np.ndarray,
    dlat: float,
    dlon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotational wind from ψ via Wu centred finite differences.

    Args:
        psi: Streamfunction ``(NL, NY, NX)`` [m²/s], latitude N→S.
        lats: Latitudes [deg] of length ``NY`` (descending, N→S).
        dlat: Grid spacing in latitude [deg].
        dlon: Grid spacing in longitude [deg].

    Returns:
        Tuple ``(u_rot, v_rot)`` each ``(NL, NY, NX)`` [m/s]; the outermost
        latitude rows and longitude columns are NaN.
    """
    psi = np.asarray(psi, dtype=np.float64)
    if psi.ndim != 3:
        raise ValueError(f"psi must be 3-D (NL,NY,NX); got {psi.shape}")
    nl, ny, nx = psi.shape
    lats = np.asarray(lats, dtype=np.float64)
    if lats.shape[0] != ny:
        raise ValueError(f"len(lats)={lats.shape[0]} != NY={ny}")

    dp = RE_WU * np.radians(dlat)
    dl = RE_WU * np.radians(dlon)
    ap = np.cos(np.radians(lats))  # (NY,)

    u_rot = np.full_like(psi, np.nan)
    v_rot = np.full_like(psi, np.nan)

    # u = -dψ/dy : centred in latitude index (N→S), interior rows only.
    u_rot[:, 1:-1, :] = (psi[:, 2:, :] - psi[:, :-2, :]) / (2.0 * dp)

    # v = +dψ/dx : centred in longitude, scaled by cosφ, interior cols only.
    for i in range(1, ny - 1):
        v_rot[:, i, 1:-1] = (psi[:, i, 2:] - psi[:, i, :-2]) / (2.0 * dl * ap[i])

    # Blank the domain edges (no one-sided closure — matches baseline).
    u_rot[:, [0, -1], :] = np.nan
    u_rot[:, :, [0, -1]] = np.nan
    v_rot[:, [0, -1], :] = np.nan
    v_rot[:, :, [0, -1]] = np.nan
    return u_rot, v_rot
