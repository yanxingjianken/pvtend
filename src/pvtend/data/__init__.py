"""Bundled sample data for pvtend quickstart and testing.

Provides :func:`load_idealized_pv` to load a synthetic Gaussian PV anomaly
that undergoes simultaneous propagation, intensification, and deformation
over one time-step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

_DATA_DIR = Path(__file__).parent


def load_idealized_pv() -> Dict[str, np.ndarray]:
    """Load the idealized Gaussian PV evolution sample data.

    The dataset contains two PV snapshots (``q0``, ``q1``) on a 101×121
    idealized f-plane grid with 60 km spacing and a 1-hour time-step.

    The PV anomaly undergoes simultaneous:
      - Zonal propagation (αx = 10 m/s eastward)
      - Meridional propagation (αy = 5 m/s northward)
      - Intensification (β = 2×10⁻⁶ s⁻¹)
      - Deformation (γ_q = 0.08 quadrupole)

    Returns:
        Dict with keys:
            - ``q0``: PV at t=0 (101, 121) [PVU]
            - ``q1``: PV at t=1 (101, 121) [PVU]
            - ``x_km``, ``y_km``: coordinate vectors in km
            - ``x_deg``, ``y_deg``: coordinate vectors in degrees
            - ``dx_arr``: zonal grid spacing per latitude row [m]
            - ``dx_m``, ``dy_m``: scalar grid spacings [m]
            - ``dt``: time-step [s]
            - ``grid_spacing_deg``: grid spacing in degrees

    Example:
        >>> from pvtend.data import load_idealized_pv
        >>> d = load_idealized_pv()
        >>> d['q0'].shape
        (101, 121)
    """
    path = _DATA_DIR / "idealized_pv.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Sample data not found at {path}.  "
            "Please reinstall pvtend or run the data-generation notebook."
        )
    npz = np.load(path, allow_pickle=False)
    return {k: npz[k] for k in npz.files}
