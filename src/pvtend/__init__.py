"""pvtend — PV-tendency decomposition for atmospheric blocking and PRP events.

This package provides a complete pipeline for:

1. Loading ERA5 pressure-level data
2. Computing PV tendencies (advection, stretching, diabatic, residual)
3. QG omega and Helmholtz decomposition
4. Moist/dry omega partitioning
5. Orthogonal basis decomposition of PV tendency
6. Rossby wave breaking detection and classification
7. Composite lifecycle analysis
8. Publication-quality visualization

Example:
    >>> import pvtend
    >>> pvtend.__version__
    '0.1.0'
"""

from __future__ import annotations

from pvtend._version import __version__

# Core data structures and grid
from pvtend.grid import NHGrid, default_nh_grid, EventPatch
from pvtend.constants import (
    R_EARTH, OMEGA_E, G0, R_DRY, KAPPA, H_SCALE,
    SP19_DRY_FRACTION, MASK_PV_THRESHOLD,
)

# Derivatives and climatology
from pvtend.derivatives import ddx, ddy, ddp, ddt
from pvtend.climatology import compute_climatology, load_climatology

# Solvers
from pvtend.omega import solve_qg_omega, solve_qg_omega_sip
from pvtend.helmholtz import helmholtz_decomposition, helmholtz_decomposition_3d
from pvtend.moist_dry import decompose_omega

# Decomposition
from pvtend.decomposition import (
    OrthogonalBasisFields,
    compute_orthogonal_basis,
    project_field,
    collect_term_fields,
)

# Isentropic interpolation
from pvtend.isentropic import (
    isentropic_interpolation,
    isentropic_interpolation_pressure,
    interp_event_fields_to_theta,
    interp_event_field_to_single_theta,
)

# RWB and composites
from pvtend.rwb import RWBConfig, detect_rwb_events
from pvtend.composites import CompositeState, load_composite_state

# Tendency computation
from pvtend.tendency import TendencyComputer, TendencyConfig

__all__ = [
    "__version__",
    # Grid
    "NHGrid", "default_nh_grid", "EventPatch",
    # Constants
    "R_EARTH", "OMEGA_E", "G0", "R_DRY", "KAPPA", "H_SCALE",
    "SP19_DRY_FRACTION", "MASK_PV_THRESHOLD",
    # Derivatives
    "ddx", "ddy", "ddp", "ddt",
    # Climatology
    "compute_climatology", "load_climatology",
    # Solvers
    "solve_qg_omega", "solve_qg_omega_sip",
    "helmholtz_decomposition", "helmholtz_decomposition_3d",
    "decompose_omega",
    # Decomposition
    "OrthogonalBasisFields", "compute_orthogonal_basis",
    "project_field", "collect_term_fields",
    # RWB
    "RWBConfig", "detect_rwb_events",
    # Composites
    "CompositeState", "load_composite_state",
    # Isentropic
    "isentropic_interpolation", "isentropic_interpolation_pressure",
    "interp_event_fields_to_theta", "interp_event_field_to_single_theta",
    # Tendency
    "TendencyComputer", "TendencyConfig",
]
