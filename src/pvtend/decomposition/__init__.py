"""Orthogonal four-basis decomposition for PV tendency analysis.

Submodules:
    smoothing: Gaussian and Fourier smoothing methods
    basis: Quadrupole basis construction and Gram-Schmidt orthogonalization
    projection: Project tendency fields onto orthogonal basis
"""

from .basis import (
    OrthogonalBasisFields,
    compute_orthogonal_basis,
    PRENORM_PHI1,
    PRENORM_PHI2,
    PRENORM_PHI3,
    PRENORM_PHI4,
)
from .projection import project_field, collect_term_fields
from .smoothing import gaussian_smooth_nan, fourier_lowpass_nan

__all__ = [
    "OrthogonalBasisFields",
    "compute_orthogonal_basis",
    "project_field",
    "collect_term_fields",
    "gaussian_smooth_nan",
    "fourier_lowpass_nan",
    "PRENORM_PHI1",
    "PRENORM_PHI2",
    "PRENORM_PHI3",
    "PRENORM_PHI4",
]
