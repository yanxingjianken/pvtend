"""Orthogonal six-basis decomposition for PV tendency analysis.

Submodules:
    smoothing: Gaussian and Fourier smoothing methods
    basis: Six-basis construction and Gram-Schmidt orthogonalization
    projection: Project tendency fields onto orthogonal basis
"""

from .basis import (
    OrthogonalBasisFields,
    compute_orthogonal_basis,
    compute_strain_basis,
    compute_laplacian_basis,
    auto_prenorm,
    PRENORM_PHI1,
    PRENORM_PHI2,
    PRENORM_PHI3,
    PRENORM_PHI4,
    PRENORM_PHI5,
    PRENORM_PHI6,
)
from ..constants import MASK_PV_THRESHOLD
from .projection import project_field, collect_term_fields
from .smoothing import gaussian_smooth_nan, fourier_lowpass_nan
from .interpolation import lerp_fields

__all__ = [
    "OrthogonalBasisFields",
    "compute_orthogonal_basis",
    "compute_strain_basis",
    "compute_laplacian_basis",
    "auto_prenorm",
    "project_field",
    "collect_term_fields",
    "gaussian_smooth_nan",
    "fourier_lowpass_nan",
    "lerp_fields",
    "PRENORM_PHI1",
    "PRENORM_PHI2",
    "PRENORM_PHI3",
    "PRENORM_PHI4",
    "PRENORM_PHI5",
    "PRENORM_PHI6",
    "MASK_PV_THRESHOLD",
]
