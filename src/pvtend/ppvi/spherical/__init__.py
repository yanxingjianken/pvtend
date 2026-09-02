"""Global spherical-harmonic piecewise potential-vorticity inversion.

Solves the Davis & Emanuel (1991) nonlinear-balance inversion on the whole
sphere.  The domain is closed, so no lateral boundary condition has to be chosen,
and every horizontal term is assembled in a form that carries no explicit metric
factor, so an event centred anywhere -- including across a pole -- inverts
without a singularity and the decomposition is exactly its sources.
"""
from __future__ import annotations

__version__ = "0.3.0"

from .levels import build_levels, pv_rhs_scale
from .sht import SHT, Grid, gaussian_grid, grid_from_axes

__all__ = [
    "SHT",
    "Grid",
    "build_levels",
    "gaussian_grid",
    "grid_from_axes",
    "pv_rhs_scale",
    "__version__",
]
