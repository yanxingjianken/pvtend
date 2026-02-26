"""Visualization tools for PV tendency analysis.

Submodules:
    basis_plots: Plot the four orthogonal basis fields
    coefficient_plots: Time series of decomposition coefficients
    field_plots: 2D field maps (PV, wind, omega)
    composite_explorer: Single-variable composite + bootstrap + projection
    baroclinic: Two-level v′ overlay for baroclinic tilt
"""

from .basis_plots import plot_four_basis, plot_basis_with_contours
from .coefficient_plots import plot_coefficient_curves
from .field_plots import plot_field_2d, plot_wind_overlay
from .composite_explorer import plot_var, get_field, bootstrap_sig, load_events
from .baroclinic import plot_baroclinic_tilt

__all__ = [
    "plot_four_basis",
    "plot_basis_with_contours",
    "plot_coefficient_curves",
    "plot_field_2d",
    "plot_wind_overlay",
    # Composite explorer (from nb04)
    "plot_var",
    "get_field",
    "bootstrap_sig",
    "load_events",
    # Baroclinic structure (from nb06)
    "plot_baroclinic_tilt",
]
