"""Orthogonal four-basis construction for PV tendency decomposition.

The four basis fields:
    Φ₁ = q'          (PV anomaly — intensification)
    Φ₂ = ∂q/∂x       (zonal gradient — zonal propagation)
    Φ₃ = ∂q/∂y       (meridional gradient — meridional propagation)
    Φ₄ = ∂²q/∂x∂y    (cross-derivative — deformation/quadrupole)

Processing order:
    1. Load raw fields in SI units
    2. Compute cross-derivative Φ₄ from ∂q/∂x
    3. Apply FIXED pre-normalization constants
    4. Apply smoothing
    5. Gram-Schmidt orthogonalization
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .smoothing import gaussian_smooth_nan, fourier_lowpass_nan

# Fixed pre-normalization constants (scale SI fields to O(1))
PRENORM_PHI1: float = 1e6   # q' (PVU)
PRENORM_PHI2: float = 1e12  # ∂q/∂x (PVU/m)
PRENORM_PHI3: float = 1e12  # ∂q/∂y (PVU/m)
PRENORM_PHI4: float = 1e18  # ∂²q/∂x∂y (PVU/m²)

R_EARTH = 6.371e6  # m


@dataclass(frozen=True)
class OrthogonalBasisFields:
    """Container for the four orthogonal basis fields.

    Attributes:
        phi_int: Intensification basis (Φ₁).
        phi_dx: Zonal propagation basis (Φ₂).
        phi_dy: Meridional propagation basis (Φ₃).
        phi_def: Deformation/quadrupole basis (Φ₄).
        weights: 2D weighting array.
        mask: Boolean mask for valid grid points.
        x_rel: 1D relative x coordinates.
        y_rel: 1D relative y coordinates.
        Y_grid: 2D latitude grid.
        geopotential: Optional geopotential field.
        raw_phi_int: Raw (unorthogonalized) Φ₁.
        raw_phi_dx: Raw Φ₂.
        raw_phi_dy: Raw Φ₃.
        raw_phi_def: Raw Φ₄.
        norms: Dict of squared norms for each basis.
        scale_factors: PRENORM constants for rescaling.
    """
    phi_int: np.ndarray
    phi_dx: np.ndarray
    phi_dy: np.ndarray
    phi_def: np.ndarray
    weights: np.ndarray
    mask: np.ndarray
    x_rel: np.ndarray
    y_rel: np.ndarray
    Y_grid: np.ndarray
    geopotential: np.ndarray | None = None
    raw_phi_int: np.ndarray | None = None
    raw_phi_dx: np.ndarray | None = None
    raw_phi_dy: np.ndarray | None = None
    raw_phi_def: np.ndarray | None = None
    norms: Dict[str, float] | None = None
    scale_factors: Dict[str, float] | None = None

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.phi_int.shape


def weighted_inner_product(
    f: np.ndarray, g: np.ndarray, weights: np.ndarray, mask: np.ndarray,
) -> float:
    """Compute weighted inner product <f, g>_w over masked region."""
    valid = mask & np.isfinite(f) & np.isfinite(g) & np.isfinite(weights)
    if not valid.any():
        return 0.0
    return float(np.sum(weights[valid] * f[valid] * g[valid]))


def gram_schmidt_orthogonalize(
    bases: list[np.ndarray],
    weights: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[np.ndarray], list[float]]:
    """Gram-Schmidt orthogonalization WITHOUT normalization.

    Makes bases mutually orthogonal but preserves their norms.

    Parameters:
        bases: List of 2D basis fields [Φ₁, Φ₂, Φ₃, Φ₄].
        weights: 2D weighting field.
        mask: 2D boolean mask.

    Returns:
        (ortho_bases, norms): Orthogonalized bases and their squared norms.
    """
    ortho = []
    norms = []

    for i, b in enumerate(bases):
        v = b.copy()
        for j, o in enumerate(ortho):
            inner_vo = weighted_inner_product(v, o, weights, mask)
            if norms[j] > 1e-30:
                v = v - (inner_vo / norms[j]) * o
        norm_v = weighted_inner_product(v, v, weights, mask)
        ortho.append(v)
        norms.append(norm_v)

    return ortho, norms


def compute_quadrupole_basis(
    pv_dx: np.ndarray, y_rel: np.ndarray,
) -> np.ndarray:
    """Compute quadrupole basis as cross-derivative ∂²q/∂x∂y.

    Parameters:
        pv_dx: Zonal PV gradient ∂q/∂x in SI units (PVU/m).
        y_rel: Relative latitude degrees.

    Returns:
        Cross-derivative ∂²q/∂x∂y (PVU/m²).
    """
    dy_deg = y_rel[1] - y_rel[0] if len(y_rel) > 1 else 1.5
    dy_m = 2 * np.pi * R_EARTH / 360
    return np.gradient(np.asarray(pv_dx, dtype=float), dy_deg, axis=0) / dy_m


def compute_orthogonal_basis(
    pv_anom: np.ndarray,
    pv_dx: np.ndarray,
    pv_dy: np.ndarray,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    *,
    geopotential: np.ndarray | None = None,
    mask_negative: bool = True,
    apply_smoothing: bool = True,
    smoothing_deg: float = 6.0,
    smoothing_method: str = "gaussian",
    grid_spacing: float = 1.5,
    apply_lat_weighting: bool = False,
) -> OrthogonalBasisFields:
    """Compute four orthogonal basis fields from composite PV fields.

    Parameters:
        pv_anom: PV anomaly q' in SI units.
        pv_dx: Zonal PV gradient ∂q/∂x in SI.
        pv_dy: Meridional PV gradient ∂q/∂y in SI.
        x_rel: 1D relative x-coordinates (degrees).
        y_rel: 1D relative y-coordinates (degrees).
        geopotential: Optional geopotential for overlay plots.
        mask_negative: Restrict basis to q' < 0 region.
        apply_smoothing: Apply spatial smoothing after pre-normalization.
        smoothing_deg: Smoothing FWHM (degrees).
        smoothing_method: 'gaussian' or 'fourier'.
        grid_spacing: Grid spacing in degrees.
        apply_lat_weighting: Use cos(lat) weighting.

    Returns:
        OrthogonalBasisFields container.
    """
    # Step 1: Compute cross-derivative (before pre-normalization)
    raw_phi_def = compute_quadrupole_basis(pv_dx, y_rel)

    # Step 2: Pre-normalize
    phi_int = pv_anom * PRENORM_PHI1
    phi_dx_pn = pv_dx * PRENORM_PHI2
    phi_dy_pn = pv_dy * PRENORM_PHI3
    phi_def = raw_phi_def * PRENORM_PHI4

    # Step 3: Smooth
    fields = [phi_int, phi_dx_pn, phi_dy_pn, phi_def]
    if apply_smoothing:
        if smoothing_method == "gaussian":
            fields = [gaussian_smooth_nan(f, smoothing_deg, grid_spacing) for f in fields]
        elif smoothing_method == "fourier":
            x_ext = float(x_rel.max() - x_rel.min())
            y_ext = float(y_rel.max() - y_rel.min())
            fields = [fourier_lowpass_nan(f, smoothing_deg, x_ext, y_ext) for f in fields]
        else:
            raise ValueError(f"Unknown smoothing_method: {smoothing_method}")
    phi_int_s, phi_dx_s, phi_dy_s, phi_def_s = fields

    # Step 4: Mask
    mask = np.isfinite(phi_int_s) & np.isfinite(phi_dx_s) & \
           np.isfinite(phi_dy_s) & np.isfinite(phi_def_s)
    if mask_negative:
        mask &= phi_int_s < 0.0

    # Step 5: Weights
    X_grid, Y_grid = np.meshgrid(x_rel, y_rel)
    if apply_lat_weighting:
        weights = np.cos(np.deg2rad(Y_grid))
        weights = np.where(np.isfinite(weights), weights, 0.0)
    else:
        weights = np.ones_like(Y_grid)
    weights = np.where(mask, weights, 0.0)

    # Step 6: Gram-Schmidt
    ortho, norms = gram_schmidt_orthogonalize(
        [phi_int_s, phi_dx_s, phi_dy_s, phi_def_s], weights, mask)

    prenorm_dict = {
        "beta": PRENORM_PHI1,
        "ax": PRENORM_PHI2,
        "ay": PRENORM_PHI3,
        "gamma": PRENORM_PHI4,
    }

    return OrthogonalBasisFields(
        phi_int=np.asarray(ortho[0], dtype=float),
        phi_dx=np.asarray(ortho[1], dtype=float),
        phi_dy=np.asarray(ortho[2], dtype=float),
        phi_def=np.asarray(ortho[3], dtype=float),
        weights=weights,
        mask=mask,
        x_rel=np.asarray(x_rel, dtype=float),
        y_rel=np.asarray(y_rel, dtype=float),
        Y_grid=Y_grid,
        geopotential=geopotential,
        raw_phi_int=pv_anom,
        raw_phi_dx=pv_dx,
        raw_phi_dy=pv_dy,
        raw_phi_def=raw_phi_def,
        norms={"beta": norms[0], "ax": norms[1], "ay": norms[2], "gamma": norms[3]},
        scale_factors=prenorm_dict,
    )
