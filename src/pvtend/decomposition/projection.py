"""Project tendency fields onto orthogonal PV basis.

The projection coefficient formula:
    c_i = <f, Φ̂_i> / <Φ̂_i, Φ̂_i>

Physical unit recovery:
    coef_physical = coef_raw × PRENORM

Coefficients:
    β (beta): Intensification rate [s⁻¹]
    αx (ax): Zonal propagation speed [m/s]
    αy (ay): Meridional propagation speed [m/s]
    γ (gamma): Deformation rate [s⁻¹]
"""

from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np

from .basis import (
    OrthogonalBasisFields,
    PRENORM_PHI1,
    PRENORM_PHI2,
    PRENORM_PHI3,
    PRENORM_PHI4,
    weighted_inner_product,
)
from .smoothing import gaussian_smooth_nan, fourier_lowpass_nan

# Standard PV budget advection terms (12 basic cross-terms)
ADVECTION_TERMS: tuple[str, ...] = (
    "u_anom_pv_anom_dx", "u_anom_pv_bar_dx",
    "u_bar_pv_anom_dx", "u_bar_pv_bar_dx",
    "v_anom_pv_anom_dy", "v_anom_pv_bar_dy",
    "v_bar_pv_anom_dy", "v_bar_pv_bar_dy",
    "w_anom_pv_anom_dp", "w_anom_pv_bar_dp",
    "w_bar_pv_anom_dp", "w_bar_pv_bar_dp",
)

# Helmholtz-decomposed horizontal eddy advection terms
HELMHOLTZ_TERMS: tuple[str, ...] = (
    "u_rot_pv_anom_dx", "u_rot_pv_bar_dx",
    "v_rot_pv_anom_dy", "v_rot_pv_bar_dy",
    "u_div_pv_anom_dx", "u_div_pv_bar_dx",
    "v_div_pv_anom_dy", "v_div_pv_bar_dy",
    "u_har_pv_anom_dx", "u_har_pv_bar_dx",
    "v_har_pv_anom_dy", "v_har_pv_bar_dy",
)

# Moist/dry divergent horizontal eddy advection terms
MOIST_DRY_H_TERMS: tuple[str, ...] = (
    "u_div_moist_pv_anom_dx", "u_div_moist_pv_bar_dx",
    "v_div_moist_pv_anom_dy", "v_div_moist_pv_bar_dy",
    "u_div_dry_pv_anom_dx", "u_div_dry_pv_bar_dx",
    "v_div_dry_pv_anom_dy", "v_div_dry_pv_bar_dy",
)

# Moist/dry omega vertical advection terms
MOIST_DRY_V_TERMS: tuple[str, ...] = (
    "omega_moist_pv_anom_dp", "omega_moist_pv_bar_dp",
    "omega_dry_pv_anom_dp", "omega_dry_pv_bar_dp",
)

# All advection terms combined
ALL_ADVECTION_TERMS: tuple[str, ...] = (
    ADVECTION_TERMS + HELMHOLTZ_TERMS + MOIST_DRY_H_TERMS + MOIST_DRY_V_TERMS
)


def project_field(
    field2d: np.ndarray,
    basis: OrthogonalBasisFields,
    *,
    apply_smoothing: bool = False,
    smoothing_deg: float = 6.0,
    smoothing_method: str = "gaussian",
    grid_spacing: float = 1.5,
) -> Dict[str, object]:
    """Project a tendency field onto the orthogonal basis.

    Parameters:
        field2d: 2D tendency field in SI units.
        basis: Orthogonal basis container.
        apply_smoothing: Smooth tendency before projection.
        smoothing_deg: Smoothing FWHM (degrees).
        smoothing_method: 'gaussian' or 'fourier'.
        grid_spacing: Grid spacing in degrees.

    Returns:
        Dict with coefficients (beta, ax, ay, gamma) in physical units,
        raw coefficients, component fields (int, prop, def), residual,
        reconstruction, and RMSE.
    """
    arr = np.asarray(field2d, dtype=float)
    if arr.shape != basis.grid_shape:
        raise ValueError("field shape does not match basis grid")

    if apply_smoothing:
        if smoothing_method == "gaussian":
            arr = gaussian_smooth_nan(arr, smoothing_deg, grid_spacing)
        elif smoothing_method == "fourier":
            x_ext = float(basis.x_rel.max() - basis.x_rel.min())
            y_ext = float(basis.y_rel.max() - basis.y_rel.min())
            arr = fourier_lowpass_nan(arr, smoothing_deg, x_ext, y_ext)

    valid = basis.mask & np.isfinite(arr)
    if not valid.any():
        raise ValueError("No valid points for projection")

    norms = basis.norms or {}
    norm_int = norms.get("beta", 1.0)
    norm_dx = norms.get("ax", 1.0)
    norm_dy = norms.get("ay", 1.0)
    norm_def = norms.get("gamma", 1.0)

    sf = basis.scale_factors or {}
    sf_int = sf.get("beta", PRENORM_PHI1)
    sf_dx = sf.get("ax", PRENORM_PHI2)
    sf_dy = sf.get("ay", PRENORM_PHI3)
    sf_def = sf.get("gamma", PRENORM_PHI4)

    inner_int = weighted_inner_product(arr, basis.phi_int, basis.weights, basis.mask)
    inner_dx = weighted_inner_product(arr, basis.phi_dx, basis.weights, basis.mask)
    inner_dy = weighted_inner_product(arr, basis.phi_dy, basis.weights, basis.mask)
    inner_def = weighted_inner_product(arr, basis.phi_def, basis.weights, basis.mask)

    beta_raw = inner_int / norm_int if norm_int > 1e-30 else 0.0
    ax_raw = -inner_dx / norm_dx if norm_dx > 1e-30 else 0.0
    ay_raw = -inner_dy / norm_dy if norm_dy > 1e-30 else 0.0
    gamma_raw = inner_def / norm_def if norm_def > 1e-30 else 0.0

    beta = beta_raw * sf_int
    ax = ax_raw * sf_dx
    ay = ay_raw * sf_dy
    gamma = gamma_raw * sf_def

    inten = beta_raw * basis.phi_int
    prop = -ax_raw * basis.phi_dx - ay_raw * basis.phi_dy
    deform = gamma_raw * basis.phi_def
    recon = inten + prop + deform
    resid = np.where(basis.mask, arr - recon, np.nan)

    rmse = math.sqrt(np.nanmean((arr[valid] - recon[valid]) ** 2))

    return {
        "beta": float(beta),
        "ax": float(ax),
        "ay": float(ay),
        "gamma": float(gamma),
        "gamma_km2": float(gamma * 1e6),
        "beta_raw": float(beta_raw),
        "ax_raw": float(ax_raw),
        "ay_raw": float(ay_raw),
        "gamma_raw": float(gamma_raw),
        "prop": prop,
        "int": inten,
        "def": deform,
        "resid": resid,
        "recon": recon,
        "rmse": float(rmse),
    }


def collect_term_fields(
    fields_dict: Dict[str, np.ndarray],
    include_terms: Iterable[str] | None = None,
) -> Dict[str, np.ndarray]:
    """Collect and sign-correct PV budget term fields.

    Parameters:
        fields_dict: Dict of field_name → 2D array.
        include_terms: Terms to include (default: all advection + pv_dt, Q).

    Returns:
        Dict of term_name → 2D numpy array (sign-corrected).
    """
    if include_terms is None:
        include_terms = list(ADVECTION_TERMS) + ["pv_dt", "Q"]

    out: Dict[str, np.ndarray] = {}
    for term in include_terms:
        if term not in fields_dict:
            continue
        arr = np.asarray(fields_dict[term], dtype=float)
        if term in ADVECTION_TERMS:
            arr = -arr  # Sign convention: -u·∇q
        out[term] = arr
    return out
