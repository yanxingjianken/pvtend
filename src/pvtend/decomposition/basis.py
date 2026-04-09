"""Orthogonal six-basis construction for PV tendency decomposition.

The six basis fields:
    Φ₁ = q'                      (PV anomaly — intensification)
    Φ₂ = ∂q/∂x                   (zonal gradient — zonal propagation)
    Φ₃ = ∂q/∂y                   (meridional gradient — meridional propagation)
    Φ₄ = ∂²q/∂x∂y                (cross-derivative — shear deformation)
    Φ₅ = ∂²q/∂x² − ∂²q/∂y²      (normal strain deformation)
    Φ₆ = ∂²q/∂x² + ∂²q/∂y²      (Laplacian — diffusion)

Processing order:
    1. Load raw fields in SI units
    2. Compute second-order derivatives (Φ₄, Φ₅, Φ₆)
    3. Apply per-event auto pre-normalization to O(1)
    4. Apply smoothing
    5. Gram-Schmidt orthogonalization
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.ndimage import label as _ndimage_label

from .smoothing import gaussian_smooth_nan, fourier_lowpass_nan
from ..derivatives import ddx as _ddx, ddy as _ddy

# Default mask threshold for q' region (SI units, PVU)
from ..constants import MASK_PV_THRESHOLD as _DEFAULT_MASK_THRESHOLD

# Fixed pre-normalization constants (legacy — kept for backward compat)
PRENORM_PHI1: float = 1e6   # q' (PVU)
PRENORM_PHI2: float = 1e12  # ∂q/∂x (PVU/m)
PRENORM_PHI3: float = 1e12  # ∂q/∂y (PVU/m)
PRENORM_PHI4: float = 1e18  # ∂²q/∂x∂y (PVU/m²)
PRENORM_PHI5: float = 1e18  # ∂²q/∂x²−∂²q/∂y² (PVU/m²)
PRENORM_PHI6: float = 1e18  # ∂²q/∂x²+∂²q/∂y² (PVU/m²)

R_EARTH = 6.371e6  # m


@dataclass(frozen=True)
class OrthogonalBasisFields:
    """Container for the six orthogonal basis fields.

    Attributes:
        phi_int: Intensification basis (Φ₁).
        phi_dx: Zonal propagation basis (Φ₂).
        phi_dy: Meridional propagation basis (Φ₃).
        phi_def: Shear deformation basis (Φ₄ = ∂²q/∂x∂y).
        phi_strain: Normal strain basis (Φ₅ = ∂²q/∂x² − ∂²q/∂y²).
        phi_lap: Laplacian/diffusion basis (Φ₆ = ∂²q/∂x² + ∂²q/∂y²).
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
        raw_phi_strain: Raw Φ₅.
        raw_phi_lap: Raw Φ₆.
        norms: Dict of squared norms for each basis.
        scale_factors: PRENORM constants for rescaling.
    """
    phi_int: np.ndarray
    phi_dx: np.ndarray
    phi_dy: np.ndarray
    phi_def: np.ndarray
    phi_strain: np.ndarray
    phi_lap: np.ndarray
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
    raw_phi_strain: np.ndarray | None = None
    raw_phi_lap: np.ndarray | None = None
    norms: Dict[str, float] | None = None
    scale_factors: Dict[str, float] | None = None

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.phi_int.shape

    @property
    def n_bases(self) -> int:
        """Number of basis fields (always 6)."""
        return 6


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
    *,
    center_lat: float = 60.0,
) -> np.ndarray:
    """Compute quadrupole basis as cross-derivative ∂²q/∂x∂y.

    Uses :func:`pvtend.derivatives.ddy` with cos(lat)-independent
    meridional metric (latitude spacing is constant on a lat-lon grid).

    Parameters:
        pv_dx: Zonal PV gradient ∂q/∂x in SI units (PVU/m).
        y_rel: Relative latitude degrees.
        center_lat: Patch centre latitude (degrees N).  Unused for the
            meridional derivative (dy is latitude-independent), but
            kept for API symmetry.

    Returns:
        Cross-derivative ∂²q/∂x∂y (PVU/m²).
    """
    dy_deg = y_rel[1] - y_rel[0] if len(y_rel) > 1 else 1.5
    dy_m = np.deg2rad(abs(dy_deg)) * R_EARTH
    return _ddy(np.asarray(pv_dx, dtype=float), dy_m)


def compute_strain_basis(
    pv_dx_dx: np.ndarray, pv_dy_dy: np.ndarray,
) -> np.ndarray:
    """Compute normal strain deformation basis: ∂²q/∂x² − ∂²q/∂y².

    Parameters:
        pv_dx_dx: ∂²q/∂x² in SI units (PVU/m²).
        pv_dy_dy: ∂²q/∂y² in SI units (PVU/m²).

    Returns:
        Normal strain (PVU/m²).
    """
    return np.asarray(pv_dx_dx, dtype=float) - np.asarray(pv_dy_dy, dtype=float)


def compute_laplacian_basis(
    pv_dx_dx: np.ndarray, pv_dy_dy: np.ndarray,
) -> np.ndarray:
    """Compute Laplacian (diffusion) basis: ∂²q/∂x² + ∂²q/∂y².

    Parameters:
        pv_dx_dx: ∂²q/∂x² in SI units (PVU/m²).
        pv_dy_dy: ∂²q/∂y² in SI units (PVU/m²).

    Returns:
        Laplacian (PVU/m²).
    """
    return np.asarray(pv_dx_dx, dtype=float) + np.asarray(pv_dy_dy, dtype=float)


def auto_prenorm(
    fields: list[np.ndarray],
    mask: np.ndarray | None = None,
) -> list[float]:
    """Compute per-event pre-normalization constants.

    Each constant is 1 / median(|field|) over the masked region,
    so that the pre-normalized field has O(1) magnitude.

    Parameters:
        fields: List of raw SI-unit basis fields.
        mask: Optional boolean mask (True = include).

    Returns:
        List of scale factors, one per field.
    """
    scales = []
    for f in fields:
        arr = np.asarray(f, dtype=float)
        if mask is not None:
            valid = mask & np.isfinite(arr) & (arr != 0.0)
        else:
            valid = np.isfinite(arr) & (arr != 0.0)
        if not valid.any():
            scales.append(1.0)
            continue
        med = float(np.median(np.abs(arr[valid])))
        scales.append(1.0 / med if med > 1e-30 else 1.0)
    return scales


# ── mask parameter parser ──────────────────────────────────────────────
_MASK_RE = re.compile(
    r"^\s*([<>]=?)\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*$"
)


def _parse_mask_spec(
    mask_spec: str | bool | np.ndarray | None,
    phi_int_s: np.ndarray,
    *,
    _mask_negative: bool | None = None,
    _mask_threshold: float | None = None,
) -> np.ndarray | None:
    """Return a boolean mask from the ``mask`` parameter.

    Parameters:
        mask_spec: New-style ``mask`` argument.
        phi_int_s: Pre-normalized + smoothed Φ₁ field.
        _mask_negative: Deprecated ``mask_negative`` kwarg (for shim).
        _mask_threshold: Deprecated ``mask_threshold`` kwarg (for shim).

    Returns:
        Boolean 2-D array (True = include), or *None* for no PV masking.
    """
    # --- Deprecation shim: old kwargs override if mask_spec was not given ---
    if mask_spec is None and (_mask_negative is not None or _mask_threshold is not None):
        if _mask_threshold is not None:
            # Explicit old-style threshold → convert to string spec
            mask_spec = f"< {_mask_threshold}"
        elif _mask_negative is True:
            mask_spec = f"< {_DEFAULT_MASK_THRESHOLD}"
        elif _mask_negative is False:
            return None  # no masking
        # else: _mask_negative is None and _mask_threshold is None → fall through

    # --- Resolve mask_spec ---
    if mask_spec is None or mask_spec is False:
        return None

    if mask_spec is True:
        # Backward compat: True → "< 0"
        mask_spec = f"< {_DEFAULT_MASK_THRESHOLD}"

    if isinstance(mask_spec, np.ndarray):
        if mask_spec.dtype == bool and mask_spec.shape == phi_int_s.shape:
            return mask_spec
        raise ValueError(
            f"mask array must be bool with shape {phi_int_s.shape}, "
            f"got dtype={mask_spec.dtype}, shape={mask_spec.shape}"
        )

    if isinstance(mask_spec, str):
        m = _MASK_RE.match(mask_spec)
        if not m:
            raise ValueError(
                f"Cannot parse mask string {mask_spec!r}. "
                "Expected e.g. '< 0', '< -2e-7', '> 2e-7', '>= 0'."
            )
        op_str, val_str = m.groups()
        thr_si = float(val_str)
        thr_pn = thr_si * PRENORM_PHI1  # convert to pre-normalized units
        ops = {"<": np.less, "<=": np.less_equal,
               ">": np.greater, ">=": np.greater_equal}
        return ops[op_str](phi_int_s, thr_pn)

    raise TypeError(f"Unsupported mask type: {type(mask_spec)}")


def _select_central_blob(
    pv_mask: np.ndarray,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
) -> np.ndarray:
    """Keep only the single connected component enclosing the patch center.

    After threshold masking, multiple disconnected blobs may survive.
    This selects the one that contains the centre pixel (0, 0 in
    relative coordinates).  If the centre pixel is not inside any
    blob (common in Eulerian mode at large ``|dh|``), the blob whose
    nearest boundary pixel is closest to the patch centre is chosen.

    Boundary-touching blobs are handled naturally — they are one
    connected component even if the contour exits the domain.

    Parameters:
        pv_mask: 2-D boolean array (True = inside threshold).
        x_rel: 1-D relative x-coordinates (degrees).
        y_rel: 1-D relative y-coordinates (degrees).

    Returns:
        Boolean 2-D mask with only the selected blob set to True.
    """
    labeled, n_features = _ndimage_label(pv_mask)
    if n_features <= 1:
        return pv_mask  # 0 or 1 blob — nothing to filter

    # Centre pixel indices (0,0 in relative coords → middle of grid)
    ny, nx = pv_mask.shape
    ci = np.argmin(np.abs(y_rel))
    cj = np.argmin(np.abs(x_rel))

    centre_label = labeled[ci, cj]
    if centre_label > 0:
        # Centre is inside a blob — keep that one
        return labeled == centre_label

    # Centre not inside any blob → pick the one whose nearest pixel
    # is closest to (0, 0) in relative-degree space.
    Y2d, X2d = np.meshgrid(y_rel, x_rel, indexing="ij")
    dist2_field = X2d ** 2 + Y2d ** 2  # squared distance from centre
    best_label = 1
    best_dist = np.inf
    for lbl in range(1, n_features + 1):
        blob = labeled == lbl
        min_d2 = float(np.min(dist2_field[blob]))
        if min_d2 < best_dist:
            best_dist = min_d2
            best_label = lbl

    return labeled == best_label


def compute_orthogonal_basis(
    pv_anom: np.ndarray,
    pv_dx: np.ndarray,
    pv_dy: np.ndarray,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    *,
    pv_dx_dy: np.ndarray | None = None,
    pv_dx_dx: np.ndarray | None = None,
    pv_dy_dy: np.ndarray | None = None,
    geopotential: np.ndarray | None = None,
    mask: str | bool | np.ndarray | None = "< 0",
    apply_smoothing: bool = True,
    smoothing_deg: float = 3.0,
    smoothing_method: str = "gaussian",
    grid_spacing: float = 1.5,
    center_lat: float = 60.0,
    apply_lat_weighting: bool = False,
    prenorm_mode: str = "auto",
    pv_anom_prev: np.ndarray | None = None,
    pv_dx_prev: np.ndarray | None = None,
    pv_dy_prev: np.ndarray | None = None,
    pv_dx_dy_prev: np.ndarray | None = None,
    pv_dx_dx_prev: np.ndarray | None = None,
    pv_dy_dy_prev: np.ndarray | None = None,
    interp_alpha: float = 1.0,
    # Deprecated — retained for backward compatibility
    mask_negative: bool | None = None,
    mask_threshold: float | None = None,
) -> OrthogonalBasisFields:
    """Compute six orthogonal basis fields from composite PV fields.

    Parameters:
        pv_anom: PV anomaly q' in SI units.
        pv_dx: Zonal PV gradient ∂q/∂x in SI.
        pv_dy: Meridional PV gradient ∂q/∂y in SI.
        x_rel: 1D relative x-coordinates (degrees).
        y_rel: 1D relative y-coordinates (degrees).
        pv_dx_dy: Pre-computed ∂²q/∂x∂y. Falls back to np.gradient if None.
        pv_dx_dx: Pre-computed ∂²q/∂x². Falls back to np.gradient if None.
        pv_dy_dy: Pre-computed ∂²q/∂y². Falls back to np.gradient if None.
        geopotential: Optional geopotential for overlay plots.
        mask: PV anomaly masking specification.  Accepts:

            * **str** — comparison expression applied to q', e.g.
              ``"< 0"`` (default, blocking), ``"< -2e-7"`` (tight single-
              event), ``"> 2e-7"`` (cyclones).
            * **bool** — ``True`` is equivalent to ``"< 0"``,
              ``False`` disables PV masking.
            * **None** — no PV-anomaly masking (NaN/Inf filtering only).
            * **np.ndarray** (bool, same shape as ``pv_anom``) — a user-
              supplied mask; ``True`` = include in projection.

            Default ``"< 0"`` selects all grid points where q' < 0 PVU.
        apply_smoothing: Apply spatial smoothing after pre-normalization.
        smoothing_deg: Smoothing FWHM (degrees).  Default 3.0°.
        smoothing_method: 'gaussian' or 'fourier'.
        grid_spacing: Grid spacing in degrees.
        center_lat: Approximate latitude of the patch centre (degrees N).
            Used to compute cos(lat)-corrected zonal metric for the
            second-order derivative fallbacks (``∂²q/∂x²``).  Defaults
            to 60.0°N (typical blocking latitude).  Set to 0.0 to
            recover the old equatorial-metric behaviour.
        apply_lat_weighting: Use cos(lat) weighting.
        prenorm_mode: ``"auto"`` (default) for per-event runtime
            normalization, or ``"fixed"`` for legacy PRENORM constants.
        pv_anom_prev: PV anomaly q' at dh−1 (the earlier snapshot). If provided,
            ``pv_dx_prev`` and ``pv_dy_prev`` must also be given.
        pv_dx_prev: Zonal PV gradient at dh−1.
        pv_dy_prev: Meridional PV gradient at dh−1.
        pv_dx_dy_prev: Pre-computed ∂²q/∂x∂y at dh−1.  Optional; when
            given together with ``pv_dx_dx_prev`` and ``pv_dy_dy_prev``,
            the second-order derivatives are also interpolated.  When
            omitted the interpolated first-order gradients are used to
            recompute them via finite differences (less accurate).
        pv_dx_dx_prev: Pre-computed ∂²q/∂x² at dh−1.
        pv_dy_dy_prev: Pre-computed ∂²q/∂y² at dh−1.
        interp_alpha: Interpolation weight for the positional (current-dh) fields.
            Default 1.0 uses only the current-dh fields (no interpolation).
        mask_negative: *Deprecated.* Use ``mask`` instead.
        mask_threshold: *Deprecated.* Use ``mask="< <value>"`` instead.

    Returns:
        OrthogonalBasisFields container.
    """
    # ── Deprecation warnings for old kwargs ──
    if mask_negative is not None:
        warnings.warn(
            "mask_negative is deprecated; use mask='< 0' or mask=False instead.",
            DeprecationWarning, stacklevel=2,
        )
    if mask_threshold is not None:
        warnings.warn(
            "mask_threshold is deprecated; use mask='< <value>' instead.",
            DeprecationWarning, stacklevel=2,
        )
    _mask_spec = mask
    if (mask_negative is not None or mask_threshold is not None) and mask == "< 0":
        _mask_spec = None

    # --- Optional temporal interpolation ---
    _prev_fields = (pv_anom_prev, pv_dx_prev, pv_dy_prev)
    _prev_given = [f is not None for f in _prev_fields]
    if any(_prev_given):
        if not all(_prev_given):
            raise ValueError(
                "Provide all three of pv_anom_prev, pv_dx_prev, "
                "pv_dy_prev, or none of them."
            )
        a = float(interp_alpha)
        pv_anom = a * np.asarray(pv_anom, dtype=np.float64) + \
                  (1.0 - a) * np.asarray(pv_anom_prev, dtype=np.float64)
        pv_dx = a * np.asarray(pv_dx, dtype=np.float64) + \
                (1.0 - a) * np.asarray(pv_dx_prev, dtype=np.float64)
        pv_dy = a * np.asarray(pv_dy, dtype=np.float64) + \
                (1.0 - a) * np.asarray(pv_dy_prev, dtype=np.float64)

        # Interpolate pre-computed 2nd-order derivatives when available
        _prev_d2 = (pv_dx_dy_prev, pv_dx_dx_prev, pv_dy_dy_prev)
        _prev_d2_given = [f is not None for f in _prev_d2]
        if any(_prev_d2_given):
            if not all(_prev_d2_given):
                raise ValueError(
                    "Provide all three of pv_dx_dy_prev, pv_dx_dx_prev, "
                    "pv_dy_dy_prev, or none of them."
                )
            if pv_dx_dy is not None:
                pv_dx_dy = a * np.asarray(pv_dx_dy, dtype=np.float64) + \
                           (1.0 - a) * np.asarray(pv_dx_dy_prev, dtype=np.float64)
            if pv_dx_dx is not None:
                pv_dx_dx = a * np.asarray(pv_dx_dx, dtype=np.float64) + \
                           (1.0 - a) * np.asarray(pv_dx_dx_prev, dtype=np.float64)
            if pv_dy_dy is not None:
                pv_dy_dy = a * np.asarray(pv_dy_dy, dtype=np.float64) + \
                           (1.0 - a) * np.asarray(pv_dy_dy_prev, dtype=np.float64)

    # Step 1: Compute second-order derivative fields (with fallbacks)
    dx_deg = x_rel[1] - x_rel[0] if len(x_rel) > 1 else grid_spacing
    dy_deg = y_rel[1] - y_rel[0] if len(y_rel) > 1 else grid_spacing

    # Meridional metric: dy is latitude-independent
    dy_m = np.deg2rad(abs(dy_deg)) * R_EARTH

    # Zonal metric: dx depends on cos(latitude) at each row
    actual_lats = float(center_lat) + np.asarray(y_rel, dtype=float)
    dx_arr = np.deg2rad(abs(dx_deg)) * R_EARTH * np.cos(np.deg2rad(actual_lats))

    # Diagnostic — shown once per unique call site
    _center_dx_km = float(dx_arr[len(dx_arr) // 2]) / 1e3
    warnings.warn(
        f"compute_orthogonal_basis: grid_spacing={grid_spacing}°, "
        f"center_lat={center_lat}°N → "
        f"dx(center)={_center_dx_km:.1f} km, "
        f"dy={dy_m/1e3:.1f} km",
        stacklevel=2,
    )

    if pv_dx_dy is not None:
        raw_phi_def = np.asarray(pv_dx_dy, dtype=float)
    else:
        raw_phi_def = compute_quadrupole_basis(pv_dx, y_rel)

    if pv_dx_dx is not None:
        _pv_dx_dx = np.asarray(pv_dx_dx, dtype=float)
    else:
        # Per-latitude centred differences with cos(lat)-corrected dx
        _pv_dx_dx = _ddx(np.asarray(pv_dx, dtype=float), dx_arr, periodic=False)

    if pv_dy_dy is not None:
        _pv_dy_dy = np.asarray(pv_dy_dy, dtype=float)
    else:
        _pv_dy_dy = _ddy(np.asarray(pv_dy, dtype=float), dy_m)

    raw_phi_strain = compute_strain_basis(_pv_dx_dx, _pv_dy_dy)
    raw_phi_lap = compute_laplacian_basis(_pv_dx_dx, _pv_dy_dy)

    # Step 2: Pre-normalize
    raw_fields = [pv_anom, pv_dx, pv_dy, raw_phi_def, raw_phi_strain, raw_phi_lap]
    if prenorm_mode == "auto":
        # Compute scales before masking (use finite mask only)
        _finite_pre = np.isfinite(pv_anom)
        pn = auto_prenorm(raw_fields, _finite_pre)
    elif prenorm_mode == "fixed":
        pn = [PRENORM_PHI1, PRENORM_PHI2, PRENORM_PHI3,
              PRENORM_PHI4, PRENORM_PHI5, PRENORM_PHI6]
    else:
        raise ValueError(f"Unknown prenorm_mode: {prenorm_mode!r}")

    phi_int = pv_anom * pn[0]
    phi_dx_pn = pv_dx * pn[1]
    phi_dy_pn = pv_dy * pn[2]
    phi_def = raw_phi_def * pn[3]
    phi_strain = raw_phi_strain * pn[4]
    phi_lap = raw_phi_lap * pn[5]

    # Step 3: Smooth
    fields = [phi_int, phi_dx_pn, phi_dy_pn, phi_def, phi_strain, phi_lap]
    if apply_smoothing:
        if smoothing_method == "gaussian":
            fields = [gaussian_smooth_nan(f, smoothing_deg, grid_spacing) for f in fields]
        elif smoothing_method == "fourier":
            x_ext = float(x_rel.max() - x_rel.min())
            y_ext = float(y_rel.max() - y_rel.min())
            fields = [fourier_lowpass_nan(f, smoothing_deg, x_ext, y_ext) for f in fields]
        else:
            raise ValueError(f"Unknown smoothing_method: {smoothing_method}")
    phi_int_s, phi_dx_s, phi_dy_s, phi_def_s, phi_strain_s, phi_lap_s = fields

    # Step 4: Mask
    finite_mask = (np.isfinite(phi_int_s) & np.isfinite(phi_dx_s) &
                   np.isfinite(phi_dy_s) & np.isfinite(phi_def_s) &
                   np.isfinite(phi_strain_s) & np.isfinite(phi_lap_s))
    pv_mask = _parse_mask_spec(
        _mask_spec, phi_int_s,
        _mask_negative=mask_negative, _mask_threshold=mask_threshold,
    )
    if pv_mask is not None:
        pv_mask = _select_central_blob(pv_mask, x_rel, y_rel)
        mask_arr = finite_mask & pv_mask
    else:
        mask_arr = finite_mask

    # Step 5: Weights
    X_grid, Y_grid = np.meshgrid(x_rel, y_rel)
    if apply_lat_weighting:
        weights = np.cos(np.deg2rad(Y_grid))
        weights = np.where(np.isfinite(weights), weights, 0.0)
    else:
        weights = np.ones_like(Y_grid)
    weights = np.where(mask_arr, weights, 0.0)

    # Step 6: Gram-Schmidt on all 6 bases
    ortho, norms = gram_schmidt_orthogonalize(
        [phi_int_s, phi_dx_s, phi_dy_s, phi_def_s, phi_strain_s, phi_lap_s],
        weights, mask_arr)

    prenorm_dict = {
        "beta": pn[0],
        "ax": pn[1],
        "ay": pn[2],
        "gamma1": pn[3],
        "gamma2": pn[4],
        "sigma": pn[5],
    }

    return OrthogonalBasisFields(
        phi_int=np.asarray(ortho[0], dtype=float),
        phi_dx=np.asarray(ortho[1], dtype=float),
        phi_dy=np.asarray(ortho[2], dtype=float),
        phi_def=np.asarray(ortho[3], dtype=float),
        phi_strain=np.asarray(ortho[4], dtype=float),
        phi_lap=np.asarray(ortho[5], dtype=float),
        weights=weights,
        mask=mask_arr,
        x_rel=np.asarray(x_rel, dtype=float),
        y_rel=np.asarray(y_rel, dtype=float),
        Y_grid=Y_grid,
        geopotential=geopotential,
        raw_phi_int=pv_anom,
        raw_phi_dx=pv_dx,
        raw_phi_dy=pv_dy,
        raw_phi_def=raw_phi_def,
        raw_phi_strain=raw_phi_strain,
        raw_phi_lap=raw_phi_lap,
        norms={
            "beta": norms[0], "ax": norms[1], "ay": norms[2],
            "gamma1": norms[3], "gamma2": norms[4], "sigma": norms[5],
        },
        scale_factors=prenorm_dict,
    )
