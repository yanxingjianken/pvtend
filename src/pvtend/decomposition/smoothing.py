"""Spatial smoothing methods for composite fields.

Two smoothing methods:
1. Gaussian smoothing (recommended for non-periodic domains)
2. Fourier low-pass filtering (assumes periodicity)

Both handle NaN values via normalised convolution or fill-before-FFT.
"""

from __future__ import annotations

import numpy as np


def gaussian_smooth_nan(
    field: np.ndarray,
    smoothing_deg: float = 3.0,
    grid_spacing: float = 1.5,
) -> np.ndarray:
    """Apply Gaussian smoothing, handling NaN values properly.

    Recommended for non-periodic domains (composite patches).

    Parameters:
        field: 2D input field with potential NaN values.
        smoothing_deg: Smoothing FWHM in degrees.
        grid_spacing: Grid spacing in degrees.

    Returns:
        Smoothed 2D field with NaN positions preserved.
    """
    from scipy.ndimage import gaussian_filter

    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2:
        return arr

    mask = np.isfinite(arr)
    if not mask.any():
        return np.full_like(arr, np.nan)

    sigma_deg = smoothing_deg / 2.355  # FWHM → sigma
    sigma_gridpts = sigma_deg / grid_spacing

    arr_filled = arr.copy()
    arr_filled[~mask] = float(np.nanmean(arr))

    smoothed = gaussian_filter(arr_filled, sigma=sigma_gridpts, mode="reflect")
    smoothed[~mask] = np.nan
    return smoothed


def fourier_lowpass_nan(
    field: np.ndarray,
    smoothing_deg: float = 1.5,
    x_extent: float = 72.0,
    y_extent: float = 42.0,
) -> np.ndarray:
    """Fourier low-pass filter handling NaN values.

    WARNING: Assumes periodicity — may cause edge artifacts on
    non-periodic domains. Use gaussian_smooth_nan for composites.

    Parameters:
        field: 2D input field.
        smoothing_deg: Minimum wavelength to retain (degrees).
        x_extent: Domain x-extent in degrees.
        y_extent: Domain y-extent in degrees.

    Returns:
        Low-pass filtered 2D field.
    """
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2:
        return arr
    mask = np.isfinite(arr)
    if not mask.any():
        return np.full_like(arr, np.nan)

    kmax = x_extent / smoothing_deg if smoothing_deg > 0 else float("inf")
    lmax = y_extent / smoothing_deg if smoothing_deg > 0 else float("inf")

    mean_val = float(np.nanmean(arr[mask]))
    centered = np.where(mask, arr - mean_val, 0.0)

    F = np.fft.rfftn(centered)
    ny, nx = arr.shape
    ky = np.fft.fftfreq(ny, d=1.0) * ny
    kx = np.fft.rfftfreq(nx, d=1.0) * nx
    KY, KX = np.meshgrid(ky, kx, indexing="ij")

    keep = (np.abs(KX) <= kmax) & (np.abs(KY) <= lmax)
    F_filtered = F * keep

    smoothed = np.fft.irfftn(F_filtered, s=arr.shape) + mean_val
    smoothed = np.where(mask, smoothed, np.nan)
    return smoothed
