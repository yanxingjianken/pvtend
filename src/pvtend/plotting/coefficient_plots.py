"""Time series and lifecycle plots of decomposition coefficients.

Plots β(t), αx(t), αy(t), γ(t) curves for onset/peak/decay stages.
"""

from __future__ import annotations
from typing import Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt


def plot_coefficient_curves(
    dh_values: np.ndarray,
    coefficients: Dict[str, np.ndarray],
    *,
    labels: Dict[str, str] | None = None,
    colors: Dict[str, str] | None = None,
    title: str = "Four-Basis Decomposition Coefficients",
    figsize: tuple[float, float] = (14, 10),
    xlabel: str = "Hours relative to event",
    zero_line: bool = True,
) -> plt.Figure:
    """Plot coefficient time series in a 2×2 panel.

    Parameters:
        dh_values: Hour offsets (x-axis).
        coefficients: Dict with keys 'beta', 'ax', 'ay', 'gamma',
            each an array of length len(dh_values).
        labels: Custom axis labels for each coefficient.
        colors: Custom colors for each coefficient.
        title: Figure title.
        figsize: Figure size.
        xlabel: X-axis label.
        zero_line: Draw horizontal line at y=0.

    Returns:
        Matplotlib Figure.
    """
    default_labels = {
        "beta": r"$\beta$ (intensification) [s$^{-1}$]",
        "ax": r"$\alpha_x$ (zonal propagation) [m/s]",
        "ay": r"$\alpha_y$ (meridional propagation) [m/s]",
        "gamma": r"$\gamma$ (deformation) [s$^{-1}$]",
    }
    default_colors = {
        "beta": "C0",
        "ax": "C1",
        "ay": "C2",
        "gamma": "C3",
    }
    labels = labels or default_labels
    colors = colors or default_colors

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    keys = ["beta", "ax", "ay", "gamma"]

    for ax, key in zip(axes.flat, keys):
        if key not in coefficients:
            ax.set_visible(False)
            continue
        vals = np.asarray(coefficients[key])
        ax.plot(dh_values, vals, "o-", color=colors.get(key, "k"),
                lw=1.5, markersize=4)
        ax.set_ylabel(labels.get(key, key), fontsize=10)
        if zero_line:
            ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.axvline(0, color="gray", ls=":", lw=0.6, alpha=0.4)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=10)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_multi_variant_curves(
    dh_values: np.ndarray,
    variant_coefficients: Dict[str, Dict[str, np.ndarray]],
    *,
    title: str = "Coefficient Comparison",
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """Compare coefficient curves across RWB variants.

    Parameters:
        dh_values: Hour offsets.
        variant_coefficients: Dict of variant_name → coefficient dict.
        title: Figure title.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    keys = ["beta", "ax", "ay", "gamma"]
    ylabels = [r"$\beta$", r"$\alpha_x$", r"$\alpha_y$", r"$\gamma$"]

    for ax, key, ylabel in zip(axes.flat, keys, ylabels):
        for vname, coeffs in variant_coefficients.items():
            if key in coeffs:
                ax.plot(dh_values, coeffs[key], "o-", label=vname,
                        markersize=3, lw=1.2)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.axhline(0, color="gray", ls="--", lw=0.6)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Hours relative to event")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig
