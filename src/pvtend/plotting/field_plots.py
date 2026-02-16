"""2D field visualization utilities.

General-purpose plotting for PV, wind, omega, and geopotential fields
on event-centred patches.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_field_2d(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str = "",
    cmap: str = "coolwarm",
    symmetric: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    contour_field: np.ndarray | None = None,
    contour_levels: int = 12,
    colorbar_label: str = "",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Axes:
    """Plot a 2D field on relative coordinates.

    Parameters:
        field: 2D array to plot.
        x: 1D x-coordinates.
        y: 1D y-coordinates.
        title: Plot title.
        cmap: Colormap name.
        symmetric: Force symmetric color limits around zero.
        vmin, vmax: Manual color limits (override symmetric).
        contour_field: Optional field for contour overlay.
        contour_levels: Number of contour levels.
        colorbar_label: Label for the colorbar.
        ax: Existing axes (creates new figure if None).
        figsize: Figure size (used only if ax is None).

    Returns:
        The matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if symmetric and vmin is None and vmax is None:
        v = np.nanmax(np.abs(field))
        vmin, vmax = -v, v

    if symmetric and vmin is not None and vmax is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = None

    extent = [x.min(), x.max(), y.min(), y.max()]
    im = ax.imshow(field, origin="lower", extent=extent, aspect="equal",
                   cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label=colorbar_label)

    if contour_field is not None:
        ax.contour(x, y, contour_field, levels=contour_levels,
                   colors="k", linewidths=0.5, alpha=0.5)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Δlon (deg)")
    ax.set_ylabel("Δlat (deg)")
    return ax


def plot_wind_overlay(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    scalar_field: np.ndarray | None = None,
    skip: int = 2,
    scale: float = 50.0,
    title: str = "",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Axes:
    """Plot wind vectors (quiver) with optional scalar background.

    Parameters:
        u, v: Wind components (2D arrays).
        x, y: 1D coordinate arrays.
        scalar_field: Optional background scalar field.
        skip: Subsample factor for quiver arrows.
        scale: Quiver scale parameter.
        title: Plot title.
        ax: Existing axes.
        figsize: Figure size.

    Returns:
        Matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if scalar_field is not None:
        plot_field_2d(scalar_field, x, y, ax=ax)

    X, Y = np.meshgrid(x, y)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip],
              scale=scale, alpha=0.7, zorder=5)
    ax.set_title(title, fontsize=11)
    return ax
