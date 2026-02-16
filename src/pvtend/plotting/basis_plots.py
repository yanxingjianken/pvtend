"""Plot the four orthogonal basis fields.

Generates 4-panel plots of (Φ₁, Φ₂, Φ₃, Φ₄) with optional geopotential
contour overlays, showing the PV anomaly, zonal/meridional gradients,
and quadrupole (deformation) basis.
"""

from __future__ import annotations
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_four_basis(
    phi_int: np.ndarray,
    phi_dx: np.ndarray,
    phi_dy: np.ndarray,
    phi_def: np.ndarray,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    *,
    geopotential: np.ndarray | None = None,
    titles: Sequence[str] | None = None,
    cmap: str = "coolwarm",
    figsize: tuple[float, float] = (16, 12),
    suptitle: str | None = None,
) -> plt.Figure:
    """Plot the four orthogonal basis fields in a 2×2 grid.

    Parameters:
        phi_int: Intensification basis (Φ₁).
        phi_dx: Zonal propagation basis (Φ₂).
        phi_dy: Meridional propagation basis (Φ₃).
        phi_def: Deformation basis (Φ₄).
        x_rel: 1D x-coordinates (relative degrees).
        y_rel: 1D y-coordinates (relative degrees).
        geopotential: Optional geopotential height for contour overlay.
        titles: Panel titles (default: mathematical notation).
        cmap: Colormap name.
        figsize: Figure size.
        suptitle: Super title for the figure.

    Returns:
        Matplotlib Figure.
    """
    if titles is None:
        titles = [
            r"$\Phi_1$: PV anomaly $q'$",
            r"$\Phi_2$: $\partial q / \partial x$",
            r"$\Phi_3$: $\partial q / \partial y$",
            r"$\Phi_4$: $\partial^2 q / \partial x \partial y$",
        ]

    fields = [phi_int, phi_dx, phi_dy, phi_def]
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for ax, field, title in zip(axes.flat, fields, titles):
        vmax = np.nanmax(np.abs(field))
        if vmax < 1e-30:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(
            field, origin="lower", cmap=cmap, norm=norm,
            extent=[x_rel.min(), x_rel.max(), y_rel.min(), y_rel.max()],
            aspect="equal",
        )
        if geopotential is not None:
            ax.contour(
                x_rel, y_rel, geopotential,
                levels=12, colors="k", linewidths=0.6, alpha=0.5,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Δlon (deg)")
        ax.set_ylabel("Δlat (deg)")
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_basis_with_contours(
    basis,  # OrthogonalBasisFields
    *,
    cmap: str = "coolwarm",
    figsize: tuple[float, float] = (16, 12),
    suptitle: str | None = None,
) -> plt.Figure:
    """Convenience wrapper using an OrthogonalBasisFields object.

    Parameters:
        basis: OrthogonalBasisFields from decomposition.basis.
        cmap: Colormap.
        figsize: Figure size.
        suptitle: Super title.

    Returns:
        Matplotlib Figure.
    """
    return plot_four_basis(
        basis.phi_int, basis.phi_dx, basis.phi_dy, basis.phi_def,
        basis.x_rel, basis.y_rel,
        geopotential=basis.geopotential,
        cmap=cmap, figsize=figsize, suptitle=suptitle,
    )
