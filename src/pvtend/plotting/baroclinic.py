"""Baroclinic structure visualisation.

Provides :func:`plot_baroclinic_tilt` — a two-level v′ overlay showing
the westward tilt with height characteristic of baroclinic blocking.

Ported from ``examples/06_baroclinic_structure.ipynb``.

Usage::

    from pvtend.plotting import plot_baroclinic_tilt
    plot_baroclinic_tilt(
        data_root="/net/flood/data2/users/x_yan/composite_blocking_tempest",
        stages=["peak"],
    )
"""

from __future__ import annotations

import os
import glob
from concurrent.futures import ThreadPoolExecutor
from zipfile import BadZipFile
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt


# ── I/O helpers ──────────────────────────────────────────────────────────────


def _load_npz(path: str) -> dict | None:
    try:
        return dict(np.load(path))
    except (BadZipFile, EOFError, OSError):
        return None


# ── Main function ────────────────────────────────────────────────────────────


def plot_baroclinic_tilt(
    *,
    data_root: str,
    stages: Sequence[str] = ("onset", "peak", "decay"),
    dh: int = 0,
    upper_hPa: int = 250,
    lower_hPa: int = 850,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot v′ at two pressure levels to reveal baroclinic tilt.

    Upper level is shown as bold black contours (solid = v′ > 0,
    dashed = v′ < 0, only where significant).  Lower level is shown as
    blue-red shading with light hatching where not significant.

    Args:
        data_root: Path to the composite NPZ archive.
        stages: Lifecycle stages to plot (one panel per stage).
        dh: Lifecycle hour offset.
        upper_hPa: Upper-level pressure (contours).
        lower_hPa: Lower-level pressure (shading).
        n_boot: Bootstrap resamples.
        alpha: Significance level.
        seed: Random seed.
        figsize: Figure size; auto-calculated if ``None``.
    """
    n_stages = len(stages)
    if figsize is None:
        figsize = (6 * n_stages, 5.5)

    # ── Discover level indices from one sample ──
    sign = "+" if dh >= 0 else ""
    sample_dir = f"{data_root}/{stages[0]}/dh={sign}{dh}"
    sample_f = sorted(glob.glob(os.path.join(sample_dir, "track_*.npz")))[0]
    _s = np.load(sample_f)
    levels = _s["levels"].astype(float)
    X_rel = _s["X_rel"]
    Y_rel = _s["Y_rel"]
    idx_upper = int(np.argmin(np.abs(levels - upper_hPa)))
    idx_lower = int(np.argmin(np.abs(levels - lower_hPa)))
    del _s
    print(
        f"Level indices: {upper_hPa} hPa → {idx_upper} "
        f"({levels[idx_upper]:.0f}), {lower_hPa} hPa → {idx_lower} "
        f"({levels[idx_lower]:.0f})"
    )

    def _load_vanom_2lev(f):
        try:
            d = np.load(f)
        except (BadZipFile, EOFError, OSError):
            return None
        v = d["v_anom_3d"]
        return v[idx_upper], v[idx_lower]

    # ── Load + bootstrap per stage ──
    vanom_data: dict[str, dict] = {}
    for stg in stages:
        npz_dir = f"{data_root}/{stg}/dh={sign}{dh}"
        files = sorted(glob.glob(os.path.join(npz_dir, "track_*.npz")))
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(_load_vanom_2lev, files))
        results = [r for r in results if r is not None]
        v_up = np.array([r[0] for r in results])
        v_lo = np.array([r[1] for r in results])
        N_ev = v_up.shape[0]
        print(f"  {stg}: N={N_ev}, bootstrapping ...")

        rng = np.random.default_rng(seed)
        boot_up = np.empty((n_boot, *v_up.shape[1:]))
        boot_lo = np.empty((n_boot, *v_lo.shape[1:]))
        for b in range(n_boot):
            idx = rng.integers(0, N_ev, size=N_ev)
            boot_up[b] = np.nanmean(v_up[idx], axis=0)
            boot_lo[b] = np.nanmean(v_lo[idx], axis=0)

        def _sig(boot):
            lo = np.percentile(boot, 100 * alpha / 2, axis=0)
            hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
            return ~((lo <= 0) & (hi >= 0))

        vanom_data[stg] = dict(
            mean_upper=np.nanmean(v_up, axis=0),
            sig_upper=_sig(boot_up),
            mean_lower=np.nanmean(v_lo, axis=0),
            sig_lower=_sig(boot_lo),
            n=N_ev,
        )

    # ── Shared colour scales ──
    all_lo = np.concatenate(
        [np.abs(vanom_data[s]["mean_lower"]).ravel() for s in stages]
    )
    all_up = np.concatenate(
        [np.abs(vanom_data[s]["mean_upper"]).ravel() for s in stages]
    )
    vmax_lo = max(np.nanpercentile(all_lo, 95), 1e-6)
    vmax_up = max(np.nanpercentile(all_up, 95), 1e-6)
    clev_lo = np.linspace(-vmax_lo, vmax_lo, 25)
    clev_up = np.linspace(-vmax_up, vmax_up, 11)
    clev_up = clev_up[clev_up != 0]

    # ── Plot ──
    fig, axes = plt.subplots(1, n_stages, figsize=figsize, sharey=True)
    if n_stages == 1:
        axes = [axes]

    for ax, stg in zip(axes, stages):
        d = vanom_data[stg]
        m_lo, m_up = d["mean_lower"], d["mean_upper"]
        sig_lo, sig_up = d["sig_lower"], d["sig_upper"]

        cf = ax.contourf(
            X_rel, Y_rel, m_lo, levels=clev_lo,
            cmap="RdBu_r", extend="both", alpha=0.55,
        )
        ax.contour(
            X_rel, Y_rel, m_lo,
            levels=clev_lo[clev_lo != 0], colors="0.55", linewidths=0.4,
        )
        ax.contourf(
            X_rel, Y_rel, (~sig_lo).astype(float),
            levels=[0.5, 1.5], hatches=["...."], colors="none",
            alpha=0.0, zorder=3,
        )

        m_up_sig = np.where(sig_up, m_up, np.nan)
        pos_lev = clev_up[clev_up > 0]
        neg_lev = clev_up[clev_up < 0]
        if pos_lev.size:
            ax.contour(
                X_rel, Y_rel, m_up_sig, levels=pos_lev,
                colors="k", linewidths=1.6, linestyles="solid",
            )
        if neg_lev.size:
            ax.contour(
                X_rel, Y_rel, m_up_sig, levels=neg_lev,
                colors="k", linewidths=1.6, linestyles="dashed",
            )

        ax.set_title(
            f"{stg.capitalize()}  (N={d['n']})",
            fontsize=12, fontweight="bold",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("Rel. lon [°]")

    axes[0].set_ylabel("Rel. lat [°]")

    cbar = fig.colorbar(
        cf, ax=list(axes), shrink=0.85, pad=0.02,
        label=f"v′ [m s⁻¹]  (shading: {lower_hPa} hPa)",
    )
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"v′ anomaly — shading: {lower_hPa} hPa  |  "
        f"black contours: {upper_hPa} hPa (sig. only)\n"
        f"Dots = {lower_hPa} insig.  Contours omitted where {upper_hPa} insig.  "
        f"({100*(1-alpha):.0f}% CI, N_boot={n_boot})",
        fontsize=12, y=1.04,
    )
    plt.show()
