"""Single-variable composite explorer with bootstrap significance.

Provides :func:`plot_var` — a self-contained function that loads NPZ
composites, computes bootstrap significance, and optionally projects
onto the dh−1 orthogonal basis.

Ported from ``examples/04_single_var_composite.ipynb`` so that any
notebook (including 07) can reuse it via::

    from pvtend.plotting import plot_var
"""

from __future__ import annotations

import os
import glob
from concurrent.futures import ThreadPoolExecutor
from zipfile import BadZipFile
from typing import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pvtend import compute_orthogonal_basis, project_field
from pvtend.decomposition.smoothing import gaussian_smooth_nan


# ── I/O helpers ──────────────────────────────────────────────────────────────


def _load_npz(path: str) -> dict | None:
    try:
        return dict(np.load(path))
    except (BadZipFile, EOFError, OSError):
        return None


def load_events(
    data_root: str,
    stage: str,
    dh: int,
    *,
    max_workers: int = 8,
) -> list[dict]:
    """Load all event dicts from ``data_root/stage/dh=±<dh>/track_*.npz``."""
    sign = "+" if dh >= 0 else ""
    d = f"{data_root}/{stage}/dh={sign}{dh}"
    files = sorted(glob.glob(os.path.join(d, "track_*.npz")))
    if not files:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_load_npz, files))
    good = [r for r in results if r is not None]
    n_bad = len(results) - len(good)
    if n_bad:
        print(f"  ⚠ dh={dh}: skipped {n_bad} corrupt/incomplete NPZ files")
    return good


# ── Field extraction ─────────────────────────────────────────────────────────


def get_field(event: dict, name: str, level: str | int = "wavg") -> np.ndarray:
    """Extract a single named 2-D field from one event dict.

    Args:
        event: One NPZ event dict.
        name: Raw NPZ key (e.g. ``"pv_anom"``).
        level: ``"wavg"`` for the pre-computed 2-D weighted average, or an
            integer pressure in hPa to slice the ``_3d`` array.

    Returns:
        2-D array (nlat, nlon).
    """
    if isinstance(level, str) and level.lower() == "wavg":
        return event[name].copy()
    key_3d = name if name.endswith("_3d") else name + "_3d"
    arr_3d = event[key_3d]
    lvls = event["levels"]
    idx = int(np.argmin(np.abs(lvls - float(level))))
    return arr_3d[idx].copy()


def _resolve_var_spec(
    event: dict,
    var_spec: str | list[str] | Callable,
    level: str | int = "wavg",
) -> np.ndarray:
    """Evaluate *var_spec* for one event → 2-D array.

    Supports three forms:

    * **callable** ``f(event) → 2-D``: full user control.
    * **str**: single NPZ key, optionally prefixed with ``'-'`` for negation.
    * **list[str]**: multiple fields summed, each optionally ``'-'`` prefixed.
    """
    if callable(var_spec):
        return var_spec(event)

    if isinstance(var_spec, str):
        var_spec = [var_spec]

    total = None
    for vs in var_spec:
        negate = vs.startswith("-")
        name = vs.lstrip("-").strip()
        arr = get_field(event, name, level)
        if negate:
            arr = -arr
        total = arr.copy() if total is None else total + arr
    return total


# ── Bootstrap ────────────────────────────────────────────────────────────────


def bootstrap_sig(
    events: list[dict],
    var_spec: str | list[str] | Callable,
    level: str | int = "wavg",
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite mean + bootstrap significance mask.

    Args:
        events: List of event dicts.
        var_spec: Field specification (str, list[str], or callable).
        level: ``"wavg"`` or int hPa.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level.
        seed: Random seed.

    Returns:
        ``(mean_2d, sig_mask)`` — *sig_mask* is ``True`` where the
        bootstrap CI excludes zero.
    """
    stack = np.array([_resolve_var_spec(e, var_spec, level) for e in events])
    N = stack.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, *stack.shape[1:]))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boot[b] = np.nanmean(stack[idx], axis=0)
    lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)
    mean = np.nanmean(stack, axis=0)
    sig_mask = ~((lo <= 0) & (hi >= 0))
    return mean, sig_mask


# ── Main plot function ───────────────────────────────────────────────────────


def plot_var(
    var_spec: str | list[str] | Callable,
    *,
    data_root: str,
    stage: str = "onset",
    dh: int = 0,
    level: str | int = "wavg",
    projection: bool = False,
    smooth_deg: float = 3.0,
    grid_sp: float = 1.5,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    pv_contour: float = 0.0,
    n_contour: int = 21,
    figsize_scale: float = 1.0,
    label: str | None = None,
    vmax: float | None = None,
    use_sig_mask: bool = True,
    mask_negative: bool = True,
) -> dict:
    """Composite + bootstrap plot for any variable(s).

    This is the main entry point, equivalent to ``plot_var()`` from
    ``04_single_var_composite.ipynb`` but usable from any notebook.

    Args:
        var_spec: Field specification — ``str``, ``list[str]`` (with optional
            ``'-'`` prefix for negation), or ``callable(event) → 2-D array``.
        data_root: Path to the composite NPZ archive (e.g.
            ``".../composite_blocking_tempest"``).
        stage: Lifecycle stage (``"onset"``, ``"peak"``, ``"decay"``).
        dh: Lifecycle hour offset.
        level: ``"wavg"`` or int hPa.
        projection: If ``True``, add 2×2 projection rows (INT/PRP/DEF/Resid).
        smooth_deg: Gaussian smoothing FWHM (degrees).
        grid_sp: Grid spacing (degrees).
        n_boot: Bootstrap resamples.
        alpha: Significance level.
        seed: Random seed.
        pv_contour: PV anomaly contour level for overlay.
        n_contour: Number of contourf levels.
        figsize_scale: Scale factor for figure size.
        label: Custom title label (auto-generated if ``None``).
        vmax: Fixed colour-scale max.  When ``None`` (default) the 95th
            percentile of ``|mean|`` is used.  Pass the same value to
            multiple calls to enforce a shared colour range.

    Returns:
        Result dict with key ``"vmax"`` (the colour-range used) and,
        when ``projection=True``, all projection keys
        (``beta``, ``ax``, ``ay``, ``gamma``, etc.).
    """
    _smooth = lambda f: gaussian_smooth_nan(
        f, smoothing_deg=smooth_deg, grid_spacing=grid_sp
    )

    # ── 1. Load events ──
    evs = load_events(data_root, stage, dh)
    N = len(evs)
    if N == 0:
        print(f"No events at dh={dh}")
        return None
    X_rel = evs[0]["X_rel"]
    Y_rel = evs[0]["Y_rel"]
    x_rel = X_rel[0, :]
    y_rel = Y_rel[:, 0]

    # ── 2. Composite mean + bootstrap ──
    print(f"Computing bootstrap (N={N}, n_boot={n_boot}) ...")
    mean_fld, sig_mask = bootstrap_sig(
        evs, var_spec, level, n_boot=n_boot, alpha=alpha, seed=seed,
    )
    pct_sig = 100 * np.mean(sig_mask)
    print(f"  {pct_sig:.1f}% significant at {100*(1-alpha):.0f}%")

    # PV anomaly composite for contour
    pv_anom_mean = np.nanmean(
        [get_field(e, "pv_anom", level) for e in evs], axis=0,
    )

    # ── Label construction ──
    if label is not None:
        var_label = label
    elif callable(var_spec):
        var_label = "custom λ"
    elif isinstance(var_spec, list):
        var_label = " + ".join(var_spec)
    else:
        var_label = var_spec
    level_str = "wavg" if isinstance(level, str) else f"{level} hPa"

    # ── 3. Projection (if requested) ──
    proj = None
    if projection:
        dh_basis = max(dh - 1, -13)
        evs_b = load_events(data_root, stage, dh_basis) if dh_basis != dh else evs
        pv_b = np.nanmean([get_field(e, "pv_anom", level) for e in evs_b], axis=0)
        dx_b = np.nanmean([get_field(e, "pv_dx", level) for e in evs_b], axis=0)
        dy_b = np.nanmean([get_field(e, "pv_dy", level) for e in evs_b], axis=0)

        # dh composite means for temporal interpolation
        pv_n = np.nanmean([get_field(e, "pv_anom", level) for e in evs], axis=0)
        dx_n = np.nanmean([get_field(e, "pv_dx", level) for e in evs], axis=0)
        dy_n = np.nanmean([get_field(e, "pv_dy", level) for e in evs], axis=0)

        basis = compute_orthogonal_basis(
            pv_n, dx_n, dy_n, x_rel, y_rel,
            mask_negative=mask_negative,
            apply_smoothing=True,
            smoothing_deg=smooth_deg,
            grid_spacing=grid_sp,
            pv_anom_prev=pv_b,
            pv_dx_prev=dx_b,
            pv_dy_prev=dy_b,
            interp_alpha=1,
        )

        if use_sig_mask:
            field_sig = np.where(sig_mask, mean_fld, 0.0)
        else:
            field_sig = mean_fld.copy()
        field_sig_s = _smooth(field_sig)
        proj = project_field(field_sig_s, basis)
        print(
            f"  Projection (sig-only): "
            f"β={proj['beta']:.3e}  αx={proj['ax']:.3f}  "
            f"αy={proj['ay']:.3f}  γ={proj['gamma']:.3e}"
        )

    # ── 4. Plot ──
    n_rows = 3 if projection else 1
    fig = plt.figure(
        figsize=(14 * figsize_scale, 5 * n_rows * figsize_scale),
    )
    gs = GridSpec(n_rows, 2, figure=fig, hspace=0.35, wspace=0.25)

    if vmax is None:
        vmax = np.nanpercentile(np.abs(mean_fld), 95)
    vmax = max(vmax, 1e-30)
    clevels = np.linspace(-vmax, vmax, n_contour)

    # Row 1, left: composite mean
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.contourf(
        X_rel, Y_rel, mean_fld, levels=clevels,
        cmap="RdBu_r", extend="both",
    )
    ax0.contour(
        X_rel, Y_rel, pv_anom_mean,
        levels=[pv_contour], colors="white", linewidths=2.5,
    )
    ax0.set_title(f"Composite Mean  (N={N})", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Y (deg)")
    ax0.set_xlabel("X (deg)")
    ax0.set_aspect("equal")
    plt.colorbar(im0, ax=ax0, shrink=0.85)

    # Row 1, right: bootstrap significance
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.contourf(
        X_rel, Y_rel, mean_fld, levels=clevels,
        cmap="RdBu_r", extend="both",
    )
    ax1.contour(
        X_rel, Y_rel, pv_anom_mean,
        levels=[pv_contour], colors="white", linewidths=2.5,
    )
    ax1.contourf(
        X_rel, Y_rel, (~sig_mask).astype(float),
        levels=[0.5, 1.5], hatches=["xxx"], colors="none", zorder=5,
    )
    ax1.set_title(
        f"Bootstrap sig ({100*(1-alpha):.0f}%)  "
        f"Hatch=n.s.  ({pct_sig:.0f}% sig)",
        fontsize=11, fontweight="bold",
    )
    ax1.set_xlabel("X (deg)")
    ax1.set_aspect("equal")
    plt.colorbar(im1, ax=ax1, shrink=0.85)

    # Rows 2-3: projection (2×2)
    if projection and proj is not None:
        panels = [
            ("INT (β · Φ₁)", proj["int"]),
            ("PRP (αx·Φ₂ + αy·Φ₃)", proj["prop"]),
            ("DEF (γ · Φ₄)", proj["def"]),
            ("Residual", proj["resid"]),
        ]
        all_abs = np.concatenate([
            np.abs(p[np.isfinite(p)]) for _, p in panels if np.any(np.isfinite(p))
        ])
        vmax_p = float(np.percentile(all_abs, 95)) if all_abs.size else 1e-30
        vmax_p = max(vmax_p, 1e-30)
        clev_p = np.linspace(-vmax_p, vmax_p, n_contour)

        coef_txt = (
            f"β={proj['beta']:.3e} s⁻¹   "
            f"αx={proj['ax']:.3f} m/s   "
            f"αy={proj['ay']:.3f} m/s   "
            f"γ={proj['gamma']:.3e} s⁻¹   "
            f"RMSE/max={proj['rmse']/(np.nanmax(np.abs(mean_fld))+1e-30):.3f}"
        )

        for idx, (lbl, field) in enumerate(panels):
            row = 1 + idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])
            im = ax.contourf(
                X_rel, Y_rel, field, levels=clev_p,
                cmap="RdBu_r", extend="both",
            )
            ax.contour(
                X_rel, Y_rel, pv_anom_mean,
                levels=[pv_contour], colors="white", linewidths=2.0,
            )
            ax.set_title(lbl, fontsize=10, fontweight="bold")
            ax.set_xlabel("X (deg)")
            if col == 0:
                ax.set_ylabel("Y (deg)")
            ax.set_aspect("equal")
            plt.colorbar(im, ax=ax, shrink=0.75)

        fig.text(
            0.5, 1 - 1.0 / n_rows - 0.01, coef_txt,
            ha="center", fontsize=10, fontstyle="italic",
            transform=fig.transFigure,
        )

    sign_str = "+" if dh >= 0 else ""
    fig.suptitle(
        f"{stage}  dh={sign_str}{dh}   Level={level_str}   N={N}\n"
        f"Field: {var_label}",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    plt.show()

    result = {"vmax": vmax}
    if proj is not None:
        result.update(proj)
    return result
