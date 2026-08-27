#!/usr/bin/env python3
"""Generate 2-panel composite lifecycle GIFs from pvtend NPZ files.

For each RWB variant, creates an animated GIF with 25 frames (t00–t24):
  Left panel:  250 hPa PV anomaly shading + geopotential Z contours
  Right panel: PV tendency (pv_dt) shading + covariance ellipse + major axis

Title includes event count, mean±std timestep size, and tilt angle.
Colorbars are FIXED across all frames (global scale from pre-scan).

Usage:
  python lifecycle_composite_gif.py [--variant onset_awb_decay_cwb] [--level 250]
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import BadZipFile

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import numpy as np
from scipy import ndimage


# ═══════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════
OUT_ROOT = Path(__file__).resolve().parent.parent
N_TIMESTAMPS = 25
LEVEL = 250  # hPa for 3D fields
SMOOTH_DEG = 3.0
GRID_SP = 1.5
GIF_DPI = 120
GIF_DURATION = 0.6  # seconds per frame


def _load_npz(path):
    """Load one NPZ file, return dict or None on error."""
    try:
        return dict(np.load(path))
    except (BadZipFile, EOFError, OSError, ValueError):
        return None


def load_events_for_dh(variant_dir, dh):
    """Load all NPZ files for a restructured dh directory."""
    d = os.path.join(variant_dir, f"dh=+{dh}")
    files = sorted(glob.glob(os.path.join(d, "track_*.npz")))
    if not files:
        return []
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_load_npz, files))
    good = [r for r in results if r is not None]
    n_bad = len(results) - len(good)
    if n_bad:
        print(f"  ⚠ dh=+{dh}: skipped {n_bad} corrupt NPZ files")
    return good


def get_3d_field(event, name, level=LEVEL):
    """Extract a 2-D slice from a _3d array at a given pressure level."""
    key = name if name.endswith("_3d") else name + "_3d"
    if key not in event:
        if name in event:
            return event[name]
        raise KeyError(f"Neither {key} nor {name} found in NPZ")
    arr_3d = event[key]
    lvls = event["levels"]
    idx = int(np.argmin(np.abs(lvls - float(level))))
    return arr_3d[idx]


def gaussian_smooth(field, sigma_deg=SMOOTH_DEG, grid_spacing=GRID_SP):
    """Simple Gaussian smoothing with nan handling."""
    sigma_px = sigma_deg / grid_spacing
    filled = np.where(np.isnan(field), 0.0, field)
    weights = np.where(np.isnan(field), 0.0, 1.0)
    s_filled = ndimage.gaussian_filter(filled, sigma=sigma_px)
    s_weights = ndimage.gaussian_filter(weights, sigma=sigma_px)
    s_weights = np.where(s_weights < 1e-10, np.nan, s_weights)
    return s_filled / s_weights


def select_central_blob(pv_anom, x_rel, y_rel):
    """Reproduce pvtend's _select_central_blob: q'<0 blob nearest centre."""
    mask = pv_anom < 0
    labeled, n_feat = ndimage.label(mask)
    if n_feat <= 1:
        return mask

    ci = int(np.argmin(np.abs(y_rel)))
    cj = int(np.argmin(np.abs(x_rel)))
    centre_label = labeled[ci, cj]

    if centre_label > 0:
        return labeled == centre_label

    X2d, Y2d = np.meshgrid(x_rel, y_rel)
    dist2 = X2d**2 + Y2d**2
    best_label, best_dist = 1, np.inf
    for lbl in range(1, n_feat + 1):
        d = dist2[labeled == lbl].min()
        if d < best_dist:
            best_dist = d
            best_label = lbl
    return labeled == best_label


def mask_ellipse_angle(mask, X2d, Y2d):
    """Fit covariance ellipse to binary mask.

    Returns (tilt_deg, semi_major, semi_minor, cx, cy, Ellipse_patch).
    Tilt angle ∈ (-90, 90], positive = CCW from east.
    """
    if mask.sum() < 3:
        return np.nan, 0.0, 0.0, 0.0, 0.0, None

    x = X2d[mask]
    y = Y2d[mask]
    cx, cy = float(x.mean()), float(y.mean())
    dx, dy = x - cx, y - cy

    Cxx = float((dx**2).mean())
    Cyy = float((dy**2).mean())
    Cxy = float((dx * dy).mean())

    eigvals, eigvecs = np.linalg.eigh(np.array([[Cxx, Cxy], [Cxy, Cyy]]))
    eigvals = np.clip(eigvals, 0, None)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    angle = ((angle + 90.0) % 180.0) - 90.0

    semi_major = 2.0 * np.sqrt(eigvals[0])
    semi_minor = 2.0 * np.sqrt(eigvals[1])

    ell = Ellipse(
        (cx, cy), width=2 * semi_major, height=2 * semi_minor,
        angle=angle, fill=False, edgecolor="lime",
        linewidth=2.0, linestyle="--", zorder=10,
    )
    return angle, semi_major, semi_minor, cx, cy, ell


def compute_lifecycle_stats(variant_dir):
    """Compute Δt stats from events CSV."""
    csv_path = os.path.join(variant_dir, "events_lifecycle.csv")
    if not os.path.exists(csv_path):
        return 0, 0.0, 0.0

    import pandas as pd
    df = pd.read_csv(csv_path)
    df["base_ts"] = pd.to_datetime(df["base_ts"])

    durations = []
    for tid, grp in df.groupby("track_id"):
        ts_sorted = grp["base_ts"].sort_values()
        if len(ts_sorted) >= 2:
            dur_h = (ts_sorted.iloc[-1] - ts_sorted.iloc[0]).total_seconds() / 3600.0
            durations.append(dur_h)

    durations = np.array(durations)
    n_events = len(durations)
    dt_per_step = durations / (N_TIMESTAMPS - 1)
    return n_events, dt_per_step.mean(), dt_per_step.std()


def make_gif(variant_name, level=LEVEL):
    """Generate a 2-panel lifecycle GIF for one variant."""
    variant_dir = str(OUT_ROOT / variant_name)
    gif_path = str(OUT_ROOT / f"{variant_name}_lifecycle_{level}hPa.gif")

    print(f"\n{'='*60}")
    print(f"  Generating GIF: {variant_name}")
    print(f"{'='*60}")

    # Lifecycle stats
    n_events, dt_mean, dt_std = compute_lifecycle_stats(variant_dir)
    print(f"  Events: N={n_events}, Δt={dt_mean:.2f}±{dt_std:.2f} h/step")

    # Check available dh directories
    available_dh = []
    for dh in range(N_TIMESTAMPS):
        d = os.path.join(variant_dir, f"dh=+{dh}")
        if os.path.isdir(d) and glob.glob(os.path.join(d, "track_*.npz")):
            available_dh.append(dh)
    print(f"  Available dh steps: {len(available_dh)} / {N_TIMESTAMPS}")

    if len(available_dh) < 3:
        print(f"  ERROR: Too few dh steps ({len(available_dh)}), skipping GIF")
        return

    # ══════════════════════════════════════════════════════
    #  Pass 1: compute composites & determine global scales
    # ══════════════════════════════════════════════════════
    print("  Pass 1: computing composites...")
    composites = {}
    for dh in range(N_TIMESTAMPS):
        if dh not in available_dh:
            continue
        events = load_events_for_dh(variant_dir, dh)
        if not events:
            continue

        pv_anom_comp = np.nanmean(
            [get_3d_field(e, "pv_anom", level) for e in events], axis=0)
        z_comp = np.nanmean(
            [get_3d_field(e, "z", level) for e in events], axis=0)
        pv_dt_comp = np.nanmean(
            [get_3d_field(e, "pv_dt", level) for e in events], axis=0)

        x_rel = events[0]["X_rel"]
        y_rel = events[0]["Y_rel"]
        if x_rel.ndim == 2:
            x_rel = x_rel[0, :]
        if y_rel.ndim == 2:
            y_rel = y_rel[:, 0]

        composites[dh] = dict(
            pv_anom_s=gaussian_smooth(pv_anom_comp),
            z_s=gaussian_smooth(z_comp),
            pv_dt_s=gaussian_smooth(pv_dt_comp),
            x_rel=x_rel, y_rel=y_rel,
            n_loaded=len(events),
        )

    # Global fixed color scales (in PVU and PVU/s after ×1e6)
    all_pv = np.concatenate(
        [np.abs(c["pv_anom_s"]).ravel() for c in composites.values()])
    all_dt = np.concatenate(
        [np.abs(c["pv_dt_s"]).ravel() for c in composites.values()])
    global_pv_vmax = float(np.nanpercentile(all_pv, 98)) * 1e6
    global_dt_vmax = float(np.nanpercentile(all_dt, 98)) * 1e6
    global_pv_vmax = max(global_pv_vmax, 1e-15)
    global_dt_vmax = max(global_dt_vmax, 1e-15)
    print(f"  Global scales: PV' ±{global_pv_vmax:.4e} PVU, "
          f"dq/dt ±{global_dt_vmax:.4e} PVU/s")

    # ══════════════════════════════════════════════════════
    #  Pass 2: render frames with fixed colorbars
    # ══════════════════════════════════════════════════════
    print("  Pass 2: rendering frames...")
    frames = []
    for dh in range(N_TIMESTAMPS):
        if dh not in composites:
            print(f"  ⚠ dh=+{dh} missing, inserting blank frame")
            frames.append(np.zeros((720, 1680, 3), dtype=np.uint8))
            continue

        c = composites[dh]
        n_loaded = c["n_loaded"]
        pct = 100.0 * dh / (N_TIMESTAMPS - 1)

        x_rel, y_rel = c["x_rel"], c["y_rel"]
        X2d, Y2d = np.meshgrid(x_rel, y_rel)
        pv_anom_s = c["pv_anom_s"]
        z_s = c["z_s"]
        pv_dt_s = c["pv_dt_s"]

        # Center mask & ellipse fit
        blob_mask = select_central_blob(pv_anom_s, x_rel, y_rel)
        tilt_deg, smaj, smin, ecx, ecy, ell_patch = mask_ellipse_angle(
            blob_mask, X2d, Y2d)

        # ── Create 2-panel figure ──
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.subplots_adjust(
            top=0.82, bottom=0.10, left=0.05, right=0.95, wspace=0.30)

        # ── Left panel: PV anomaly + Z contours ──
        im1 = ax1.pcolormesh(
            X2d, Y2d, pv_anom_s * 1e6,
            cmap="RdBu_r",
            vmin=-global_pv_vmax, vmax=global_pv_vmax,
            shading="auto")
        z_dam = z_s / (9.81 * 10)
        z_levels = np.arange(
            np.nanmin(z_dam) - 2, np.nanmax(z_dam) + 2, 4)
        if len(z_levels) > 2:
            ax1.contour(
                X2d, Y2d, z_dam, levels=z_levels,
                colors="k", linewidths=0.6, alpha=0.7)
        cb1 = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
        cb1.set_label("PV' [PVU]", fontsize=9)
        ax1.set_xlabel("Relative Lon [°]")
        ax1.set_ylabel("Relative Lat [°]")
        ax1.set_title(f"PV anomaly + Z ({level} hPa)", fontsize=10)
        ax1.set_aspect("equal")

        # ── Right panel: PV tendency + ellipse + major axis ──
        im2 = ax2.pcolormesh(
            X2d, Y2d, pv_dt_s * 1e6,
            cmap="RdBu_r",
            vmin=-global_dt_vmax, vmax=global_dt_vmax,
            shading="auto")
        # Mask contour (black)
        ax2.contour(
            X2d, Y2d, blob_mask.astype(float),
            levels=[0.5], colors="k", linewidths=1.2)
        # Covariance ellipse + major axis line
        if ell_patch is not None:
            ax2.add_patch(ell_patch)
            if np.isfinite(tilt_deg) and smaj > 0:
                theta_rad = np.radians(tilt_deg)
                L = smaj * 1.2
                dx_l = L * np.cos(theta_rad)
                dy_l = L * np.sin(theta_rad)
                ax2.plot(
                    [ecx - dx_l, ecx + dx_l],
                    [ecy - dy_l, ecy + dy_l],
                    "-", color="lime", lw=2.5, zorder=11)

        cb2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
        cb2.set_label("dq/dt [PVU/s]", fontsize=9)
        ax2.set_xlabel("Relative Lon [°]")
        ax2.set_ylabel("Relative Lat [°]")
        tilt_str = (f"tilt = {tilt_deg:.1f}°"
                    if np.isfinite(tilt_deg) else "tilt = N/A")
        ax2.set_title(
            f"PV tendency + mask ellipse ({level} hPa)  [{tilt_str}]",
            fontsize=10)
        ax2.set_aspect("equal")

        # ── Suptitle (fully visible) ──
        variant_label = (variant_name
                         .replace("_", " → ")
                         .replace("onset", "Onset")
                         .replace("decay", "Decay")
                         .replace("awb", "AWB")
                         .replace("cwb", "CWB"))
        fig.suptitle(
            f"{variant_label}\n"
            f"t{dh:02d} ({pct:.0f}% lifecycle)  |  "
            f"N={n_events} events  |  "
            f"Δt = {dt_mean:.1f} ± {dt_std:.1f} h/step  |  "
            f"n_loaded={n_loaded}",
            fontsize=10, fontweight="bold", y=0.97)

        # Render to array
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

        tilt_info = (f", tilt={tilt_deg:.1f}°"
                     if np.isfinite(tilt_deg) else "")
        print(f"  Frame t{dh:02d} ({pct:.0f}%): "
              f"{n_loaded} events{tilt_info}")

    # Write GIF
    print(f"\n  Writing GIF: {gif_path}")
    imageio.mimsave(gif_path, frames, duration=GIF_DURATION, loop=0)
    print(f"  Done! {len(frames)} frames, "
          f"{os.path.getsize(gif_path)/1e6:.1f} MB")
    return gif_path


def main():
    parser = argparse.ArgumentParser(description="Lifecycle composite GIF generator")
    parser.add_argument("--variant", nargs="+",
                        default=["onset_awb_decay_cwb", "onset_cwb_decay_awb"],
                        help="Variant name(s)")
    parser.add_argument("--level", type=int, default=250, help="Pressure level in hPa")
    args = parser.parse_args()

    for vname in args.variant:
        try:
            make_gif(vname, level=args.level)
        except Exception as e:
            print(f"\n  ERROR generating GIF for {vname}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
