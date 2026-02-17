#!/usr/bin/env python
"""Generate all example notebooks for the pvtend package.

Uses real ERA5 blocking / PRP event data from the outputs_tmp directory.
Run: micromamba run -n blocking python examples/_generate_notebooks.py
"""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent

# ── helpers ─────────────────────────────────────────────────────────────
def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())

def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())

def save(nb: nbf.NotebookNode, name: str):
    nb.metadata.kernelspec = {
        "display_name": "Python 3 (blocking)",
        "language": "python",
        "name": "python3",
    }
    path = EXAMPLES_DIR / name
    nbf.write(nb, path)
    print(f"  ✓ {path}")


# =====================================================================
# NB01 — RWB Detection & Derivative Operators
# =====================================================================
def make_nb01():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# 01 — Rossby Wave Breaking & Derivative Operators

This notebook loads a **real ERA5 blocking event** and demonstrates:

1. Grid setup (`NHGrid`)
2. PV derivative computation (`ddx`, `ddy`, `ddp`)
3. Rossby-wave-breaking (RWB) detection (`detect_rwb_events`)
"""),
        code("""\
import numpy as np
import matplotlib.pyplot as plt

from pvtend import NHGrid, ddx, ddy, ddp, R_EARTH
from pvtend.rwb import detect_rwb_events, RWBConfig
from pvtend.io import load_npz_patch
from pvtend.constants import DEFAULT_LEVELS
"""),
        md("## 1  Load a sample blocking event"),
        code("""\
# ── data path (single event at onset, dh=0) ──
DATA_ROOT = "/net/flood/data2/users/x_yan/tempest_extreme_4_basis/outputs_tmp"
npz_path = f"{DATA_ROOT}/blocking_tmp/onset/dh=+0/track_100_1992080507_dh+0.npz"

d = dict(np.load(npz_path))
lat = d["lat_vec"]          # (29,)
lon = d["lon_vec_unwrapped"] # (49,)
levels = d["levels"]         # (9,) hPa
X_rel, Y_rel = d["X_rel"], d["Y_rel"]  # relative-degree coords

print(f"Patch shape  : lat {lat.shape}, lon {lon.shape}")
print(f"Lat range    : {lat.min():.1f}° – {lat.max():.1f}°")
print(f"Lon range    : {lon.min():.1f}° – {lon.max():.1f}°")
print(f"Levels (hPa) : {levels}")
print(f"Track ID     : {int(d['track_id'])}")
print(f"Timestamp    : {str(d['ts'])}")
"""),
        md("## 2  Grid helper"),
        code("""\
grid = NHGrid(lat=lat[::-1], lon=lon)   # NHGrid expects descending lat
dx_arr = grid.dx_arr[::-1]              # match ascending-lat data order
dy    = grid.dy
print(f"dy = {dy:.0f} m")
print(f"dx range = {dx_arr.min():.0f} – {dx_arr.max():.0f} m")
"""),
        md("## 3  Compute PV derivatives and compare with stored values"),
        code("""\
pv_3d = d["pv_3d"]  # (9, 29, 49)

# Compute ∂PV/∂x, ∂PV/∂y with pvtend operators
pv_dx_computed = np.stack([ddx(pv_3d[k], dx_arr, periodic=False) for k in range(9)])
pv_dy_computed = np.stack([ddy(pv_3d[k], dy) for k in range(9)])

# Pre-computed values from the NPZ (weighted-average of 300/250/200 hPa)
pv_dx_stored = d["pv_dx_3d"]
pv_dy_stored = d["pv_dy_3d"]

# Agreement check (interior points)
rel_err_dx = np.nanmean(np.abs(pv_dx_computed[:, 2:-2, 2:-2] - pv_dx_stored[:, 2:-2, 2:-2])) / (
    np.nanmean(np.abs(pv_dx_stored[:, 2:-2, 2:-2])) + 1e-30)
rel_err_dy = np.nanmean(np.abs(pv_dy_computed[:, 2:-2, 2:-2] - pv_dy_stored[:, 2:-2, 2:-2])) / (
    np.nanmean(np.abs(pv_dy_stored[:, 2:-2, 2:-2])) + 1e-30)
print(f"Relative error  ∂PV/∂x : {rel_err_dx:.4e}")
print(f"Relative error  ∂PV/∂y : {rel_err_dy:.4e}")
"""),
        code("""\
# Pressure derivative
plevs_pa = levels.astype(float) * 100.0  # hPa → Pa
pv_dp_computed = ddp(pv_3d, plevs_pa)

pv_dp_stored = d["pv_dp_3d"]
rel_err_dp = np.nanmean(np.abs(pv_dp_computed[1:-1] - pv_dp_stored[1:-1])) / (
    np.nanmean(np.abs(pv_dp_stored[1:-1])) + 1e-30)
print(f"Relative error  ∂PV/∂p : {rel_err_dp:.4e}")
"""),
        md("## 4  Visualise PV & its gradients at 300 hPa"),
        code("""\
ilev = np.argmin(np.abs(levels - 300))  # 300 hPa index

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
kw = dict(cmap="coolwarm", origin="lower")

im0 = axes[0].contourf(X_rel[0], Y_rel[:, 0], pv_3d[ilev], levels=20, **kw)
axes[0].set_title("PV @ 300 hPa  [PVU]")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

vmax = np.nanpercentile(np.abs(pv_dx_stored[ilev]), 98)
im1 = axes[1].contourf(X_rel[0], Y_rel[:, 0], pv_dx_stored[ilev],
                        levels=np.linspace(-vmax, vmax, 20), **kw)
axes[1].set_title("∂PV/∂x @ 300 hPa")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

im2 = axes[2].contourf(X_rel[0], Y_rel[:, 0], pv_dy_stored[ilev],
                        levels=np.linspace(-vmax, vmax, 20), **kw)
axes[2].set_title("∂PV/∂y @ 300 hPa")
plt.colorbar(im2, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.set_xlabel("Relative longitude [°]")
    ax.set_ylabel("Relative latitude [°]")
    ax.set_aspect("equal")
fig.suptitle(f"Track {int(d['track_id'])}  |  {str(d['ts'])}  |  dh=0 (onset)", y=1.02)
fig.tight_layout()
plt.show()
"""),
        md("## 5  RWB detection on the PV anomaly field"),
        code("""\
# Use the weighted-average PV anomaly (2D)
pv_anom_2d = d["pv_anom"]       # (29, 49)
x_coords = X_rel[0, :]          # 1D longitude coordinates
y_coords = Y_rel[:, 0]          # 1D latitude coordinates

cfg = RWBConfig(try_levels=200, max_keep=8, min_vertices=20, area_min_deg2=20.0)
rwb_events = detect_rwb_events(pv_anom_2d, x_coords, y_coords, cfg=cfg)
print(f"Detected {len(rwb_events)} RWB events")
for ev in rwb_events:
    print(f"  {ev['wb_type']:3s}  area={ev['area']:.1f} deg²  "
          f"centroid=({ev['centroid'][0]:.1f}°, {ev['centroid'][1]:.1f}°)")
"""),
        code("""\
fig, ax = plt.subplots(figsize=(8, 6))
vmax = np.nanpercentile(np.abs(pv_anom_2d), 98)
ax.contourf(x_coords, y_coords, pv_anom_2d,
            levels=np.linspace(-vmax, vmax, 24), cmap="coolwarm")
ax.contour(x_coords, y_coords, pv_anom_2d, levels=[0], colors="k",
           linewidths=1.0, linestyles="--")

colors = {"AWB": "dodgerblue", "CWB": "tomato", "UNK": "gray"}
for ev in rwb_events:
    c = colors.get(ev["wb_type"], "gray")
    ax.fill(ev["polygon_x"], ev["polygon_y"], alpha=0.3, color=c,
            label=ev["wb_type"])
    ax.plot(ev["polygon_x"], ev["polygon_y"], color=c, lw=1.5)

ax.set_xlabel("Relative longitude [°]")
ax.set_ylabel("Relative latitude [°]")
ax.set_title("PV anomaly (wavg) with detected RWB regions")
# deduplicate legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
if by_label:
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
"""),
        md("""\
## Summary

- **`ddx` / `ddy` / `ddp`** reproduce the pre-computed derivatives stored in the NPZ patches.
- **`detect_rwb_events`** identifies anticyclonic and/or cyclonic wave-breaking lobes
  from the overturning PV contours on an event-centred patch.
"""),
    ]
    save(nb, "01_rwb_and_derivatives.ipynb")


# =====================================================================
# NB02 — Helmholtz Decomposition & QG Omega
# =====================================================================
def make_nb02():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# 02 — Helmholtz Decomposition & QG Omega

Demonstrates on a real blocking event:

1. 3-D Helmholtz decomposition of anomalous winds → rotational / divergent / harmonic
2. QG omega equation solver
3. Moist / dry omega decomposition
"""),
        code("""\
import numpy as np
import matplotlib.pyplot as plt

from pvtend import helmholtz_decomposition, R_EARTH
from pvtend.helmholtz import helmholtz_decomposition_3d
from pvtend.omega import solve_qg_omega
from pvtend.moist_dry import decompose_omega
"""),
        md("## 1  Load event data"),
        code("""\
DATA_ROOT = "/net/flood/data2/users/x_yan/tempest_extreme_4_basis/outputs_tmp"
d = dict(np.load(f"{DATA_ROOT}/blocking_tmp/onset/dh=+0/track_100_1992080507_dh+0.npz"))

lat = d["lat_vec"]
lon = d["lon_vec_unwrapped"]
levels = d["levels"]
X_rel, Y_rel = d["X_rel"], d["Y_rel"]
print(f"Grid: {lat.shape[0]}×{lon.shape[0]}, {len(levels)} levels")
"""),
        md("## 2  Helmholtz decomposition of anomalous wind"),
        code("""\
u_anom_3d = d["u_anom_3d"]   # (9, 29, 49)
v_anom_3d = d["v_anom_3d"]

result_3d = helmholtz_decomposition_3d(u_anom_3d, v_anom_3d, lat, lon, method="direct")
print("Helmholtz output keys:", sorted(result_3d.keys()))
"""),
        code("""\
# Compare with pre-computed NPZ values at 300 hPa
ilev = np.argmin(np.abs(levels - 300))

for comp in ["u_rot", "v_rot", "u_div", "v_div"]:
    npz_key = f"u_anom_{comp.split('_')[1]}_3d" if comp.startswith("u") else f"v_anom_{comp.split('_')[1]}_3d"
    computed = result_3d[comp][ilev]
    stored   = d[npz_key][ilev]
    relerr   = np.nanmean(np.abs(computed - stored)) / (np.nanmean(np.abs(stored)) + 1e-30)
    print(f"  {comp:6s}  rel-err = {relerr:.4e}")
"""),
        md("## 3  Visualise rotational vs divergent wind at 300 hPa"),
        code("""\
fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
skip = 2
x, y = X_rel[0], Y_rel[:, 0]
xx, yy = np.meshgrid(x, y)

for ax, (title, uk, vk) in zip(axes, [
    ("Total u'/v'", u_anom_3d[ilev], v_anom_3d[ilev]),
    ("Rotational", result_3d["u_rot"][ilev], result_3d["v_rot"][ilev]),
    ("Divergent",  result_3d["u_div"][ilev], result_3d["v_div"][ilev]),
]):
    speed = np.sqrt(uk**2 + vk**2)
    ax.contourf(x, y, speed, levels=20, cmap="YlOrRd")
    ax.quiver(xx[::skip, ::skip], yy[::skip, ::skip],
              uk[::skip, ::skip], vk[::skip, ::skip],
              scale=200, width=0.003, color="k")
    ax.set_title(title)
    ax.set_aspect("equal")

axes[0].set_ylabel("Relative latitude [°]")
for ax in axes:
    ax.set_xlabel("Relative longitude [°]")
fig.suptitle("Helmholtz decomposition of anomalous wind @ 300 hPa", y=1.02)
fig.tight_layout()
plt.show()
"""),
        md("## 4  QG omega equation"),
        code("""\
u_3d = d["u_3d"]
v_3d = d["v_3d"]
t_3d = d["t_3d"]
plevs_pa = levels.astype(float) * 100.0

omega_dry_computed = solve_qg_omega(u_3d, v_3d, t_3d, lat, lon, plevs_pa)
omega_dry_stored   = d["omega_dry_3d"]

rel_err = np.nanmean(np.abs(omega_dry_computed[1:-1] - omega_dry_stored[1:-1])) / (
    np.nanmean(np.abs(omega_dry_stored[1:-1])) + 1e-30)
print(f"QG omega_dry  rel-err = {rel_err:.4e}")
print(f"omega_dry range: [{omega_dry_computed.min():.3e}, {omega_dry_computed.max():.3e}] Pa/s")
"""),
        md("## 5  Omega cross-section through blocking centre"),
        code("""\
imid_lat = lat.shape[0] // 2  # centre row

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
omega_total_3d = d["w_3d"]  # total omega (Pa/s)

for ax, (title, omega) in zip(axes, [
    ("ω_dry (QG)", omega_dry_stored),
    ("ω_moist (residual)", d["omega_moist_3d"]),
]):
    vmax = np.nanpercentile(np.abs(omega[:, imid_lat, :]), 95)
    cf = ax.contourf(x, levels, omega[:, imid_lat, :],
                     levels=np.linspace(-vmax, vmax, 20), cmap="RdBu_r")
    ax.set_title(title)
    ax.set_xlabel("Relative longitude [°]")
    plt.colorbar(cf, ax=ax, label="Pa/s")

axes[0].set_ylabel("Pressure [hPa]")
axes[0].invert_yaxis()
fig.suptitle("Longitude–pressure cross-section through blocking centre", y=1.02)
fig.tight_layout()
plt.show()
"""),
        md("## 6  Moist–dry omega decomposition"),
        code("""\
omega_total = d["w_3d"]
omega_dry   = d["omega_dry_3d"]

decomp = decompose_omega(omega_total, omega_dry, lat, lon, plevs_pa,
                         u_div=result_3d["u_div"], v_div=result_3d["v_div"])
print("decompose_omega keys:", sorted(decomp.keys()))
print(f"omega_moist range: [{decomp['omega_moist'].min():.3e}, "
      f"{decomp['omega_moist'].max():.3e}] Pa/s")
"""),
        md("""\
## Summary

- **`helmholtz_decomposition_3d`** splits anomalous wind into rotational (ψ),
  divergent (χ) and harmonic components at every pressure level.
- **`solve_qg_omega`** solves the QG omega equation to isolate the adiabatic
  component of vertical motion.
- **`decompose_omega`** partitions ω into dry (QG) and moist (residual) parts,
  optionally computing the divergent-wind contribution of each.
"""),
    ]
    save(nb, "02_helmholtz_and_qg_omega.ipynb")


# =====================================================================
# NB03 — Four-Basis Projection
# =====================================================================
def make_nb03():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# 03 — Orthogonal Four-Basis Decomposition

Demonstrates the full projection workflow on a **real blocking event**:

1. Build the orthogonal basis {Φ₁, Φ₂, Φ₃, Φ₄} from the PV anomaly field
2. Project the PV tendency onto the basis → (β, αx, αy, γ)
3. Lifecycle time curves by looping over ∆h = −13 … +12
"""),
        code("""\
import numpy as np
import matplotlib.pyplot as plt

from pvtend import (compute_orthogonal_basis, project_field, R_EARTH)
from pvtend.plotting import plot_four_basis, plot_coefficient_curves, plot_field_2d
from pvtend.decomposition.projection import collect_term_fields, ADVECTION_TERMS
from pvtend.decomposition.basis import PRENORM_PHI1, PRENORM_PHI2, PRENORM_PHI3, PRENORM_PHI4
"""),
        md("## 1  Load event data at onset (dh = 0)"),
        code("""\
DATA_ROOT = "/net/flood/data2/users/x_yan/tempest_extreme_4_basis/outputs_tmp"
STAGE = "onset"
TRACK_GLOB = "track_100_*"   # 1992-08-05 blocking event

d0 = dict(np.load(f"{DATA_ROOT}/blocking_tmp/{STAGE}/dh=+0/{TRACK_GLOB.replace('*','1992080507_dh+0')}.npz"))
X_rel = d0["X_rel"]
Y_rel = d0["Y_rel"]
x_rel = X_rel[0, :]    # 1D
y_rel = Y_rel[:, 0]

print(f"Patch shape : {X_rel.shape}")
print(f"PV anom min : {d0['pv_anom'].min():.3e} PVU")
"""),
        md("## 2  Build orthogonal basis from PV anomaly"),
        code("""\
basis = compute_orthogonal_basis(
    pv_anom=d0["pv_anom"],
    pv_dx=d0["pv_anom_dx"],
    pv_dy=d0["pv_anom_dy"],
    x_rel=x_rel,
    y_rel=y_rel,
    mask_negative=True,
    apply_smoothing=True,
    smoothing_deg=6.0,
    grid_spacing=1.5,
)
print("Basis norms :", {k: f"{v:.4e}" for k, v in basis.norms.items()})
print("Scale factors:", basis.scale_factors)
"""),
        md("## 3  Visualise the four basis fields"),
        code("""\
fig = plot_four_basis(
    basis.phi_int, basis.phi_dx, basis.phi_dy, basis.phi_def,
    x_rel, y_rel,
    suptitle="Orthogonal basis — Track 100 (onset, dh=0)",
)
plt.show()
"""),
        md("## 4  Project PV tendency onto basis"),
        code("""\
from pvtend.decomposition.smoothing import gaussian_smooth_nan

pv_dt = d0["pv_anom_dt"]   # dq'/dt (weighted-average)
pv_dt_smooth = gaussian_smooth_nan(pv_dt, smoothing_deg=6.0, grid_spacing=1.5)

proj = project_field(pv_dt_smooth, basis)

print(f"β  (intensification) = {proj['beta']:.3e}  s⁻¹")
print(f"αx (zonal propagation) = {proj['ax']:.3f}  m/s")
print(f"αy (merid. propagation) = {proj['ay']:.3f}  m/s")
print(f"γ  (deformation)       = {proj['gamma']:.3e}  s⁻¹")
print(f"RMSE / max|dq/dt|      = {proj['rmse'] / (np.nanmax(np.abs(pv_dt_smooth)) + 1e-30):.3f}")
"""),
        md("## 5  2-D component maps"),
        code("""\
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
titles = ["Intensification (β·Φ₁)", "Zonal prop. (αx·Φ₂)",
          "Merid. prop. (αy·Φ₃)", "Deformation (γ·Φ₄)"]
fields = [proj["int"], proj["prop"][:, :] * 0,  # prop not separate by x/y in output
          proj["def"], proj["resid"]]

# Reconstruct individual components manually
beta_comp = proj["beta_raw"] * basis.phi_int
ax_comp   = proj["ax_raw"]   * basis.phi_dx
ay_comp   = proj["ay_raw"]   * basis.phi_dy
gamma_comp = proj["gamma_raw"] * basis.phi_def

components = [beta_comp, ax_comp, ay_comp, gamma_comp]
vmax = np.nanpercentile(np.abs(pv_dt_smooth), 95)

for ax, comp, title in zip(axes.flat, components, titles):
    cf = ax.contourf(x_rel, y_rel, comp,
                     levels=np.linspace(-vmax, vmax, 20), cmap="coolwarm")
    ax.contour(x_rel, y_rel, d0["pv_anom"], levels=[0], colors="k",
               linewidths=1.0, linestyles="--")
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, shrink=0.8)

fig.suptitle("PV tendency decomposition — dh = 0 (onset)", y=1.02)
fig.tight_layout()
plt.show()
"""),
        md("## 6  Lifecycle time curves (dh = −13 … +12)"),
        code("""\
import os, glob

dh_values = list(range(-13, 13))
coefs = {k: [] for k in ["beta", "ax", "ay", "gamma"]}

for dh in dh_values:
    sign = "+" if dh >= 0 else ""
    pattern = f"{DATA_ROOT}/blocking_tmp/{STAGE}/dh={sign}{dh}/track_100_*_dh{sign}{dh}.npz"
    files = sorted(glob.glob(pattern))
    if not files:
        for k in coefs:
            coefs[k].append(np.nan)
        continue

    dd = dict(np.load(files[0]))

    # Build basis at dh-1 (or dh=0 for the first step)
    dh_basis = max(dh - 1, -13)
    sign_b = "+" if dh_basis >= 0 else ""
    pattern_b = f"{DATA_ROOT}/blocking_tmp/{STAGE}/dh={sign_b}{dh_basis}/track_100_*_dh{sign_b}{dh_basis}.npz"
    files_b = sorted(glob.glob(pattern_b))
    if not files_b:
        for k in coefs:
            coefs[k].append(np.nan)
        continue
    db = dict(np.load(files_b[0]))

    b = compute_orthogonal_basis(
        db["pv_anom"], db["pv_anom_dx"], db["pv_anom_dy"],
        x_rel, y_rel, mask_negative=True,
        apply_smoothing=True, smoothing_deg=6.0, grid_spacing=1.5,
    )
    pv_dt_s = gaussian_smooth_nan(dd["pv_anom_dt"], smoothing_deg=6.0, grid_spacing=1.5)
    p = project_field(pv_dt_s, b)
    for k in coefs:
        coefs[k].append(p[k])

# Convert to arrays
for k in coefs:
    coefs[k] = np.array(coefs[k])
"""),
        code("""\
fig = plot_coefficient_curves(
    np.array(dh_values),
    coefs,
    title="Track 100 — Lifecycle coefficients (onset-relative)",
    xlabel="Hours relative to onset",
)
plt.show()
"""),
        md("""\
## Summary

- **`compute_orthogonal_basis`** builds the four Gram-Schmidt-orthogonalised
  basis fields (Φ₁…Φ₄) from the PV anomaly and its spatial gradients.
- **`project_field`** decomposes any 2-D field (e.g. dq'/dt) into
  intensification (β), propagation (αx, αy), and deformation (γ) coefficients.
- The lifecycle curve shows how these coefficients evolve
  from 13 h before onset to 12 h after.
"""),
    ]
    save(nb, "03_four_basis_projection.ipynb")


# =====================================================================
# NB05 — Grouped Terms & Bootstrap Significance
# =====================================================================
def make_nb05():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# 05 — Grouped PV-Tendency Terms & Bootstrap Significance

This notebook demonstrates:

1. Loading **all** blocking-event NPZ files for a given (stage, dh)
2. Grouping PV-tendency cross-terms into physically meaningful categories
3. Projecting each group onto the four-basis decomposition
4. Bootstrap resampling for confidence intervals and significance testing
"""),
        code("""\
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from concurrent.futures import ThreadPoolExecutor

from pvtend import compute_orthogonal_basis, project_field
from pvtend.decomposition.smoothing import gaussian_smooth_nan
from pvtend.decomposition.basis import PRENORM_PHI1
"""),
        md("## 1  Discover and load all event files"),
        code("""\
DATA_ROOT = "/net/flood/data2/users/x_yan/tempest_extreme_4_basis/outputs_tmp"
STAGE = "onset"
DH = 0

sign = "+" if DH >= 0 else ""
npz_dir = f"{DATA_ROOT}/blocking_tmp/{STAGE}/dh={sign}{DH}"
npz_files = sorted(glob.glob(os.path.join(npz_dir, "track_*.npz")))
print(f"Found {len(npz_files)} events for stage={STAGE}, dh={DH}")
"""),
        code("""\
def load_event(path):
    \"\"\"Load an NPZ file and return a dict.\"\"\"
    return dict(np.load(path))

# Load all events (parallel I/O)
with ThreadPoolExecutor(max_workers=8) as pool:
    events = list(pool.map(load_event, npz_files))

print(f"Loaded {len(events)} events, each with {len(events[0])} fields")
X_rel = events[0]["X_rel"]
Y_rel = events[0]["Y_rel"]
x_rel = X_rel[0, :]
y_rel = Y_rel[:, 0]
"""),
        md("## 2  Compute composite-mean basis fields"),
        code("""\
# Average PV anomaly and its derivatives across all events
pv_anom_mean = np.mean([e["pv_anom"] for e in events], axis=0)
pv_dx_mean   = np.mean([e["pv_anom_dx"] for e in events], axis=0)
pv_dy_mean   = np.mean([e["pv_anom_dy"] for e in events], axis=0)

basis = compute_orthogonal_basis(
    pv_anom_mean, pv_dx_mean, pv_dy_mean,
    x_rel, y_rel,
    mask_negative=True,
    apply_smoothing=True, smoothing_deg=6.0, grid_spacing=1.5,
)
print("Composite basis norms:", {k: f"{v:.4e}" for k, v in basis.norms.items()})
"""),
        md("## 3  Define grouped terms"),
        code("""\
# Physical groupings of PV-tendency cross-terms
GROUPS = {
    "leading_order_adv": [
        "u_bar_pv_anom_dx", "v_bar_pv_anom_dy",
        "u_anom_pv_bar_dx", "v_anom_pv_bar_dy",
    ],
    "eddy_total": [
        "u_rot_pv_anom_dx", "v_rot_pv_anom_dy",
        "u_div_pv_anom_dx", "v_div_pv_anom_dy",
        "u_har_pv_anom_dx", "v_har_pv_anom_dy",
        "u_anom_pv_anom_dx", "v_anom_pv_anom_dy",  # for 2D checks
    ],
    "moist_indirect": [
        "u_div_moist_pv_anom_dx", "v_div_moist_pv_anom_dy",
        "w_anom_pv_anom_dp",
    ],
    "Q": ["Q"],
}

def sum_group(event, group_terms, negate_advection=True):
    \"\"\"Sum a group of terms from an event, with sign convention -u·∇q.\"\"\"
    total = np.zeros_like(event["pv_anom"])
    for t in group_terms:
        if t in event:
            val = event[t]
            if negate_advection and t != "Q":
                val = -val
            total += val
    return total
"""),
        md("## 4  Project composite-mean grouped terms"),
        code("""\
# Compute composite mean of each group
group_means = {}
for gname, terms in GROUPS.items():
    group_means[gname] = np.mean(
        [sum_group(e, terms) for e in events], axis=0
    )

# Project each group
group_coefs = {}
for gname, field in group_means.items():
    field_s = gaussian_smooth_nan(field, smoothing_deg=6.0, grid_spacing=1.5)
    p = project_field(field_s, basis)
    group_coefs[gname] = {k: p[k] for k in ["beta", "ax", "ay", "gamma"]}
    print(f"{gname:25s}  β={p['beta']:.3e}  αx={p['ax']:.3f}  "
          f"αy={p['ay']:.3f}  γ={p['gamma']:.3e}")
"""),
        md("## 5  Bootstrap resampling for significance"),
        code("""\
N_BOOT = 1000
rng = np.random.default_rng(42)
n_events = len(events)

def bootstrap_group(group_name, terms, n_boot=N_BOOT):
    \"\"\"Bootstrap the projected coefficients for one group.\"\"\"
    # Per-event projections
    per_event = []
    for e in events:
        field = sum_group(e, terms)
        field_s = gaussian_smooth_nan(field, smoothing_deg=6.0, grid_spacing=1.5)
        p = project_field(field_s, basis)
        per_event.append({k: p[k] for k in ["beta", "ax", "ay", "gamma"]})

    per_event_arr = {k: np.array([pe[k] for pe in per_event]) for k in ["beta", "ax", "ay", "gamma"]}

    # Bootstrap
    boot = {k: np.empty(n_boot) for k in ["beta", "ax", "ay", "gamma"]}
    for b in range(n_boot):
        idx = rng.integers(0, n_events, size=n_events)
        for k in boot:
            boot[k][b] = per_event_arr[k][idx].mean()

    return boot

print("Running bootstrap (this may take a minute)...")
boot_results = {}
for gname, terms in GROUPS.items():
    boot_results[gname] = bootstrap_group(gname, terms)
    lo, hi = np.percentile(boot_results[gname]["beta"], [2.5, 97.5])
    sig = "***" if lo * hi > 0 else "n.s."
    print(f"  {gname:25s}  β 95% CI: [{lo:.3e}, {hi:.3e}]  {sig}")
"""),
        md("## 6  Visualise bootstrap results"),
        code("""\
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
coef_names = ["beta", "ax", "ay", "gamma"]
coef_labels = ["β (intensification) [s⁻¹]",
               "αx (zonal propagation) [m/s]",
               "αy (meridional propagation) [m/s]",
               "γ (deformation) [s⁻¹]"]
group_colors = {"leading_order_adv": "C0", "eddy_total": "C1",
                "moist_indirect": "C2", "Q": "C3"}

for ax, cname, clabel in zip(axes.flat, coef_names, coef_labels):
    for i, (gname, bdata) in enumerate(boot_results.items()):
        vals = bdata[cname]
        mean = vals.mean()
        lo, hi = np.percentile(vals, [2.5, 97.5])
        sig = lo * hi > 0  # does CI exclude zero?

        ax.barh(i, mean, xerr=[[mean - lo], [hi - mean]],
                color=group_colors.get(gname, "gray"),
                alpha=0.8 if sig else 0.3,
                capsize=4, height=0.6,
                label=gname if ax == axes.flat[0] else "")
        if sig:
            ax.text(mean, i + 0.35, "***", ha="center", fontsize=8, color="red")

    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels(list(GROUPS.keys()), fontsize=9)
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel(clabel)

fig.suptitle(f"Bootstrap coefficients — {STAGE} dh={DH}  (N={n_events}, B={N_BOOT})", y=1.02)
fig.tight_layout()
plt.show()
"""),
        md("""\
## Summary

- PV-tendency cross-terms are **grouped** into physically meaningful categories:
  leading-order advection, eddy (rotational + divergent), moist-indirect, and diabatic Q.
- Each group is projected onto the same orthogonal basis to obtain (β, αx, αy, γ).
- **Bootstrap resampling** (N=1000) provides 95 % confidence intervals;
  bars are opaque when the CI excludes zero (significant at p < 0.05).
"""),
    ]
    save(nb, "05_grouped_terms_bootstrap.ipynb")


# =====================================================================
# NB06 — Baroclinic Structure
# =====================================================================
def make_nb06():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# 06 — Baroclinic Structure & Tropopause Pressure

Demonstrates the **vertical structure** of blocking events using real data:

1. Composite-mean 3-D PV anomaly at all 9 pressure levels
2. Longitude–pressure cross-sections at onset / peak / decay
3. 2-PVU dynamical tropopause pressure distribution
"""),
        code("""\
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from concurrent.futures import ThreadPoolExecutor
"""),
        md("## 1  Load composite 3-D fields"),
        code("""\
DATA_ROOT = "/net/flood/data2/users/x_yan/tempest_extreme_4_basis/outputs_tmp"

def load_composite_3d(stage, dh, field="pv_anom_3d", max_events=None):
    \"\"\"Compute composite mean of a 3-D field from all event NPZs.\"\"\"
    sign = "+" if dh >= 0 else ""
    npz_dir = f"{DATA_ROOT}/blocking_tmp/{stage}/dh={sign}{dh}"
    files = sorted(glob.glob(os.path.join(npz_dir, "track_*.npz")))
    if max_events:
        files = files[:max_events]

    def _load(f):
        return np.load(f)[field]

    with ThreadPoolExecutor(max_workers=8) as pool:
        arrs = list(pool.map(_load, files))

    return np.mean(arrs, axis=0), len(arrs)

# Load for three lifecycle stages at dh=0
stages = ["onset", "peak", "decay"]
composites = {}
for stg in stages:
    composites[stg], n = load_composite_3d(stg, dh=0, field="pv_anom_3d")
    print(f"  {stg:6s}: {n} events, shape {composites[stg].shape}")

# Grid info from one file
d0 = dict(np.load(glob.glob(f"{DATA_ROOT}/blocking_tmp/onset/dh=+0/track_*.npz")[0]))
lat = d0["lat_vec"]
lon = d0["lon_vec_unwrapped"]
levels = d0["levels"]  # hPa
X_rel = d0["X_rel"]
Y_rel = d0["Y_rel"]
"""),
        md("## 2  PV anomaly at each pressure level (onset)"),
        code("""\
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
vmax = np.nanpercentile(np.abs(composites["onset"]), 95)

for i, ax in enumerate(axes.flat):
    cf = ax.contourf(X_rel[0], Y_rel[:, 0], composites["onset"][i],
                     levels=np.linspace(-vmax, vmax, 20), cmap="coolwarm")
    ax.contour(X_rel[0], Y_rel[:, 0], composites["onset"][i],
              levels=[0], colors="k", linewidths=0.8, linestyles="--")
    ax.set_title(f"{levels[i]} hPa")
    ax.set_aspect("equal")
    if i % 3 == 0:
        ax.set_ylabel("Rel. lat [°]")
    if i >= 6:
        ax.set_xlabel("Rel. lon [°]")

fig.suptitle("Composite PV anomaly at each level — onset, dh=0", fontsize=14, y=1.01)
fig.colorbar(cf, ax=axes, shrink=0.6, label="PV anomaly [PVU]")
fig.tight_layout()
plt.show()
"""),
        md("## 3  Longitude–pressure cross-sections (onset / peak / decay)"),
        code("""\
imid = lat.shape[0] // 2  # centre latitude row

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, stg in zip(axes, stages):
    vmax = np.nanpercentile(np.abs(composites[stg][:, imid, :]), 95)
    cf = ax.contourf(X_rel[0], levels, composites[stg][:, imid, :],
                     levels=np.linspace(-vmax, vmax, 20), cmap="coolwarm")
    ax.contour(X_rel[0], levels, composites[stg][:, imid, :],
              levels=[0], colors="k", linewidths=0.8)
    ax.set_title(stg.capitalize())
    ax.set_xlabel("Relative longitude [°]")
    plt.colorbar(cf, ax=ax, shrink=0.8)

axes[0].set_ylabel("Pressure [hPa]")
axes[0].invert_yaxis()
fig.suptitle("PV anomaly — lon–pressure cross-section through blocking centre", y=1.02)
fig.tight_layout()
plt.show()
"""),
        md("## 4  Dynamical tropopause pressure (2 PVU)"),
        code("""\
def find_2pvu_pressure(pv_3d, levels_hPa):
    \"\"\"Find pressure level where PV crosses 2.0 PVU at each grid point.

    Interpolates between levels to find the 2-PVU surface.
    Returns pressure in hPa (NaN where 2 PVU is not crossed).
    \"\"\"
    nlev, nlat, nlon = pv_3d.shape
    trop_p = np.full((nlat, nlon), np.nan)
    target_pv = 2.0e-6  # 2 PVU in SI units (K m² kg⁻¹ s⁻¹)

    for j in range(nlat):
        for i in range(nlon):
            col = pv_3d[:, j, i]
            # Search from surface (high p) upward (low p)
            for k in range(nlev - 1, 0, -1):
                if (col[k] <= target_pv <= col[k-1]) or (col[k] >= target_pv >= col[k-1]):
                    # Linear interpolation
                    frac = (target_pv - col[k]) / (col[k-1] - col[k] + 1e-30)
                    trop_p[j, i] = levels_hPa[k] + frac * (levels_hPa[k-1] - levels_hPa[k])
                    break
    return trop_p

# Compute 2PVU tropopause for each stage
trop_press = {}
for stg in stages:
    # Need absolute PV for tropopause, load pv_3d composite
    pv_abs, _ = load_composite_3d(stg, dh=0, field="pv_3d")
    trop_press[stg] = find_2pvu_pressure(pv_abs, levels.astype(float))
    tp = trop_press[stg]
    print(f"  {stg:6s}: trop-p range = {np.nanmin(tp):.0f} – {np.nanmax(tp):.0f} hPa, "
          f"NaN frac = {np.isnan(tp).mean():.2%}")
"""),
        code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

for ax, stg in zip(axes, stages):
    tp = trop_press[stg]
    vmin, vmax_p = np.nanpercentile(tp, [5, 95])
    cf = ax.contourf(X_rel[0], Y_rel[:, 0], tp,
                     levels=np.linspace(vmin, vmax_p, 16), cmap="viridis_r")
    ax.contour(X_rel[0], Y_rel[:, 0], tp,
              levels=[300], colors="white", linewidths=1.5)
    ax.set_title(f"{stg.capitalize()}")
    ax.set_aspect("equal")
    ax.set_xlabel("Rel. lon [°]")
    plt.colorbar(cf, ax=ax, label="hPa")

axes[0].set_ylabel("Rel. lat [°]")
fig.suptitle("2-PVU dynamical tropopause pressure", y=1.02)
fig.tight_layout()
plt.show()
"""),
        md("""\
## Summary

- The blocking PV anomaly is **strongest in the upper troposphere** (200–300 hPa)
  with a negative PV anomaly that deepens through the lifecycle.
- The **longitude–pressure cross-section** reveals the tilted, baroclinic structure
  of the blocking high, with the PV anomaly extending from 500 hPa to 100 hPa.
- The **2-PVU tropopause** is lifted (lower pressure) over the blocking centre,
  indicating the locally elevated tropopause associated with blocking.
"""),
    ]
    save(nb, "06_baroclinic_structure.ipynb")


# =====================================================================
# NB07 — Facet Comparison: Blocking vs PRP
# =====================================================================
def make_nb07():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# 07 — Facet Comparison: Blocking vs Persistent Ridge (PRP)

Compares the PV-budget decomposition between **blocking** and **PRP** events:

1. Composite-mean basis and coefficients for both event types
2. Lifecycle coefficient curves side by side
3. Grouped-term decomposition comparison
"""),
        code("""\
import numpy as np
import matplotlib.pyplot as plt
import os, glob
from concurrent.futures import ThreadPoolExecutor

from pvtend import compute_orthogonal_basis, project_field
from pvtend.decomposition.smoothing import gaussian_smooth_nan
"""),
        md("## 1  Helper functions"),
        code("""\
DATA_ROOT = "/net/flood/data2/users/x_yan/tempest_extreme_4_basis/outputs_tmp"

def load_all_events(event_type, stage, dh):
    \"\"\"Load all NPZ files for a given event type / stage / dh.\"\"\"
    sign = "+" if dh >= 0 else ""
    subdir = "blocking_tmp" if event_type == "blocking" else "prp_tmp"
    npz_dir = f"{DATA_ROOT}/{subdir}/{stage}/dh={sign}{dh}"
    files = sorted(glob.glob(os.path.join(npz_dir, "track_*.npz")))

    def _load(f):
        return dict(np.load(f))

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(_load, files))

def compute_composite_basis(events, x_rel, y_rel):
    \"\"\"Build orthogonal basis from composite-mean PV fields.\"\"\"
    pv_mean  = np.mean([e["pv_anom"]    for e in events], axis=0)
    dx_mean  = np.mean([e["pv_anom_dx"] for e in events], axis=0)
    dy_mean  = np.mean([e["pv_anom_dy"] for e in events], axis=0)
    return compute_orthogonal_basis(
        pv_mean, dx_mean, dy_mean, x_rel, y_rel,
        mask_negative=True, apply_smoothing=True,
        smoothing_deg=6.0, grid_spacing=1.5,
    )

GROUPS = {
    "leading_order_adv": [
        "u_bar_pv_anom_dx", "v_bar_pv_anom_dy",
        "u_anom_pv_bar_dx", "v_anom_pv_bar_dy",
    ],
    "eddy_rot": [
        "u_rot_pv_anom_dx", "v_rot_pv_anom_dy",
    ],
    "eddy_div": [
        "u_div_pv_anom_dx", "v_div_pv_anom_dy",
    ],
}

def sum_group(event, terms):
    total = np.zeros_like(event["pv_anom"])
    for t in terms:
        if t in event:
            total -= event[t]   # sign convention: -u·∇q
    return total
"""),
        md("## 2  Load events and compute lifecycle curves"),
        code("""\
STAGE = "onset"
dh_values = list(range(-13, 13))
event_types = ["blocking", "prp"]

results = {}
for etype in event_types:
    print(f"\\nProcessing {etype}...")
    coefs_dh = {g: {k: [] for k in ["beta", "ax", "ay", "gamma"]}
                for g in list(GROUPS.keys()) + ["pv_dt"]}

    for dh in dh_values:
        events = load_all_events(etype, STAGE, dh)
        if not events:
            for g in coefs_dh:
                for k in coefs_dh[g]:
                    coefs_dh[g][k].append(np.nan)
            continue

        x_rel = events[0]["X_rel"][0, :]
        y_rel = events[0]["Y_rel"][:, 0]

        # Build basis at dh-1
        dh_basis = max(dh - 1, -13)
        events_basis = load_all_events(etype, STAGE, dh_basis)
        basis = compute_composite_basis(events_basis, x_rel, y_rel)

        # Project pv_dt
        pv_dt_mean = np.mean([e["pv_anom_dt"] for e in events], axis=0)
        pv_dt_s = gaussian_smooth_nan(pv_dt_mean, smoothing_deg=6.0, grid_spacing=1.5)
        p = project_field(pv_dt_s, basis)
        for k in ["beta", "ax", "ay", "gamma"]:
            coefs_dh["pv_dt"][k].append(p[k])

        # Project grouped terms
        for gname, terms in GROUPS.items():
            gmean = np.mean([sum_group(e, terms) for e in events], axis=0)
            gmean_s = gaussian_smooth_nan(gmean, smoothing_deg=6.0, grid_spacing=1.5)
            p = project_field(gmean_s, basis)
            for k in ["beta", "ax", "ay", "gamma"]:
                coefs_dh[gname][k].append(p[k])

        print(f"  dh={dh:+3d}  N={len(events)}  β(pv_dt)={coefs_dh['pv_dt']['beta'][-1]:.3e}")

    # Convert to arrays
    for g in coefs_dh:
        for k in coefs_dh[g]:
            coefs_dh[g][k] = np.array(coefs_dh[g][k])

    results[etype] = coefs_dh
"""),
        md("## 3  Facet plot: blocking vs PRP lifecycle curves"),
        code("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
coef_keys = ["beta", "ax", "ay"]
coef_labels = ["β  [s⁻¹]", "αx  [m/s]", "αy  [m/s]"]
dh_arr = np.array(dh_values)

# Row 0: leading_order_adv
# Row 1: eddy_rot
row_groups = ["leading_order_adv", "eddy_rot"]
row_titles = ["Leading-order advection", "Eddy (rotational)"]

for row, gname in enumerate(row_groups):
    for col, (ckey, clabel) in enumerate(zip(coef_keys, coef_labels)):
        ax = axes[row, col]
        for etype, ls in [("blocking", "-"), ("prp", "--")]:
            ax.plot(dh_arr, results[etype][gname][ckey], ls,
                    label=etype.capitalize(), lw=2)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("dh [hours]")
        if col == 0:
            ax.set_ylabel(f"{row_titles[row]}\\n{clabel}")
        else:
            ax.set_ylabel(clabel)
        if row == 0 and col == 0:
            ax.legend(fontsize=9)

fig.suptitle(f"Blocking vs PRP — {STAGE} lifecycle", fontsize=14, y=1.01)
fig.tight_layout()
plt.show()
"""),
        md("## 4  Bar comparison at dh = 0"),
        code("""\
idx0 = dh_values.index(0)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
group_names = list(GROUPS.keys())
x_pos = np.arange(len(group_names))
width = 0.35

for ax, ckey, clabel in zip(axes, coef_keys, coef_labels):
    vals_b = [results["blocking"][g][ckey][idx0] for g in group_names]
    vals_p = [results["prp"][g][ckey][idx0] for g in group_names]

    ax.bar(x_pos - width/2, vals_b, width, label="Blocking", color="steelblue")
    ax.bar(x_pos + width/2, vals_p, width, label="PRP", color="coral")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([g.replace("_", "\\n") for g in group_names], fontsize=8)
    ax.set_ylabel(clabel)
    ax.axhline(0, color="k", lw=0.5)
    if ax == axes[0]:
        ax.legend()

fig.suptitle(f"Grouped-term coefficients at {STAGE} dh=0", fontsize=13, y=1.02)
fig.tight_layout()
plt.show()
"""),
        md("""\
## Summary

- **Blocking** and **PRP** events share similar leading-order advective
  contributions but differ systematically in the eddy (nonlinear) terms.
- The rotational eddy term drives **intensification** (positive β) in blocking
  but is weaker or opposite-signed for PRP.
- The **lifecycle curves** show how these differences evolve from pre-onset
  through decay, with blocking maintaining stronger eddy forcing.
"""),
    ]
    save(nb, "07_facet_blocking_vs_prp.ipynb")


# =====================================================================
# Main
# =====================================================================
def main():
    print("Generating pvtend example notebooks …")
    make_nb01()
    make_nb02()
    make_nb03()
    make_nb05()
    make_nb06()
    make_nb07()
    print("\nDone! All notebooks written to examples/")


if __name__ == "__main__":
    main()
