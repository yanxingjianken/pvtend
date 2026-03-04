Quickstart
==========

Basic usage — load data & compute derivatives
----------------------------------------------

``pvtend`` ships with an idealized Gaussian PV anomaly that undergoes
simultaneous propagation, intensification, and deformation over one
hour.  Let's load it, compute all the spatial/temporal derivatives, and
visualise them.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import pvtend

   # ── 1. Load bundled sample data ──────────────────────────────────
   d = pvtend.load_idealized_pv()
   q0, q1 = d["q0"], d["q1"]          # PV at t=0 and t=1   [PVU]
   x_km, y_km = d["x_km"], d["y_km"]  # coordinate vectors   [km]
   x_deg, y_deg = d["x_deg"], d["y_deg"]
   dx_arr, dy_m, dt = d["dx_arr"], float(d["dy_m"]), float(d["dt"])

   print(f"Grid shape: {q0.shape},  Δx = {float(d['dx_m'])/1e3:.0f} km,  Δt = {dt:.0f} s")

   # ── 2. Derivatives ───────────────────────────────────────────────
   dq_dx = pvtend.ddx(q0, dx_arr, periodic=False)  # ∂q/∂x  [PVU/m]
   dq_dy = pvtend.ddy(q0, dy_m)                     # ∂q/∂y  [PVU/m]
   dq_dxdy = pvtend.ddy(dq_dx, dy_m)                # ∂²q/∂x∂y [PVU/m²]
   dq_dt = (q1 - q0) / dt                           # Δq/Δt  [PVU/s]

   # ── 3. Visualise (2 rows × 3 columns) ────────────────────────────
   panels = [
       (q0,      r"$q_0$"),
       (q1,      r"$q_1$"),
       (dq_dt,   r"$\Delta q / \Delta t$"),
       (dq_dx,   r"$\partial q / \partial x$"),
       (dq_dy,   r"$\partial q / \partial y$"),
       (dq_dxdy, r"$\partial^2 q / \partial x \partial y$"),
   ]

   fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
   for ax, (fld, title) in zip(axes.ravel(), panels):
       vm = np.nanmax(np.abs(fld)) or 1.0
       im = ax.contourf(x_km, y_km, fld, 21, vmin=-vm, vmax=vm, cmap="RdBu_r")
       ax.set_title(title, fontsize=12)
       ax.set_aspect("equal")
       plt.colorbar(im, ax=ax, shrink=0.75)
   fig.suptitle("Idealized PV evolution and spatial derivatives", fontsize=14)
   plt.show()

|derivatives_plot|

.. |derivatives_plot| raw:: html

   <a href="https://pvtend.readthedocs.io/en/latest/_images/quickstart_derivatives.png">
     <img src="https://pvtend.readthedocs.io/en/latest/_images/quickstart_derivatives.png"
          alt="PV fields and spatial derivatives (2×3 panel)" style="width:100%" />
   </a>


Orthogonal basis decomposition
------------------------------

Using the derivatives from above, build the four orthogonal bases
(intensification / propagation / deformation) via Gram-Schmidt, project
the PV tendency, and compare the reconstruction with the original.

.. code-block:: python

   from pvtend import compute_orthogonal_basis, project_field

   # ── 1. Build orthogonal basis ────────────────────────────────────
   #   pv_anom = q0  (the anomaly at t=0)
   #   smoothing_deg and grid_spacing are in degrees
   grid_sp = float(d["grid_spacing_deg"])

   basis = compute_orthogonal_basis(
       pv_anom=q0,
       pv_dx=dq_dx,
       pv_dy=dq_dy,
       x_rel=x_deg,
       y_rel=y_deg,
       mask_negative=True,
       apply_smoothing=True,
       smoothing_deg=3.0,
       grid_spacing=grid_sp,
   )

   # ── 2. Project tendency onto basis ───────────────────────────────
   result = project_field(dq_dt, basis)

   print(f"β  (intensification) = {result['beta']:.3e} s⁻¹")
   print(f"αx (zonal prop.)     = {result['ax']:.1f} m/s")
   print(f"αy (merid. prop.)    = {result['ay']:.1f} m/s")
   print(f"γ  (deformation)     = {result['gamma']:.3e} m² s⁻¹")
   print(f"RMSE                 = {result['rmse']:.3e}")

   # ── 3. Visualise (2 rows × 3 columns) ────────────────────────────
   panels2 = [
       (result["int"],   rf"INT ($\beta$={result['beta']:.2e} s$^{{-1}}$)"),
       (result["prop"],  rf"PRP ($\alpha_x$={result['ax']:.1f}, $\alpha_y$={result['ay']:.1f} m/s)"),
       (result["def"],   rf"DEF ($\gamma$={result['gamma']:.2e} m$^2$ s$^{{-1}}$)"),
       (dq_dt,           r"Original $\Delta q / \Delta t$"),
       (result["recon"], r"Reconstructed $\hat{F}$"),
       (result["resid"], "Residual (error)"),
   ]

   fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
   for ax, (fld, title) in zip(axes.ravel(), panels2):
       vm = np.nanmax(np.abs(fld)) or 1.0
       im = ax.contourf(x_km, y_km, fld, 21, vmin=-vm, vmax=vm, cmap="RdBu_r")
       ax.set_title(title, fontsize=11)
       ax.set_aspect("equal")
       plt.colorbar(im, ax=ax, shrink=0.75)
   fig.suptitle("Basis decomposition — smoothed + Gram-Schmidt", fontsize=14)
   plt.show()

|decomposition_plot|

.. |decomposition_plot| raw:: html

   <a href="https://pvtend.readthedocs.io/en/latest/_images/quickstart_decomposition.png">
     <img src="https://pvtend.readthedocs.io/en/latest/_images/quickstart_decomposition.png"
          alt="Orthogonal basis decomposition and reconstruction (2×3 panel)" style="width:100%" />
   </a>


Command-line pipeline
---------------------

The full production pipeline is a **three-pass** workflow:

1. **Compute** (Pass 0) — extract PV tendency terms for each tracked event into
   per-timestep NPZ files.  This is the most expensive step; with
   ``--n-workers 48`` it takes **~12 hours** for ~500 events.
2. **Classify** (Pass 1) — detect Rossby Wave Breaking on the NPZ
   patches and label each event as AWB, CWB, or neutral (~minutes).
3. **Composite** (Pass 2) — accumulate NPZ fields into variant-aware
   composites, stratified by RWB type (~seconds).

.. code-block:: bash

   # ── Pass 0: Compute PV tendencies ────────────────────────────────
   # Recommended: 48 parallel workers → ~12 h for ~500 blocking events
   pvtend-pipeline compute \
       --event-type blocking \
       --events-csv tracked_events.csv \
       --era5-dir /path/to/era5/ \
       --clim-path /path/to/climatology/era5_hourly_clim.nc \
       --out-dir /path/to/output/ \
       --dh-range '-49:25:1' \
       --center-mode eulerian \
       --n-workers 48 \
       --skip-existing

   # ── Pass 1: RWB classification ───────────────────────────────────
   #   Detects overturning PV contours on multiple pressure levels and
   #   classifies each event-stage as AWB / CWB / neutral.
   pvtend-pipeline classify \
       --npz-dir /path/to/output/ \
       --output /path/to/outputs/rwb_variant_tracksets.pkl \
       --levels 500 400 300 200 \
       --threshold 3

   # ── Pass 2: Variant-aware composite ──────────────────────────────
   #   Accumulates NPZ fields, separately for "original", "AWB_onset",
   #   "CWB_peak", etc.  Produces a single composite.pkl.
   pvtend-pipeline composite \
       --npz-dir /path/to/output/ \
       --rwb-pkl /path/to/outputs/rwb_variant_tracksets.pkl \
       --pkl-out /path/to/outputs/composite_blocking.pkl

   # ── (Optional) Pass 3: Orthogonal basis decomposition ────────────
   pvtend-pipeline decompose \
       --pkl-in /path/to/outputs/composite_blocking.pkl \
       --out-dir /path/to/outputs/decomposition/

The resulting ``composite_blocking.pkl`` can be loaded and inspected in
Python:

.. code-block:: python

   from pvtend import load_composite_state

   comp = load_composite_state("composite_blocking.pkl")

   # Get composite-mean PV at peak + 0 h for AWB events
   pv_awb = comp.composite_mean_3d("pv_3d", stage="peak", dh=0, variant="AWB_peak")
