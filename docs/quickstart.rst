Quickstart
==========

Basic usage
-----------

.. code-block:: python

   import pvtend

   # Check version
   print(pvtend.__version__)

   # Compute derivatives
   import numpy as np
   R_EARTH = 6.371e6
   field = np.random.randn(21, 41)
   lat_rad = np.deg2rad(np.linspace(30, 50, 21))
   dlon_rad = np.deg2rad(0.5)
   dx_arr = R_EARTH * np.cos(lat_rad)[:, None] * dlon_rad
   dfdx = pvtend.ddx(field, dx_arr)

Command-line pipeline
---------------------

.. code-block:: bash

   # Compute PV tendencies for blocking events
   pvtend-pipeline compute \
       --event-type blocking \
       --events-csv tracked_events.csv \
       --era5-dir /path/to/era5/ \
       --clim-dir /path/to/climatology/ \
       --out-dir /path/to/output/ \
       --skip-existing

   # Aggregate into composite
   pvtend-pipeline composite \
       --npz-dir /path/to/output/ \
       --pkl-out composite_blocking.pkl

Orthogonal basis decomposition
------------------------------

.. code-block:: python

   from pvtend.decomposition import compute_orthogonal_basis, project_field

   basis = compute_orthogonal_basis(
       pv_anom=q_anomaly,
       pv_dx=dq_dx,
       pv_dy=dq_dy,
       x_rel=x_coords,
       y_rel=y_coords,
       mask_negative=True,
       apply_smoothing=True,
       smoothing_deg=6.0,
       grid_spacing=1.5,
   )

   result = project_field(dq_dt, basis)
   print(f"β (intensification) = {result['beta']:.3e}")
   print(f"αx (zonal prop.) = {result['ax']:.3e}")
