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
   field = np.random.randn(21, 41)
   dfdx = pvtend.ddx(field, dlon_rad=np.deg2rad(0.5),
                      lat_1d=np.linspace(30, 50, 21))

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
       q_prime=q_anomaly,
       q_full=q_total,
       x_rel=x_coords,
       y_rel=y_coords,
       dx=dx_meters,
       dy=dy_meters,
   )

   result = project_field(
       tendency_field=dq_dt,
       basis=basis,
       dx=dx_meters,
       dy=dy_meters,
   )
   print(f"β (intensification) = {result['beta']:.3e}")
   print(f"αx (zonal prop.) = {result['ax']:.3e}")
