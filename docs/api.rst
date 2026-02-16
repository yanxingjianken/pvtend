API Reference
=============

.. currentmodule:: pvtend

Grid & Constants
----------------

.. autosummary::
   :toctree: generated/

   NHGrid
   default_nh_grid
   EventPatch
   constants

Derivatives
-----------

.. autosummary::
   :toctree: generated/

   ddx
   ddy
   ddp
   ddt

Climatology
-----------

.. autosummary::
   :toctree: generated/

   compute_climatology
   load_climatology

QG Omega Equation
-----------------

.. autosummary::
   :toctree: generated/

   solve_qg_omega

Helmholtz Decomposition
-----------------------

.. autosummary::
   :toctree: generated/

   helmholtz_decomposition
   helmholtz_decomposition_3d

Moist/Dry Omega
---------------

.. autosummary::
   :toctree: generated/

   decompose_omega

Orthogonal Basis Decomposition
------------------------------

.. autosummary::
   :toctree: generated/

   OrthogonalBasisFields
   compute_orthogonal_basis
   project_field
   collect_term_fields

RWB Detection
-------------

.. autosummary::
   :toctree: generated/

   RWBConfig
   detect_rwb_events

Composites
----------

.. autosummary::
   :toctree: generated/

   CompositeState
   load_composite_state

Tendency Pipeline
-----------------

.. autosummary::
   :toctree: generated/

   TendencyConfig
   TendencyComputer

I/O
---

.. autosummary::
   :toctree: generated/

   io.load_era5_month
   io.open_months_dataset
   io.load_npz_patch
   io.list_npz_patches
   io.load_pkl
   io.save_pkl

Plotting
--------

.. autosummary::
   :toctree: generated/

   plotting.plot_four_basis
   plotting.plot_coefficient_curves
   plotting.plot_field_2d
   plotting.plot_wind_overlay
