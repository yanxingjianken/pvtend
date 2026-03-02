pvtend: PV Tendency Decomposition
=================================

|version| · `Source on GitHub <https://github.com/yanxingjianken/pvtend>`_ · `README <https://github.com/yanxingjianken/pvtend#readme>`_

.. |version| replace:: **v\ |release|**

**PV tendency decomposition for atmospheric blocking, propagating anticyclones,
and all synoptic-scale cyclonic event lifecycle analysis.**

``pvtend`` diagnoses the growth, propagation, and decay of mid-latitude weather
events by decomposing potential vorticity (PV) tendencies from ERA5
pressure-level data onto physically meaningful components using an orthogonal
basis framework.  This is the Part I work of Yan et al. (in prep.) about
blocking lifecycle analyses on onset, peak, decay stages.

Gallery
-------

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. image:: _static/reconstruction_demo.png
          :alt: Idealized four-basis reconstruction demo
          :width: 100%

       *Idealized validation: a Gaussian PV anomaly with prescribed
       propagation, intensification, and deformation is decomposed into
       four orthogonal bases and reconstructed with near-zero residual.*

     - .. image:: _static/lifecycle_demo.gif
          :alt: Real blocking lifecycle decomposition
          :width: 100%

       *Real ERA5 blocking event (track 425) — animated lifecycle showing
       total PV on a cartopy map (left) and the four projected basis
       components (right) evolving from 13 h pre-onset to 12 h post-onset. 
       The analysis is done on a weighted average surface across 300, 250, 200 hPa levels.*

Event catalogues
~~~~~~~~~~~~~~~~

Blocking and PRP-high events are identified as persistent anticyclonic anomalies
in 500 hPa geopotential height using
`TempestExtremes v2.1 <https://gmd.copernicus.org/articles/14/5023/2021/>`_
to track contiguous Z500 anomaly features that exceed a fixed threshold for
≥5 days, producing CSV catalogues with columns for event ID, centre lat/lon,
onset/peak/decay timestamps, and area.

**Sample catalogue (ERA5, 1990–2020 blocking):**
:download:`ERA5_TempestExtremes_z500_anticyclone_blocking.csv <_static/ERA5_TempestExtremes_z500_anticyclone_blocking.csv>`
(`view on GitHub <https://github.com/yanxingjianken/pvtend/blob/main/docs/_static/ERA5_TempestExtremes_z500_anticyclone_blocking.csv>`__)

Features
--------

- **PV tendency computation** — zonal advection, baroclinic counter propagation,
  vertical advection, and approximated diabatic heating terms.
- **QG omega solver** — Hoskins Q-vector formulation with FFT+Thomas (default)
  and 3-D direct/iterative (BiCGSTAB+ILU) backends.
- **Helmholtz decomposition** — 4 backends (direct, FFT, DCT, SOR) for
  limited-area domains.
- **Moist/dry omega splitting** — decomposes vertical motion into moist and dry
  contributions.
- **Isentropic diagnostics** — PV-tendency analysis on isentropic surfaces.
- **Orthogonal basis decomposition** — projects PV tendency onto intensification
  (β), propagation (αx, αy), and deformation (γ) modes.
- **Composite explorer** — ``plot_var`` for bootstrap-significant spatial maps
  with shared colorbars; ``plot_baroclinic_tilt`` for two-level v′ overlay.
- **RWB detection** — anticyclonic/cyclonic Rossby wave breaking classification.
- **Composite lifecycle** — multi-stage ensemble averaging with onset/peak/decay
  staging.
- **CLI pipeline** — end-to-end processing via ``pvtend-pipeline``.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   notebooks
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
