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

.. image:: _static/reconstruction_demo.png
   :alt: Idealized four-basis reconstruction demo
   :width: 100%
   :align: center

*Idealized validation: a Gaussian PV anomaly with prescribed propagation,
intensification, and deformation is decomposed into four orthogonal bases
and reconstructed with near-zero residual.*

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
