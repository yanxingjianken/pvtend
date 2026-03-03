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

.. raw:: html

   <style>
   .gallery-carousel { position: relative; max-width: 90%; margin: 0 auto 1.5em; }
   .gallery-slide { display: none; text-align: center; }
   .gallery-slide.active { display: block; }
   .gallery-slide img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
   .gallery-slide .caption { font-style: italic; font-size: 0.92em; margin-top: 0.4em; color: #555; }
   .gallery-nav { text-align: center; margin-top: 0.6em; }
   .gallery-nav button { font-size: 1.1em; padding: 0.3em 1.2em; margin: 0 0.4em;
     cursor: pointer; border: 1px solid #aaa; border-radius: 4px; background: #f7f7f7; }
   .gallery-nav button:hover { background: #e0e0e0; }
   .gallery-counter { text-align: center; font-weight: bold; margin-bottom: 0.3em; font-size: 0.95em; }
   </style>

   <div class="gallery-carousel" id="pvtend-gallery">
     <div class="gallery-counter" id="gallery-counter">1 / 3</div>
     <div class="gallery-slide active" data-idx="0">
       <img src="_static/reconstruction_demo.png"
            alt="Idealized four-basis reconstruction demo"/>
       <div class="caption">
         Idealized validation: a Gaussian PV anomaly with prescribed
         propagation, intensification, and deformation is decomposed into
         four orthogonal bases and reconstructed with near-zero residual.
       </div>
     </div>
     <div class="gallery-slide" data-idx="1">
       <img src="_static/lifecycle_demo.gif"
            alt="Real blocking lifecycle decomposition"/>
       <div class="caption">
         Real ERA5 blocking event (track 425) &mdash; animated lifecycle showing
         total PV on a cartopy map (left) and the four projected basis
         components (right) evolving from 13&thinsp;h pre-onset to 12&thinsp;h post-onset.
         The analysis is done on a weighted average surface across 300, 250, 200&thinsp;hPa levels.
       </div>
     </div>
     <div class="gallery-slide" data-idx="2">
       <img src="_static/z_lifecycle_demo.gif"
            alt="Geopotential-height lifecycle decomposition"/>
       <div class="caption">
         Geopotential-height (Z500) variant of the four-basis decomposition
         (track 425) &mdash; animated lifecycle showing Z anomaly from
         the 1990&ndash;2020 hourly climatology, with adaptive prenorm and
         blockid contour overlay. See notebook
         <code>03z_four_basis_projection_geopotential</code>.
       </div>
     </div>
     <div class="gallery-nav">
       <button onclick="pvtGallery(-1)">&larr; Prev</button>
       <button onclick="pvtGallery(1)">Next &rarr;</button>
     </div>
   </div>

   <script>
   (function(){
     var idx = 0, slides = document.querySelectorAll('#pvtend-gallery .gallery-slide'),
         counter = document.getElementById('gallery-counter');
     window.pvtGallery = function(d) {
       slides[idx].classList.remove('active');
       idx = (idx + d + slides.length) % slides.length;
       slides[idx].classList.add('active');
       counter.textContent = (idx+1) + ' / ' + slides.length;
     };
   })();
   </script>

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
