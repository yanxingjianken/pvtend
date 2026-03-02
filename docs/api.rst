API Reference
=============

.. currentmodule:: pvtend

Every public function and class in **pvtend** is documented below with
full parameter descriptions.  Click **[source]** next to any entry to
jump directly to the implementation.


Grid & Constants
----------------

The grid module provides the standard 1.5° Northern-Hemisphere grid used
throughout the package, spatial cropping/interpolation utilities, and
event-centred patch extraction.  Constants include Earth parameters,
thermodynamic constants, and default level lists.

.. autosummary::
   :toctree: generated/

   NHGrid
   default_nh_grid
   EventPatch
   constants


Derivatives
-----------

Centred finite-difference operators on lat-lon-pressure grids.  All
operators handle:

* **Periodic zonal boundaries** — the globe wraps in longitude
  (toggled via ``periodic`` flag).
* **One-sided differences** at polar / top / bottom boundaries.
* **Non-uniform pressure spacing** for :func:`ddp`.

``ddx``, ``ddy`` accept both 2-D ``(nlat, nlon)`` and 3-D
``(nlev, nlat, nlon)`` inputs; ``ddp`` operates along axis 0;
``ddt`` uses centred differences with one-sided at the temporal
boundaries.

.. autosummary::
   :toctree: generated/

   ddx
   ddy
   ddp
   ddt


Climatology
-----------

Compute and load an hourly climatology for ERA5 pressure-level variables.
The climatology is smoothed in time (day-of-year) using a low-pass
**Fourier filter** (4 harmonics) and optionally with 2 zonal modes,
producing per-variable, per-month NetCDF files for anomaly computation.

.. autosummary::
   :toctree: generated/

   compute_climatology
   load_climatology


.. _qg-omega:

QG Omega Equation
-----------------

Solves the Hoskins (1978) quasi-geostrophic omega equation:

.. math::

   \nabla^2_p \omega + \frac{f_0^2}{\sigma}\,\frac{\partial^2\omega}
   {\partial p^2}
   \;=\; -2\,\nabla\cdot\mathbf{Q}
         - \beta\,\frac{R_d}{\sigma\,p}\,\frac{\partial T}{\partial x}

where :math:`\mathbf{Q}` is the Q-vector (Hoskins et al. 1978).

Three solver methods
~~~~~~~~~~~~~~~~~~~~

1. **sp19** *(default)* — Steinfeld & Pfahl (2019) empirical scaling.
   Sets :math:`\omega_\text{dry} = \frac{1}{3}\,\omega_\text{total}`.
   Zero-cost, preserves spatial structure, requires ``omega_total``
   as input.

2. **fft** — FFT in longitude (periodic) + Thomas tridiagonal in
   pressure.  Drops :math:`\partial^2\omega/\partial y^2` from the LHS
   but retains :math:`\partial^2\omega/\partial x^2` via the spectral
   representation.  Fast (~2 s per event) and captures >90 % of the
   spatial variance.  Reference: Schumann & Sweet (1988).

3. **log20** — Strongly Implicit Procedure (SIP, Stone 1968) with a
   full 3-D spherical finite-difference stencil including the
   :math:`\tan\varphi` metric term.  Closest analogue to
   Li & O'Gorman (2020).  Numba-accelerated when available (~3–6 s
   per event pair); falls back to pure-Python if ``numba`` is not
   installed.

All methods apply a **latitude taper** (linearly 0 → 1 between 15°N and
25°N, tapering back to 0 above 80°N) to enforce QG validity.

Boundary conditions: :math:`\omega = 0` at top/bottom pressure levels
and at lateral boundaries (Dirichlet for local patches; periodic for
full rings).

**References:** Hoskins B J, Draghici I, Davies H C (1978) *Q.J.R.M.S.*
104, 31–38.  Steinfeld D, Pfahl S (2019) *Clim. Dyn.* 53, 6159–6180.
Li L, O'Gorman P A (2020) *J. Climate*.  Stone H L (1968) *SIAM J.
Numer. Anal.* 5, 530–558.

.. autosummary::
   :toctree: generated/

   solve_qg_omega
   solve_qg_omega_sip


.. _helmholtz:

Helmholtz Decomposition
-----------------------

Decomposes a 2-D wind field :math:`(u, v)` into three orthogonal
components:

.. math::

   \mathbf{u} = \underbrace{\mathbf{u}_\text{rot}}_{\text{non-divergent}}
              + \underbrace{\mathbf{u}_\text{div}}_{\text{irrotational}}
              + \underbrace{\mathbf{u}_\text{har}}_{\text{harmonic}}

where the rotational wind derives from a streamfunction
:math:`\psi` (:math:`u_\text{rot}=-\partial\psi/\partial y`,
:math:`v_\text{rot}=\partial\psi/\partial x`),
the divergent wind from a velocity potential :math:`\chi`
(:math:`u_\text{div}=\partial\chi/\partial x`,
:math:`v_\text{div}=\partial\chi/\partial y`),
and the harmonic residual satisfies :math:`\nabla\cdot\mathbf{u}_\text{har}=0`
*and* :math:`\zeta(\mathbf{u}_\text{har})=0`.

The decomposition proceeds by:

1. Computing vorticity :math:`\zeta` and divergence :math:`\delta` from
   the input wind.
2. Solving Poisson equations :math:`\nabla^2\psi=\zeta` and
   :math:`\nabla^2\chi=\delta` with one of four backend solvers.
3. Recovering :math:`(u_\text{rot},v_\text{rot})` and
   :math:`(u_\text{div},v_\text{div})` via gradient operators.
4. Computing the harmonic residual by subtraction.

Four Poisson solver backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 40 25 20
   :header-rows: 1

   * - ``method``
     - Description
     - BCs
     - Speed
   * - ``'direct'``
     - Sparse LU (Lynch 1989). Builds the 5-point Laplacian as a sparse
       CSC matrix with **exact variable** :math:`\Delta x(\varphi)` and
       solves via ``scipy.sparse.linalg.spsolve``.
     - Dirichlet :math:`\phi=0`
     - Moderate
   * - ``'fft'``
     - FFT in longitude + Thomas tridiagonal in latitude
       (Schumann & Sweet 1988).  Exact variable
       :math:`\Delta x(\varphi)` via per-wavenumber tridiagonal.
     - Periodic in lon, Dirichlet in lat
     - Fast
   * - ``'dct'``
     - DST-I with constant ``mean(dx)``.  Fully spectral
       :math:`O(N\log N)` — fastest backend, but loses accuracy at high
       latitudes where :math:`\Delta x` varies with
       :math:`\cos\varphi`.
     - Dirichlet all sides
     - Fastest
   * - ``'sor'``
     - Successive Over-Relaxation iteration (5-point stencil).
       ~1000× slower than direct; kept only for verification.
     - Dirichlet :math:`\phi=0`
     - Very slow

**References:** Lynch P (1989) *MWR* 117, 1492–1500.
Schumann U & Sweet R (1988) *J. Comput. Phys.* 75, 123–137.

.. autosummary::
   :toctree: generated/

   helmholtz_decomposition
   helmholtz_decomposition_3d

**Low-level Poisson solvers** (used internally; importable from
``pvtend.helmholtz``):

.. currentmodule:: pvtend.helmholtz

.. autosummary::
   :toctree: generated/

   solve_poisson_direct
   solve_poisson_fft
   solve_poisson_dct
   solve_poisson_sor
   compute_vorticity_divergence

.. currentmodule:: pvtend


Moist/Dry Omega
---------------

Partitions total vertical velocity into dry (QG) and moist (residual)
components, then recovers the moist divergent wind:

1. :math:`\omega_\text{moist} = \omega_\text{total} - \omega_\text{dry}`
2. Solve :math:`\nabla^2\chi_\text{moist} = -\partial\omega_\text{moist}/\partial p`
   on each pressure level (FFT Poisson solver).
3. :math:`(u_\text{div,moist}, v_\text{div,moist}) = \nabla\chi_\text{moist}`
4. Dry divergent wind by subtraction:
   :math:`\mathbf{u}_\text{div,dry} = \mathbf{u}_\text{div} - \mathbf{u}_\text{div,moist}`

.. autosummary::
   :toctree: generated/

   decompose_omega


Orthogonal Basis Decomposition
------------------------------

Projects PV tendency fields onto a set of four **quadrupole** basis
functions constructed via Gram-Schmidt orthogonalisation.  Each basis
captures a distinct dynamical mode of blocking intensification /
weakening:

1. :math:`\Phi_1` — Monopole (uniform PV tendency)
2. :math:`\Phi_2` — Dipole N-S (meridional shift)
3. :math:`\Phi_3` — Dipole E-W (zonal propagation)
4. :math:`\Phi_4` — Quadrupole (intensification / weakening)

The projection coefficients :math:`c_i(t)` quantify each term's
contribution to the blocking lifecycle at each timestep.

.. autosummary::
   :toctree: generated/

   OrthogonalBasisFields
   compute_orthogonal_basis
   project_field
   collect_term_fields


RWB Detection
-------------

Identifies Rossby Wave Breaking events by detecting overturning (folding)
of geopotential-height or PV contours and classifying them as
**Anticyclonic Wave Breaking (AWB)** or **Cyclonic Wave Breaking (CWB)**.

Two classification methods are available, selected via the ``method``
argument of :func:`detect_rwb_events`:

.. list-table:: Classification methods
   :header-rows: 1
   :widths: 15 45 40

   * - ``method``
     - Algorithm
     - When to use
   * - ``"bay"`` *(default, recommended)*
     - MATLAB-consistent **path-order** of max / min latitude
       intersections across sample meridians.
     - **Recommended for circumpolar-cropped contours** and standard
       event-centred patches.  Reproduces Peters & Waugh (1996).
   * - ``"tilt"``
     - **Centerline-tilt slope** of the overturning envelope.
       Slope < −0.15 → AWB; slope > +0.15 → CWB; otherwise UNK.
     - Alternative metric useful for sensitivity analysis. The ±0.15
       dead-zone threshold is stored in :data:`TILT_SLOPE_THRESHOLD`.

Circumpolar-first approach (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When analysing RWB on cropped patches the recommended workflow is:

1. Extract **circumpolar contours** from the full Northern-Hemisphere
   Z field using :func:`circumpolar_contours`.  This finds contours
   that span ≥ 355° longitude (i.e. truly circumpolar), mimicking the
   MATLAB ``blue_circumcontour`` approach with periodic doubling.
2. **Crop** each circumpolar contour to the event-centred patch with
   :func:`crop_contour_to_patch`, returning Δlon/Δlat coordinates.
3. Pass the full-NH field (``field_nh``, ``lat_nh``, ``lon_nh``) and
   patch centre (``centre_lat``, ``centre_lon``) to
   :func:`detect_rwb_events`, which performs steps 1–2 automatically.

If no full-NH field is supplied, :func:`detect_rwb_events` falls back to
:func:`sampled_longest_contours` on the local patch (legacy mode).

Algorithm overview
^^^^^^^^^^^^^^^^^^

1. Obtain contours (circumpolar-first or legacy local-patch sampling).
2. Detect overturning via meridian intersection counting
   (:func:`overturn_x_intervals`).
3. Classify AWB / CWB using the chosen method
   (:func:`classify_bay` or :func:`classify_tilt`).
4. Apply centroid-x gates to filter spurious detections.
5. Construct envelope polygons for visualisation.

Multi-level reduction
^^^^^^^^^^^^^^^^^^^^^

For 3-D (nlev × nlat × nlon) inputs, :func:`reduce_to_2d` extracts a
single pressure level or computes an exponential height-weighted vertical
average (``level_mode="wavg"``).  The weighting uses
:func:`weighted_mean_2d` with the atmospheric e-folding height scale.

**References:** Peters D, Waugh D W (1996) *J. Atmos. Sci.* 53,
3013–3031.  Thorncroft C D, Hoskins B J, McIntyre M E (1993)
*Q.J.R.M.S.* 119, 17–55.

.. autosummary::
   :toctree: generated/

   RWBConfig
   detect_rwb_events
   circumpolar_contours
   crop_contour_to_patch
   classify_bay
   classify_tilt
   centerline_tilt
   overturn_x_intervals
   envelope_polygon
   sampled_longest_contours
   reduce_to_2d
   weighted_mean_2d
   nearest_level_index
   TILT_SLOPE_THRESHOLD


Composites
----------

Accumulates per-event NPZ patch fields into running sums (grouped by
event stage and RWB variant) and exports / loads the composite state via
pickle.  :class:`CompositeState` provides ``composite_mean_3d()`` and
``composite_reduce()`` methods for rapid access to composite-mean fields.

.. autosummary::
   :toctree: generated/

   CompositeState
   load_composite_state


Isentropic Interpolation
-------------------------

Interpolates 3-D isobaric fields onto constant-:math:`\theta` (isentropic)
surfaces using the algorithm of Ziv & Alpert (1994) as implemented in
MetPy v1.7.

**Algorithm (Newton–Raphson on Poisson's equation):**

1. Sort pressure levels in **descending** order (surface → top).
2. Compute :math:`\theta = T\,(P_0/p)^\kappa` at every grid point.
3. For each target :math:`\theta^*`, find bounding pressure levels.
4. Assume :math:`T` varies linearly with :math:`\ln p` between
   bounding levels.
5. Solve for :math:`p^*` via Newton–Raphson:

   .. math::

      f(\ln p) = \theta^* - (a\,\ln p + b)\,P_0^\kappa\,
                 e^{-\kappa\,\ln p}

6. Interpolate all auxiliary fields to the discovered
   :math:`p^*` surface using log-pressure weighting.

Convenience wrappers handle multi-event NPZ data directly.

**References:** Ziv A, Alpert P (1994).
`MetPy isentropic_interpolation source
<https://github.com/Unidata/MetPy/blob/main/src/metpy/calc/basic.py>`_.

.. autosummary::
   :toctree: generated/

   isentropic_interpolation
   isentropic_interpolation_pressure
   interp_event_fields_to_theta
   interp_event_field_to_single_theta


Tendency Pipeline
-----------------

Orchestrates the full per-event computation:

1. Load ERA5 data for the time window.
2. Subtract hourly climatology → anomalies.
3. Compute all spatial / temporal derivatives.
4. FFT Helmholtz decomposition on the full NH hemisphere.
5. QG omega → :math:`\omega_\text{dry}`.
6. Moist / dry decomposition → :math:`\omega_\text{moist}`,
   :math:`\chi_\text{moist}`.
7. Extract event-centred patches.
8. Compute PV cross-terms and vertical weighted averages.
9. Write per-timestep NPZ files.

:class:`TendencyComputer` is parameterised by event type
(blocking / PRP), eliminating code duplication between scripts.

.. autosummary::
   :toctree: generated/

   TendencyConfig
   TendencyComputer


I/O
---

Utilities for loading ERA5 monthly NetCDF files, per-event NPZ patches,
and composite-state pickle files.

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

Publication-quality visualisation tools for PV tendency analysis.

Basis & coefficient plots
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   plotting.plot_four_basis
   plotting.plot_basis_with_contours
   plotting.plot_coefficient_curves
   plotting.plot_field_2d
   plotting.plot_wind_overlay


.. _composite-explorer:

Composite explorer
~~~~~~~~~~~~~~~~~~

:func:`~pvtend.plotting.plot_var` is a **self-contained** single-variable
composite viewer that loads NPZ events, computes a bootstrap significance
mask (N = 1000, 95 % CI by default), and optionally projects onto the
dh − 1 orthogonal basis.

**Two layout modes:**

* ``projection=False`` → **2-panel** figure: composite mean + hatched
  bootstrap significance.
* ``projection=True``  → **6-panel** figure: adds 2 × 2 projection rows
  (INT / PRP / DEF / Residual) with β, αₓ, αy, γ in the subtitle.

**Supported options:**

.. list-table::
   :widths: 20 50
   :header-rows: 1

   * - Parameter
     - Description
   * - ``data_root``
     - Path to composite archive (blocking **or** PRP)
   * - ``stage``
     - ``"onset"`` / ``"peak"`` / ``"decay"``
   * - ``dh``
     - Lifecycle hour offset (any integer)
   * - ``level``
     - ``"wavg"`` or int hPa (e.g. 200, 500)
   * - ``var_spec``
     - ``str``, ``list[str]`` (``'-'``-negation), or ``callable(event)``
   * - ``projection``
     - ``True`` for 6-panel, ``False`` for 2-panel

Helper functions :func:`~pvtend.plotting.load_events`,
:func:`~pvtend.plotting.get_field`, and
:func:`~pvtend.plotting.bootstrap_sig` are also public so notebooks can
do ad-hoc analysis without re-implementing loaders.

.. autosummary::
   :toctree: generated/

   plotting.plot_var
   plotting.load_events
   plotting.get_field
   plotting.bootstrap_sig


.. _baroclinic-tilt:

Baroclinic tilt overlay
~~~~~~~~~~~~~~~~~~~~~~~

:func:`~pvtend.plotting.plot_baroclinic_tilt` shows the **westward tilt
with height** characteristic of baroclinic blocking by overlaying
upper-level v′ (bold black contours, significant only) on lower-level v′
(blue-red shading).  Panels are generated per lifecycle stage.

.. autosummary::
   :toctree: generated/

   plotting.plot_baroclinic_tilt
