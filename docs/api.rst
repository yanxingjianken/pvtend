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
   d_dlambda
   d_dphi
   div_spherical


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

Two solver methods
~~~~~~~~~~~~~~~~~~

1. **log20** *(default)* — Strongly Implicit Procedure (SIP, Stone 1968)
   with a full 3-D spherical finite-difference stencil including the
   :math:`\tan\varphi` metric term.  Closest analogue to
   Li & O'Gorman (2020).  Numba-accelerated (``nogil=True``) when
   available (~3–6 s per event); falls back to pure-Python if ``numba``
   is not installed.  Always solved on the **full Northern Hemisphere**
   grid (periodic in longitude, Dirichlet at equatorial and polar faces)
   and the event patch is extracted afterward.

2. **sp19** — Steinfeld & Pfahl (2019) empirical scaling.
   Sets :math:`\omega_\text{dry} = \frac{1}{3}\,\omega_\text{total}`.
   Zero-cost, preserves spatial structure.  QG-moist ω equals the moist
   residual (no separate term-C solve).

Both methods apply a **latitude taper** (linearly 0 → 1 between 15°N and
25°N, tapering back to 0 above 85°N) to all RHS terms (A, B, and C)
to suppress polar metric-term singularities and enforce QG validity.

Boundary conditions (LOG20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inhomogeneous Dirichlet: **real ERA5** :math:`\omega` is prescribed on
all six faces (top, bottom, and four lateral boundaries) via the
``omega_b`` parameter.  This anchors the QG inversion to the reanalysis
vertical velocity field at the domain edges, eliminating the bias of
homogeneous (:math:`\omega = 0`) boundaries.

Diabatic heating — term C
~~~~~~~~~~~~~~~~~~~~~~~~~

The solver supports **two** independent diabatic forcings, and is called
**three** times per timestep:

**LOG20 term C** (budget-closure, Li & O'Gorman 2020):
When the Eulerian temperature tendency :math:`\partial T/\partial t` is
available (``dT_dt`` parameter), the diabatic heating rate is diagnosed
from the thermodynamic equation:

.. math::

   J = \underbrace{c_p\!\left(\frac{\partial T}{\partial t}
       + \mathbf{v}\cdot\nabla_p T\right)}_{J_1}
     + \underbrace{\left(-\frac{\sigma\,p}{R_d}\;c_p\;\omega\right)}_{J_2}

.. math::

   \text{Term C}_\text{LOG20} = -\frac{\kappa}{p}\;\nabla^2_s\,J

This captures **all** diabatic processes (LHR + radiation + sensible
heat flux + diffusion).

**Emanuel term** :math:`C_\text{em}` (parameterised LHR, Emanuel 1987;
Tamarin & Kaspi 2016): The latent heat release is parameterised from
thermodynamic profiles during saturated ascent (:math:`\omega < 0`):

.. math::

   \dot\theta_\text{LHR} = \omega\!\left(\frac{\partial\theta}{\partial p}
   - \frac{\gamma_m}{\gamma_d}\;\frac{\theta}{\theta_E}\;
     \frac{\partial\theta_E}{\partial p}\right)

.. math::

   J_\text{em} = c_p\;\dot\theta_\text{LHR}\;\frac{T}{\theta},
   \qquad
   C_\text{em} = -\frac{\kappa}{p}\;\nabla^2_s\,J_\text{em}

This captures **only** latent heating during saturated ascent.

**Three-solve strategy:**

1. **QG-dry** (terms A+B, homogeneous top/bottom BCs) →
   :math:`\omega_\text{dry}`
2. **QG-total** (A+B+\ :math:`C_\text{LOG20}`, ERA5 BCs) →
   :math:`\omega_\text{qg,total}`
3. **Emanuel** (A+B+\ :math:`C_\text{em}`, ERA5 BCs) →
   :math:`\omega_\text{em}`

From these, **four** omega components are derived:

.. math::

   \omega_\text{qg\_moist} &= \omega_\text{qg,total} - \omega_\text{dry} \\
   \omega_\text{em\_moist} &= \omega_\text{em} - \omega_\text{dry} \\
   \omega_\text{moist}     &= \omega_\text{total} - \omega_\text{dry}

This yields a **4-way vertical-velocity decomposition**:

.. math::

   \omega = \underbrace{\omega_\text{dry}}_{\text{A+B}}
          + \underbrace{\omega_\text{qg\_moist}}_{\text{all diabatic (LOG20)}}
          + \underbrace{(\omega_\text{moist} - \omega_\text{qg\_moist})}_{\text{non-QG residual}}

with the Emanuel-moist component
:math:`\omega_\text{em\_moist}` providing a sub-partition that isolates
the LHR-only contribution.  Comparing :math:`\omega_\text{qg\_moist}`
and :math:`\omega_\text{em\_moist}` separates LHR from non-LHR diabatic
effects.

**References:** Hoskins B J, Draghici I, Davies H C (1978) *Q.J.R.M.S.*
104, 31–38.  Li L, O'Gorman P A (2020) *J. Climate*.
Emanuel K A, Fantini M, Thorpe A J (1987) *J. Atmos. Sci.* 44, 1559–1573.
Tamarin T, Kaspi Y (2016) *J. Atmos. Sci.* 73, 1687–1707.
Stone H L (1968) *SIAM J. Numer. Anal.* 5, 530–558.
Steinfeld D, Pfahl S (2019) *Clim. Dyn.* 53, 6159–6180.

.. autosummary::
   :toctree: generated/

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

1. Computing spherical vorticity :math:`\zeta` and divergence
   :math:`\delta` from the input wind via
   :func:`~pvtend.helmholtz.compute_vorticity_divergence`:

   .. math::

      \zeta = \frac{1}{a\cos\varphi}\!\left[
        \frac{\partial v}{\partial\lambda}
        - \frac{\partial(u\cos\varphi)}{\partial\varphi}\right]
      = \frac{\partial v}{\partial x}
        - \frac{\partial u}{\partial y}
        + \frac{u\tan\varphi}{a}

   .. math::

      \delta = \frac{1}{a\cos\varphi}\!\left[
        \frac{\partial u}{\partial\lambda}
        + \frac{\partial(v\cos\varphi)}{\partial\varphi}\right]
      = \frac{\partial u}{\partial x}
        + \frac{\partial v}{\partial y}
        - \frac{v\tan\varphi}{a}

   The :math:`\pm\tan\varphi/a` metric terms arise from spherical
   curvature and are significant at blocking latitudes (30–70°N).

2. Removing the area-weighted (:math:`\cos\varphi`) mean from each
   (Fredholm solvability condition).
3. Solving Poisson equations :math:`\nabla^2\psi=\zeta` and
   :math:`\nabla^2\chi=\delta` using the **spherical FFT Poisson solver**
   :func:`~pvtend.helmholtz.solve_poisson_spherical_fft`.
4. Recovering :math:`(u_\text{rot},v_\text{rot})` and
   :math:`(u_\text{div},v_\text{div})` via the **spectral gradient**
   :func:`~pvtend.helmholtz.gradient`.
5. Computing the harmonic residual by subtraction.

Spherical Poisson solver
~~~~~~~~~~~~~~~~~~~~~~~~

The Poisson equations are solved using a **conservative spherical
Laplacian** (following MiniUFO/xinvert) that combines:

* FFT in longitude (exploiting 360° periodicity), and
* a tridiagonal solve in latitude with the full
  :math:`\cos\varphi` metric factors.

This is the sole Poisson backend in v1.0.  The ``method`` keyword on
:func:`helmholtz_decomposition` is accepted for API compatibility but
ignored — all solves use the spherical FFT method.

**References:** Lynch P (1989) *MWR* 117, 1492–1500.

Spectral gradient (wind recovery)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After solving the Poisson equations, the divergent and rotational winds
are recovered via :func:`~pvtend.helmholtz.gradient`.  The **zonal**
derivative uses a spectral (FFT) method:

.. math::

   \frac{\partial\chi}{\partial x}\bigg|_{\!j}
   = \frac{1}{\Delta x_j}\;\mathrm{IFFT}\!\left(
     \mathrm{i}\,\frac{2\pi m}{N}\,\widehat{\chi}_m\right)

where :math:`\widehat{\chi}_m` are the Fourier coefficients along a
latitude circle of :math:`N` points.  This ensures that the discrete
composition :math:`\nabla\cdot(\nabla\chi)` is consistent with the
compact spherical Laplacian used by the Poisson solver, eliminating the
:math:`\cos^2(m\pi/N)` attenuation that centred finite differences
would introduce.  The Nyquist mode is zeroed for even :math:`N`.

The **meridional** derivative retains centred finite differences (the
:math:`\varphi` direction is non-periodic).

.. autosummary::
   :toctree: generated/

   helmholtz_decomposition
   helmholtz_decomposition_3d

**Low-level Poisson helpers** (importable from ``pvtend.helmholtz``):

.. currentmodule:: pvtend.helmholtz

.. autosummary::
   :toctree: generated/

   solve_poisson_spherical_fft
   laplacian_spherical_fft
   compute_vorticity_divergence
   gradient

``laplacian_spherical_fft`` is the **forward operator** matching the
Poisson solver's conservative spherical Laplacian stencil.  It enables
machine-precision round-trip verification:
:math:`\mathcal{L}(\text{solve\_poisson}(f)) = f` (interior points).

.. currentmodule:: pvtend


Moist/Dry Omega
---------------

Partitions total vertical velocity into **four** components and
recovers the corresponding divergent winds:

.. math::

   \omega = \underbrace{\omega_\text{dry}}_{\text{QG (A+B)}}
          + \underbrace{\omega_\text{qg\_moist}}_{\text{QG (A+B+C}_\text{LOG20}\text{) − (A+B)}}
          + \underbrace{\omega_\text{residual}}_{\text{non-QG}}

with :math:`\omega_\text{moist} = \omega_\text{total} - \omega_\text{dry}`
encompassing both QG-moist and non-QG contributions.

An additional **Emanuel-moist** component
:math:`\omega_\text{em\_moist} = \omega_{A+B+C_\text{em}} - \omega_{A+B}`
isolates the LHR-only response (see :ref:`qg-omega` for details).

The divergent-wind recovery proceeds as follows:

1. :math:`\omega_\text{moist} = \omega_\text{total} - \omega_\text{dry}`
2. Solve :math:`\nabla^2\chi_\text{moist} = -\partial\omega_\text{moist}/\partial p`
   on each pressure level (spherical Poisson solver).
3. :math:`(u_\text{div,moist}, v_\text{div,moist}) = \nabla\chi_\text{moist}`
4. Solve :math:`\nabla^2\chi_\text{dry} = -\partial\omega_\text{dry}/\partial p`
   independently on each pressure level (same spherical Poisson solver).
5. :math:`(u_\text{div,dry}, v_\text{div,dry}) = \nabla\chi_\text{dry}`
6. QG-moist divergent wind via the same Poisson inversion applied to
   :math:`\omega_\text{qg\_moist}`:
   :math:`(u_\text{div,qg\_moist}, v_\text{div,qg\_moist}) = \nabla\chi_\text{qg\_moist}`
7. Emanuel-moist divergent wind via the same Poisson inversion applied to
   :math:`\omega_\text{em\_moist}`:
   :math:`(u_\text{div,em\_moist}, v_\text{div,em\_moist}) = \nabla\chi_\text{em\_moist}`

All four divergent-wind components are recovered via **independent
Poisson inversions** using :func:`~pvtend.moist_dry.solve_chi_from_omega`.
This avoids amplification of discretisation errors from the Helmholtz
decomposition that would occur with a residual-based approach
(:math:`\mathbf{u}_{\text{div,dry}} = \mathbf{u}_{\text{div}} - \mathbf{u}_{\text{div,moist}}`).

Which solver is used where
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Equation
     - Solver
     - Function
   * - :math:`\nabla^2\psi = \zeta` (Helmholtz)
     - Spherical FFT Poisson
     - :func:`~pvtend.helmholtz.solve_poisson_spherical_fft`
   * - :math:`\nabla^2\chi = \delta` (Helmholtz)
     - Spherical FFT Poisson
     - :func:`~pvtend.helmholtz.solve_poisson_spherical_fft`
   * - :math:`\nabla^2\chi_\text{moist} = -\partial\omega_\text{moist}/\partial p`
     - Spherical FFT Poisson
     - :func:`~pvtend.moist_dry.solve_chi_from_omega`
   * - :math:`\nabla^2\chi_\text{dry} = -\partial\omega_\text{dry}/\partial p`
     - Spherical FFT Poisson
     - :func:`~pvtend.moist_dry.solve_chi_from_omega`
   * - :math:`\nabla^2\chi_\text{qg\_moist} = -\partial\omega_\text{qg\_moist}/\partial p`
     - Spherical FFT Poisson
     - :func:`~pvtend.moist_dry.solve_chi_from_omega`
   * - :math:`\nabla^2\chi_\text{em\_moist} = -\partial\omega_\text{em\_moist}/\partial p`
     - Spherical FFT Poisson
     - :func:`~pvtend.moist_dry.solve_chi_from_omega`
   * - :math:`\mathcal{L}(\chi) = f` (forward Laplacian for verification)
     - Conservative spherical stencil
     - :func:`~pvtend.helmholtz.laplacian_spherical_fft`

Note: the **horizontal wind** decomposition remains **2-way**
(dry + moist).  The QG-moist and Emanuel-moist divergent winds are
physically interpretable *sub-partitions* of the full moist divergent
wind — they can substitute for :math:`\mathbf{u}_{\chi,\text{moist}}`
in PV cross-term analyses but are **not** additional additive
components of the Helmholtz decomposition.  Comparing their PV
cross-terms isolates LHR from non-LHR diabatic effects.

**Total-field approximation for horizontal divergence.**
The QG omega solve and Poisson inversion are performed on **total
fields** (:math:`\omega`, not :math:`\omega'`), exploiting the
dominance of the anomaly vertical velocity in midlatitude synoptic
systems (:math:`|\omega'| \gg |\bar\omega|`).  Because the Poisson
operator is linear, the same approximation propagates to the
horizontal divergent wind:

.. math::

   \omega_\text{moist} \approx \omega'_\text{moist}
   \;\;\Longrightarrow\;\;
   \mathbf{u}_{\chi,\text{moist}} \approx \mathbf{u}'_{\chi,\text{moist}}

This avoids a costly second anomaly-field QG inversion while capturing
the dominant moist--dry partition of both vertical velocity and
horizontal divergent wind.

.. autosummary::
   :toctree: generated/

   decompose_omega
   solve_chi_from_omega


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

1. Load ERA5 data for the time window (including specific humidity *q*).
2. Subtract hourly climatology → anomalies; compute :math:`\partial T/\partial t`.
3. Compute all spatial / temporal derivatives, including Emanuel (1987)
   LHR :math:`\dot\theta_\text{LHR}` and :math:`Q_\text{LHR}`.
4. Helmholtz decomposition on the full NH hemisphere (spherical Poisson).
5. QG omega (SIP, terms A+B) → :math:`\omega_\text{dry}`.
6. QG omega (SIP, terms A+B+\ :math:`C_\text{LOG20}` with
   :math:`\partial T/\partial t`) →
   :math:`\omega_\text{qg\_moist} = \omega_{A+B+C} - \omega_{A+B}`.
7. QG omega (SIP, terms A+B+\ :math:`C_\text{em}` with Emanuel LHR) →
   :math:`\omega_\text{em\_moist} = \omega_{A+B+C_\text{em}} - \omega_{A+B}`.
8. Moist / dry decomposition →
   :math:`\omega_\text{moist}`, :math:`\omega_\text{qg\_moist}`,
   :math:`\omega_\text{em\_moist}`,
   and their divergent winds via spherical Poisson inversion.
9. Extract event-centred patches.
10. Compute PV cross-terms (dry, moist, QG-moist, Emanuel-moist) and
    vertical weighted averages.
11. Write per-timestep NPZ files.

:class:`TendencyComputer` is parameterised by event type
(blocking / PRP), eliminating code duplication between scripts.

Helper functions
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   TendencyConfig
   TendencyComputer

.. currentmodule:: pvtend.tendency

.. autosummary::
   :toctree: generated/

   load_climatology
   month_keys_for_window
   open_months_ds
   with_derivs_for_window
   get_tracked_center

.. currentmodule:: pvtend


.. _classify:

RWB Classification (Pass 1)
----------------------------

Reads the ``dh=0`` NPZ snapshots produced by :class:`TendencyComputer`,
classifies each event as **AWB** (Anticyclonic Wave Breaking),
**CWB** (Cyclonic Wave Breaking), or **NEUTRAL** at multiple pressure
levels, and emits a "variant tracksets" PKL.

The classification uses :func:`sampled_longest_contours` on Z-field
contours followed by :func:`overturn_x_intervals` and
:func:`classify_bay`, with a tilt-based fallback.  A multi-level
threshold is applied: the event must be classified consistently across
at least ``classify_threshold`` pressure levels to qualify.

The CLI subcommand is ``pvtend-pipeline classify``.

.. autosummary::
   :toctree: generated/

   ClassifyConfig
   ClassifyResult

.. currentmodule:: pvtend.classify

.. autosummary::
   :toctree: generated/

   run_pass1

.. currentmodule:: pvtend


.. _composite-builder:

Variant-aware Composite Builder (Pass 2)
-----------------------------------------

Accumulates per-event NPZ fields into running sums grouped by event
stage (onset / peak / decay), relative hour offset, and **RWB variant**.

Ten variants are supported:

* ``original`` — all events (no RWB filter).
* ``AWB_onset``, ``AWB_peak``, ``AWB_decay``
* ``CWB_onset``, ``CWB_peak``, ``CWB_decay``
* ``NEUTRAL_onset``, ``NEUTRAL_peak``, ``NEUTRAL_decay``

:class:`CompositeResult` provides :meth:`~CompositeResult.mean_3d` and
:meth:`~CompositeResult.reduce_2d` methods for rapid access to
composite-mean fields, including exponential height-weighted vertical
averages (``level_mode="wavg"``).

The CLI subcommand is ``pvtend-pipeline composite``.

.. autosummary::
   :toctree: generated/

   CompositeConfig
   CompositeResult

.. currentmodule:: pvtend.composite_builder

.. autosummary::
   :toctree: generated/

   build_composites

.. currentmodule:: pvtend


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
   * - ``use_sig_mask``
     - If ``True`` (default), zero out non-significant grid points before
       projection; set ``False`` to project the full composite mean.
   * - ``mask_negative``
     - If ``True`` (default), mask negative PV lobes in basis construction;
       set ``False`` to retain them.

.. note::

   All bootstrap and composite-mean routines use ``np.nanmean`` /
   ``np.nanpercentile`` internally, so events with partial NaN coverage
   (e.g. high-latitude edge patches) are handled gracefully without
   corrupting the composite or flipping projection signs.

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


Sample Data
-----------

Bundled idealized PV evolution for quickstart examples and unit tests.

.. autosummary::
   :toctree: generated/

   data.load_idealized_pv
