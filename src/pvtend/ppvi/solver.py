"""Chain the Wu PPVI Fortran cores: pvpialln → qinvert21 → qinvertp21.

The full piecewise potential-vorticity inversion proceeds as:

1. **pvpialln** on the *mean* state (H, T, U, V) → mean PV ``Q_mean``,
   balanced streamfunction ``psi_mean``, boundary potential temperatures.
2. **pvpialln** on the *event* state → ``Q_event``, ``psi_event``, θ.
3. **qinvert21** (total balanced inversion) on the event fields, using the
   pvpialln output as the first guess → balanced total ``H_bal``, ``SI_bal``.
4. **qinvertp21** (perturbation inversion) run once per vertical *piece*.
   Pieces are now **per-level** (one single Wu level each, 1000→100 hPa) →
   piecewise ``HP``, ``SP``.

All cores are the exact validated v0.8.0 serial Gauss–Seidel solvers,
transcribed verbatim and compiled I/O-free.  The numerics are byte-exact
(to the F10.2 / F8.1 output-rounding floor) against the standalone
``*.exe`` pipeline.

Grid / unit conventions
-----------------------
- Cubes are ``(NL, NY, NX)`` with ``NL=9`` Wu levels
  [1000, 850, 700, 500, 400, 300, 250, 200, 100] hPa, latitude N→S,
  longitude on a monotonic event-relative frame.
- ``H`` is geopotential height z = Φ/g [m]; ``T`` is **raw temperature** [K]
  (pvpialln converts T→θ internally — do NOT pre-convert).
- ``psi`` returned by pvpialln is in m²/s; the cores exchange ψ in the
  Wu file convention (ψ/1e5) internally.
- ``zhdr`` is the 8-element Wu header
  ``[lat_s, lon_w, lat_n, lon_e, dlat, dlon, nx, ny]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._ext import load_ext

NY: int = 51
NX: int = 134
NL: int = 9
MI: float = 9999.90  # Wu missing-value sentinel

#: Wu pressure levels as σ = p/p0 (1000 hPa reference); 9 levels 1000→100 hPa.
PR = np.array([1.0, 0.85, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1], dtype=np.float32)

#: Vertical pieces (1-based Wu level indices) → name. Per-level decomposition:
#: one single-level piece per Wu level, keyed by the 1-based index as a string.
#: Index 1 is the 1000 hPa bottom-θ (Bretherton) piece, index NL the 100 hPa
#: top-θ piece, and 2..NL−1 the interior-PV pieces. ``tendency.py`` maps each
#: index ``str(i)`` to its hPa level for the ``*_ppvi_{L}`` npz keys.
PIECES: dict[str, list[int]] = {str(i): [i] for i in range(1, NL + 1)}


@dataclass(frozen=True)
class PassABParams:
    """pvpialln (pass A/B) SOR parameters (validated defaults)."""

    imax: int = 5000
    omegs: float = 1.75
    thrs: float = 5.0e4


@dataclass(frozen=True)
class PassCParams:
    """qinvert21 (pass C, total inversion) SOR parameters."""

    max_iter: int = 10000
    max_outer: int = 10000
    omegs: float = 1.4
    omegah: float = 1.4
    part: float = 0.05
    thrsh: float = 0.01
    qmin: float = 0.01


@dataclass(frozen=True)
class PassDParams:
    """qinvertp21 (pass D, perturbation inversion) parameters.

    ``ibc`` selects the lateral boundary condition of every piece solve:
    0 homogeneous Dirichlet, 1 full perturbation on the walls, 2 walls and
    first guess from per-piece fields (nesting) — with ``ibc=2`` the caller
    must hand :func:`invert_piecewise` a ``bc_fields`` entry (or the
    ``bc_residual`` role) for every piece.
    """

    omegs: float = 1.85
    omegah: float = 1.4
    part: float = 0.05
    thrsh: float = 0.1
    tscal: float = 1.0
    qscal: float = 1.0
    inlin: int = 1
    ibc: int = 0


def _to_core(cube: np.ndarray) -> np.ndarray:
    """``(NL, NY, NX)`` → ``(NY, NX, NL)`` F-contiguous float32.

    ``NY`` / ``NX`` are inferred from the cube; only ``NL`` is fixed (the
    Fortran cores are compiled with ``NL=9`` and infer ``NY``/``NX`` from
    the array shapes).
    """
    arr = np.asarray(cube)
    if arr.ndim != 3 or arr.shape[0] != NL:
        raise ValueError(
            f"expected cube shape (NL={NL}, NY, NX), got {arr.shape}"
        )
    return np.asfortranarray(np.transpose(arr, (1, 2, 0)), dtype=np.float32)


def _from_core(arr: np.ndarray) -> np.ndarray:
    """``(NY, NX, NL)`` → ``(NL, NY, NX)`` C-contiguous float64."""
    return np.ascontiguousarray(np.transpose(np.asarray(arr), (2, 0, 1)),
                                dtype=np.float64)


def fill_below_ground(
    H: np.ndarray, T: np.ndarray, U: np.ndarray, V: np.ndarray,
    pr: np.ndarray = PR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill below-ground NaN so the Wu SOR cores (not NaN-aware) converge.

    Pressure-level fields derived from terrain-following model levels (e.g.
    CESM f09) are genuinely missing where the level lies below ground (1000/
    850 hPa under high terrain or cold high-latitude highs). The Fortran SOR
    solvers propagate any NaN across the whole 3-D elliptic solve, so the input
    must be gap-filled first. T, U, V get constant downward extrapolation from
    the lowest valid level; H (geopotential height) is filled hydrostatically
    ``H_k = H_{k+1} + (R_d·T̄/g)·ln(p_{k+1}/p_k)`` (with ``p_k > p_{k+1}`` the
    height decreases downward).

    Cubes are ``(NL, NY, NX)`` with level 0 the highest pressure (1000 hPa).
    Returns the inputs **unchanged** when there are no NaNs (e.g. ERA5
    pressure-level data, which is filled below ground at the source) — so this
    is a strict no-op on the validated ERA5 path.
    """
    if all(np.isfinite(x).all() for x in (H, T, U, V)):
        return H, T, U, V
    H, T, U, V = (np.array(x, dtype=np.float64, copy=True) for x in (H, T, U, V))
    Rd, g = 287.05, 9.80665
    p_hpa = np.asarray(pr, dtype=np.float64) * 1000.0   # σ → hPa
    nl = H.shape[0]
    for k in range(nl - 2, -1, -1):       # fill lower (higher-p) levels from above
        for A in (T, U, V):
            m = ~np.isfinite(A[k])
            A[k][m] = A[k + 1][m]
        mz = ~np.isfinite(H[k])
        dz = (Rd * 0.5 * (T[k] + T[k + 1]) / g) * np.log(p_hpa[k + 1] / p_hpa[k])
        H[k][mz] = (H[k + 1] + dz)[mz]
    return H, T, U, V


def fill_below_ground_stack(
    fields: dict[str, np.ndarray],
    plev_hpa: np.ndarray,
    height_key: str = "z",
    temp_key: str = "t",
) -> dict[str, np.ndarray]:
    """:func:`fill_below_ground` over an arbitrary set of fields.

    Same convention, same sweep, same formula — persistence downward from the
    lowest valid level for everything except the height, which is continued
    hydrostatically. This exists because the tendency chain needs ``q``, ``w``
    and ``pv`` filled as well, and running the inversion and the tendency under
    *different* below-ground conventions would plant an artefact in exactly the
    difference between them.

    ERA5 needs none of this — ECMWF extrapolates below ground at the source, so
    its pressure-level fields are already gap-free and this is a no-op on them.
    CESM's are not: at 1000 hPa ~52 % of the NH is below ground, 4.6 % at
    850 hPa, 1.1 % at 700 hPa. Left alone, a single NaN destroys a whole
    spherical-harmonic transform and quietly poisons the QG-omega solve.

    Args:
        fields: ``{name: (nlev, ny, nx)}``; must contain *height_key* and
            *temp_key*. Modified copies are returned; the inputs are untouched.
        plev_hpa: ``(nlev,)`` pressure levels [hPa], **descending in altitude**
            — index 0 is the highest pressure, as in ``PR``.
        height_key: Field continued hydrostatically rather than by persistence.
        temp_key: Temperature [K], needed for the hypsometric thickness.

    Returns:
        A new dict with the same keys, gap-filled.
    """
    if all(np.isfinite(v).all() for v in fields.values()):
        return dict(fields)
    for k in (height_key, temp_key):
        if k not in fields:
            raise KeyError(f"fill_below_ground_stack needs {k!r}; got "
                           f"{sorted(fields)}")

    out = {k: np.array(v, dtype=np.float64, copy=True) for k, v in fields.items()}
    Rd, g = 287.05, 9.80665
    p = np.asarray(plev_hpa, dtype=np.float64)
    if p[0] < p[-1]:
        raise ValueError(f"plev_hpa must descend in altitude (index 0 = highest "
                         f"pressure); got {p[0]} .. {p[-1]}")
    H, T = out[height_key], out[temp_key]
    persist = [v for k, v in out.items() if k != height_key]

    for k in range(len(p) - 2, -1, -1):      # downward: fill k from k+1
        for A in persist:
            m = ~np.isfinite(A[k])
            A[k][m] = A[k + 1][m]
        mz = ~np.isfinite(H[k])
        # T[k] has just been filled, so 0.5*(T[k]+T[k+1]) is the persisted
        # value — an isothermal layer, matching `fill_below_ground`.
        dz = (Rd * 0.5 * (T[k] + T[k + 1]) / g) * np.log(p[k + 1] / p[k])
        H[k][mz] = (H[k + 1] + dz)[mz]
    return out


def _run_pvpialln(ext, H, T, U, V, zhdr10, p):
    """Run one pvpialln pass; return psi, q, thb, tht (all (NY,NX,·))."""
    psi, q, thb, tht = ext.pvpialln_core(
        _to_core(H), _to_core(T), _to_core(U), _to_core(V),
        zhdr10, PR,
        int(p.imax), np.float32(p.omegs), np.float32(p.thrs),
    )
    return psi, q, thb, tht


def invert_piecewise(
    H_mean: np.ndarray,
    T_mean: np.ndarray,
    U_mean: np.ndarray,
    V_mean: np.ndarray,
    H_event: np.ndarray,
    T_event: np.ndarray,
    U_event: np.ndarray,
    V_event: np.ndarray,
    zhdr: np.ndarray,
    pieces: dict[str, list[int]] | None = None,
    qp_anoms: dict[str, np.ndarray] | None = None,
    th_anoms: dict[str, np.ndarray] | None = None,
    fill_nan: bool = True,
    pab: PassABParams | None = None,
    pc: PassCParams | None = None,
    pd: PassDParams | None = None,
    bc_fields: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    bc_residual: str | None = None,
) -> dict:
    """Run the full Wu piecewise PV inversion on a mean + event state.

    Args:
        H_mean, T_mean, U_mean, V_mean: Mean-state cubes ``(NL, NY, NX)``.
            ``H`` = geopotential height [m]; ``T`` = raw temperature [K].
        H_event, T_event, U_event, V_event: Event-state cubes ``(NL, NY, NX)``.
        zhdr: Wu 8-element header
            ``[lat_s, lon_w, lat_n, lon_e, dlat, dlon, nx, ny]``. Works on any
            regular grid including **anisotropic** ones (Δlat≠Δlon, e.g. CESM
            f09): the two spacings are reordered internally to match the
            Fortran's ``HDR(5)=Δlon, HDR(6)=Δlat`` convention.
        pieces: Mapping name → 1-based Wu level list. Defaults to
            :data:`PIECES`.
        qp_anoms: Optional mapping name → ``(NL, NY, NX)`` PV **anomaly** to use
            for that piece instead of ``Q_event - Q_mean``::

                QPIN_piece = QBIN + qp_anoms[name]

            The base state is untouched, so this partitions the anomaly
            horizontally (or spectrally) instead of by level.  Pass D is linear
            in the PV source, so arrays that sum to the full anomaly give pieces
            that sum exactly to the unmasked result -- which is what makes a
            scale split (planetary vs eddy) a legitimate piecewise inversion
            rather than an approximation.

            An additive override rather than a multiplicative mask on purpose:
            a spectral split needs ``q'_p = filter(q')`` inside an object mask,
            and recovering that as a multiplicative weight would need
            ``filter(q')/q'``, which is unbounded wherever ``q'`` is small.

            Names not present here fall back to the unmasked ``QPIN``.
        th_anoms: Optional mapping name → ``(NY, NX, 2)`` boundary-θ **anomaly**
            (``[..., 0]`` bottom, ``[..., 1]`` top) to use for that piece::

                THPIN_piece = THBIN + th_anoms[name]

            The counterpart of `qp_anoms` for the boundary pieces.  ``qp_anoms``
            cannot reach them: K=1 and K=NL are driven by ``THPIN``, not
            ``QPIN``, so masking the PV array leaves the boundary-θ source
            untouched.  Splitting a boundary piece by the same 3-D object mask
            (its slice at that boundary level) needs this.

            Names not present here fall back to the unmasked ``THPIN``.
        fill_nan: If True (default), hydrostatically gap-fill below-ground NaN
            in the input cubes via :func:`fill_below_ground` before inverting
            (required for terrain-following model data such as CESM f09; a
            no-op on already-filled ERA5 pressure-level data).
        pab, pc, pd: Solver parameter overrides for passes A/B, C, D.
        bc_fields: Per-piece lateral boundary condition + first guess for
            ``pd.ibc == 2`` (nesting): mapping name → ``(HP0, SP0)`` cubes
            ``(NL, NY, NX)`` in the pass-D OUTPUT units — ``HP0`` in metres
            (``H_pieces`` convention), ``SP0`` in ψ/1e5 (``SP_pieces``
            convention) — so an outer-domain piece solution, subset to this
            window, feeds back verbatim. The boundary ring supplies the fixed
            Dirichlet values; the interior is only the SOR first guess.
        bc_residual: With ``pd.ibc == 2``, the name of the one piece whose
            boundary fields are not given explicitly but computed as
            ``full perturbation − Σ bc_fields``: the wall/remainder piece.
            Guarantees the piece boundary values telescope exactly to the
            full perturbation, so the nested pieces + wall still sum to the
            all-sources inversion.

    Returns:
        Dict with keys:

        - ``psi_total`` ``(NL, NY, NX)`` — total balanced ψ′ [m²/s]
          (event − mean, from the qinvert21 total inversion ``SI_bal``).
        - ``psi_pieces`` — dict name → ``(NL, NY, NX)`` piecewise ψ′ [m²/s].
        - ``H_pieces`` — dict name → ``(NL, NY, NX)`` piecewise Φ′ [Wu units].
        - ``Q_mean``, ``Q_event`` ``(NL, NY, NX)`` — Wu PV (interior
          K=2..NL−1 valid; boundary levels set to :data:`MI`).
        - ``psi_mean``, ``psi_event`` ``(NL, NY, NX)`` — pvpialln ψ [m²/s].
        - ``SP_pieces`` / ``HP_pieces`` — raw qinvertp block outputs
          (Wu file units; ψ_phys = SP × 1e5).
        - ``SI_bal`` ``(NL, NY, NX)`` — qinvert21 total balanced ψ block
          (Wu file units; ψ_phys = SI_bal × 1e5).
    """
    pieces = pieces or PIECES
    pab = pab or PassABParams()
    pc = pc or PassCParams()
    pd = pd or PassDParams()
    ext = load_ext()

    if fill_nan:
        H_mean, T_mean, U_mean, V_mean = fill_below_ground(
            H_mean, T_mean, U_mean, V_mean)
        H_event, T_event, U_event, V_event = fill_below_ground(
            H_event, T_event, U_event, V_event)

    zhdr10 = np.zeros(10, dtype=np.float32)
    zhdr8 = np.asarray(zhdr, dtype=np.float32).ravel()[:8]
    zhdr10[:8] = zhdr8
    # The Fortran cores reconstruct each row's latitude as ``HDR(3)-(I-1)*HDR(6)``
    # and use ``HDR(5)`` for the zonal grid distance — i.e. they require
    # ``HDR(5)=Δlon`` and ``HDR(6)=Δlat``. The public ``zhdr`` is documented as
    # ``[..,dlat,dlon,..]`` (index 4=dlat, 5=dlon), so swap the two spacings here.
    # No-op on isotropic grids (ERA5 1.5°, Δlat=Δlon ⇒ byte-identical); fixes
    # anisotropic grids (CESM f09, Δlat=0.942≠Δlon=1.25) which would otherwise
    # reconstruct latitudes running off into the SH and blow the solver up.
    zhdr10[4], zhdr10[5] = zhdr8[5], zhdr8[4]

    # Guard the Fortran's latitude contract: it reconstructs row I's latitude as
    # ``lat_n - (I-1)·Δlat`` (HDR(3), HDR(6)), so the header band edges and Δlat
    # must be mutually consistent — ``lat_n - lat_s ≈ (ny-1)·Δlat``. A caller
    # passing nominal band edges (e.g. 10.5/85.5) with off-grid actual edges and
    # an anisotropic Δlat would silently invert at the wrong latitudes. Cubes
    # must also be ordered N→S (row 0 = northernmost), per the module docstring.
    lat_s, lat_n, dlat_in, ny_in = (float(zhdr8[0]), float(zhdr8[2]),
                                    float(zhdr8[4]), int(round(float(zhdr8[7]))))
    span_err = abs((lat_n - lat_s) - (ny_in - 1) * dlat_in)
    if span_err > 0.5 * dlat_in:
        raise ValueError(
            f"zhdr band edges inconsistent with Δlat·(ny-1): "
            f"lat_n−lat_s={lat_n - lat_s:.3f}° but (ny−1)·Δlat="
            f"{(ny_in - 1) * dlat_in:.3f}° (Δlat={dlat_in}, ny={ny_in}). "
            f"Set lat_s/lat_n to the ACTUAL grid band-edge latitudes (the "
            f"Fortran reconstructs rows as lat_n−(I−1)·Δlat)."
        )

    # ── Pass A: mean state ───────────────────────────────────────────
    psi_m, q_m, thb_m, tht_m = _run_pvpialln(
        ext, H_mean, T_mean, U_mean, V_mean, zhdr10, pab)
    # ── Pass B: event state ──────────────────────────────────────────
    psi_e, q_e, thb_e, tht_e = _run_pvpialln(
        ext, H_event, T_event, U_event, V_event, zhdr10, pab)

    # ── Pass C: total balanced inversion (event) ─────────────────────
    HZIN = _to_core(H_event)                       # first guess H [m]
    SIIN = np.asfortranarray(np.asarray(psi_e) / 1.0e5, dtype=np.float32)
    THTIN = np.asfortranarray(
        np.stack([np.asarray(thb_e), np.asarray(tht_e)], axis=2),
        dtype=np.float32)
    H_bal, SI_bal = ext.qinvert_core(
        HZIN, SIIN, np.asfortranarray(q_e, dtype=np.float32), THTIN,
        zhdr10, PR,
        int(pc.max_iter), int(pc.max_outer),
        np.float32(pc.omegs), np.float32(pc.omegah),
        np.float32(pc.part), np.float32(pc.thrsh), np.float32(pc.qmin),
    )

    # ── Pass D: perturbation inversion, one call per piece ───────────
    MBIN = _to_core(H_mean)                         # mean H [m]
    SBIN = np.asfortranarray(np.asarray(psi_m) / 1.0e5, dtype=np.float32)
    QBIN = np.asfortranarray(q_m, dtype=np.float32)
    QPIN = np.asfortranarray(q_e, dtype=np.float32)
    THBIN = np.asfortranarray(
        np.stack([np.asarray(thb_m), np.asarray(tht_m)], axis=2),
        dtype=np.float32)
    THPIN = THTIN                                   # reuse event θ stack
    # Pass D needs the SAME Fortran convention as passes A/B/C: qinvertp's BALP
    # reads ZHDR(5) as dlon and ZHDR(6) as dlat (SIGM=ZHDR(5)/ZHDR(6),
    # LL=AA*ZHDR(5), and the row latitudes as ZHDR(3)-(I-1)*ZHDR(6)).  Handing it
    # the unswapped public header is a no-op on an isotropic grid but wrong on an
    # anisotropic one: on CESM f09 (0.9424 x 1.25) pass D laid its 80 rows out at
    # 1.25 deg spacing, spanning 98.75 deg from 85.29 N down past the equator to
    # -13.5 N, so the Coriolis parameter changed sign inside the box.  Measured
    # effect: RMS(sum of pieces)/RMS(observed u'_rot) = 1.43-1.66 on CESM against
    # 0.99 on ERA5 and JRA-3Q, which are isotropic and therefore unaffected.
    zhdr_d = zhdr10[:8]                             # pass D uses 8-elt header

    psi_pieces: dict[str, np.ndarray] = {}
    H_pieces: dict[str, np.ndarray] = {}
    SP_pieces: dict[str, np.ndarray] = {}
    HP_pieces: dict[str, np.ndarray] = {}
    qp_anoms = qp_anoms or {}
    th_anoms = th_anoms or {}

    # ── ibc=2 nesting: per-piece boundary/guess fields ───────────────
    zeros_bc = np.zeros_like(np.asarray(MBIN))
    bc_core: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if int(pd.ibc) == 2:
        bc_fields = bc_fields or {}
        missing = [n for n in pieces
                   if n not in bc_fields and n != bc_residual]
        if missing:
            raise ValueError(
                f"pd.ibc=2 needs bc_fields for every piece (or the "
                f"bc_residual role); missing {missing}")
        for n, (hp0, sp0) in bc_fields.items():
            bc_core[n] = (_to_core(hp0), _to_core(sp0))
        if bc_residual is not None:
            # Whatever boundary flow the explicit pieces do not carry is
            # this piece's, so the walls telescope to the full perturbation
            # exactly: HP_full = H_bal − MBIN [m], SP_full = SI_bal − SBIN
            # [ψ/1e5] (the same subtraction qinvertp does internally).
            hp_res = np.asarray(H_bal) - np.asarray(MBIN)
            sp_res = np.asarray(SI_bal) - np.asarray(SBIN)
            for n, (hp0, sp0) in bc_core.items():
                hp_res = hp_res - hp0
                sp_res = sp_res - sp0
            bc_core[bc_residual] = (
                np.asfortranarray(hp_res, dtype=np.float32),
                np.asfortranarray(sp_res, dtype=np.float32))
    elif bc_fields or bc_residual:
        raise ValueError("bc_fields/bc_residual are only meaningful with "
                         "PassDParams(ibc=2)")
    for name, levels in pieces.items():
        nmlv = len(levels)
        qlv = np.zeros(NL, dtype=np.int32)
        qlv[:nmlv] = levels
        if name in qp_anoms:
            a = np.asarray(qp_anoms[name], dtype=np.float64)
            # accept the (NL, NY, NX) layout every other array in this API uses;
            # QPIN itself is the Fortran (NY, NX, NL) order
            if a.shape == QPIN.shape[::-1] or a.shape == (QPIN.shape[2],) + QPIN.shape[:2]:
                a = _to_core(a)
            elif a.shape != QPIN.shape:
                raise ValueError(
                    f"qp_anoms['{name}'] has shape {a.shape}; expected "
                    f"{(QPIN.shape[2],) + QPIN.shape[:2]} (NL,NY,NX) or {QPIN.shape} (NY,NX,NL)")
            QP_use = np.asfortranarray(QBIN + np.asarray(a, dtype=np.float32),
                                       dtype=np.float32)
        else:
            QP_use = QPIN
        if name in th_anoms:
            t = np.asarray(th_anoms[name], dtype=np.float32)
            if t.shape != THPIN.shape:
                raise ValueError(
                    f"th_anoms['{name}'] has shape {t.shape}, expected {THPIN.shape}")
            TH_use = np.asfortranarray(THBIN + t, dtype=np.float32)
        else:
            TH_use = THPIN
        hp0, sp0 = bc_core.get(name, (zeros_bc, zeros_bc))
        HPOUT, SPOUT = ext.qinvertp_core(
            MBIN, SBIN, H_bal, SI_bal, QBIN, QP_use, THBIN, TH_use,
            zhdr_d, PR, qlv, int(nmlv),
            np.float32(pd.omegs), np.float32(pd.omegah), np.float32(pd.part),
            np.float32(pd.thrsh), np.float32(pd.tscal), np.float32(pd.qscal),
            int(pd.inlin), int(pd.ibc), hp0, sp0,
        )
        SP = _from_core(SPOUT)
        HP = _from_core(HPOUT)
        SP_pieces[name] = SP
        HP_pieces[name] = HP
        psi_pieces[name] = SP * 1.0e5
        H_pieces[name] = HP

    return dict(
        psi_total=_from_core(SI_bal) * 1.0e5,
        psi_pieces=psi_pieces,
        H_pieces=H_pieces,
        Q_mean=_from_core(q_m),
        Q_event=_from_core(q_e),
        psi_mean=_from_core(psi_m),
        psi_event=_from_core(psi_e),
        SP_pieces=SP_pieces,
        HP_pieces=HP_pieces,
        SI_bal=_from_core(SI_bal),
        H_bal=_from_core(H_bal),
    )
