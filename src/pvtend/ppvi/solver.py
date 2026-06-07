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
    """qinvertp21 (pass D, perturbation inversion) parameters."""

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
    pab: PassABParams | None = None,
    pc: PassCParams | None = None,
    pd: PassDParams | None = None,
) -> dict:
    """Run the full Wu piecewise PV inversion on a mean + event state.

    Args:
        H_mean, T_mean, U_mean, V_mean: Mean-state cubes ``(NL, NY, NX)``.
            ``H`` = geopotential height [m]; ``T`` = raw temperature [K].
        H_event, T_event, U_event, V_event: Event-state cubes ``(NL, NY, NX)``.
        zhdr: Wu 8-element header
            ``[lat_s, lon_w, lat_n, lon_e, dlat, dlon, nx, ny]``.
        pieces: Mapping name → 1-based Wu level list. Defaults to
            :data:`PIECES`.
        pab, pc, pd: Solver parameter overrides for passes A/B, C, D.

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

    zhdr10 = np.zeros(10, dtype=np.float32)
    zhdr8 = np.asarray(zhdr, dtype=np.float32).ravel()[:8]
    zhdr10[:8] = zhdr8

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
    zhdr_d = zhdr8                                  # pass D uses 8-elt header

    psi_pieces: dict[str, np.ndarray] = {}
    H_pieces: dict[str, np.ndarray] = {}
    SP_pieces: dict[str, np.ndarray] = {}
    HP_pieces: dict[str, np.ndarray] = {}
    for name, levels in pieces.items():
        nmlv = len(levels)
        qlv = np.zeros(NL, dtype=np.int32)
        qlv[:nmlv] = levels
        HPOUT, SPOUT = ext.qinvertp_core(
            MBIN, SBIN, H_bal, SI_bal, QBIN, QPIN, THBIN, THPIN,
            zhdr_d, PR, qlv, int(nmlv),
            np.float32(pd.omegs), np.float32(pd.omegah), np.float32(pd.part),
            np.float32(pd.thrsh), np.float32(pd.tscal), np.float32(pd.qscal),
            int(pd.inlin), int(pd.ibc),
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
