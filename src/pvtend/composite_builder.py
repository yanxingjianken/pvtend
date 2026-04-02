"""Composite accumulation from per-event NPZ files.

Corresponds to **Pass 2** of the core script
``ss01_rwb_stage_multilevel_composites.py``.

Usage (via CLI)::

    pvtend-pipeline composite \\
        --npz-dir /path/to/composite_blocking_tempest \\
        --rwb-pkl /path/to/outputs/rwb_variant_tracksets.pkl \\
        --output  /path/to/outputs/composite.pkl

Or programmatically::

    from pvtend.composite_builder import build_composites, CompositeConfig
    from pvtend.classify import ClassifyResult

    rwb = ClassifyResult.load("rwb_variant_tracksets.pkl")
    cfg = CompositeConfig(npz_dir=Path("..."), stages=["onset","peak","decay"])
    comp = build_composites(cfg, rwb)
    comp.save("composite.pkl")
"""

from __future__ import annotations

import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .classify import ClassifyResult, _parse_track_id, _parse_dh, _load_excluded


# ── Picklable defaultdict factories (lambdas are NOT picklable) ──────
def _dd_dict() -> defaultdict:
    return defaultdict(dict)


def _dd_int() -> defaultdict:
    return defaultdict(int)


# ── Metadata keys in NPZ — skip when accumulating ────────────────────
_META = frozenset({
    "Y_rel", "X_rel", "levels", "wavg_levels", "H_SCALE", "G0",
    "lat_vec", "lon_vec_unwrapped",
    "track_id", "lat0", "lon0", "center_lat", "center_lon",
    "center_mode", "ts", "dh",
})


def _levels_indexer(
    levels_file: np.ndarray, levels_ref: np.ndarray,
) -> np.ndarray | None:
    """Map file levels → reference levels (index array or None)."""
    if (levels_file.shape == levels_ref.shape
            and np.all(levels_file == levels_ref)):
        return np.arange(levels_ref.size, dtype=int)
    pos = {int(lv): i for i, lv in enumerate(levels_file.tolist())}
    try:
        return np.array(
            [pos[int(lv)] for lv in levels_ref.tolist()], dtype=int
        )
    except (KeyError, ValueError):
        return None


def _accumulate(
    sums: dict[str, np.ndarray],
    valids: dict[str, np.ndarray],
    key: str,
    arr: np.ndarray,
) -> None:
    """NaN-safe in-place accumulation."""
    mask = np.isfinite(arr)
    a0 = np.where(mask, arr, 0.0)
    if key not in sums:
        sums[key] = a0.astype(np.float64, copy=True)
        valids[key] = mask.astype(np.uint16, copy=True)
    else:
        sums[key] += a0
        valids[key] += mask


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class CompositeConfig:
    """Configuration for Pass-2 composite accumulation.

    Attributes:
        npz_dir: Root directory with ``{stage}/dh=±N/*.npz``.
        stages: Event stages to process.
        exclude_file: Optional exclude-track CSV.
    """

    npz_dir: Path = Path(".")
    stages: list[str] = field(
        default_factory=lambda: ["onset", "peak", "decay"]
    )
    exclude_file: Path | None = None


# ── Result container ──────────────────────────────────────────────────

@dataclass
class CompositeResult:
    """Accumulated composite data, supporting *original* + RWB variants.

    Variants exposed:
        ``original`` — all events (no RWB filter);
        ``AWB_{stage}``, ``CWB_{stage}``, ``NEUTRAL_{stage}``
        for each stage.

    Access composites via :meth:`mean_3d` and :meth:`reduce_2d`.
    """

    levels: np.ndarray
    x_rel: np.ndarray
    y_rel: np.ndarray
    h_scale: float | None
    stages: list[str]
    fields_3d: list[str]

    # ``original`` accumulators — {evt: {dh: {field: arr}}}
    sums: dict[str, dict[int, dict[str, np.ndarray]]]
    valids: dict[str, dict[int, dict[str, np.ndarray]]]
    counts: dict[str, dict[int, int]]

    # RWB-variant accumulators — {variant: {evt: {dh: {field: arr}}}}
    sums_v: dict[str, dict[str, dict[int, dict[str, np.ndarray]]]]
    valids_v: dict[str, dict[str, dict[int, dict[str, np.ndarray]]]]
    counts_v: dict[str, dict[str, dict[int, int]]]

    variant_names: list[str]

    # ── access helpers ──

    def _pick(
        self, variant: str | None, stage: str, dh: int,
    ) -> tuple[dict, dict, int]:
        if not variant or str(variant).lower() == "original":
            s = self.sums.get(stage, {}).get(dh, {})
            v = self.valids.get(stage, {}).get(dh, {})
            c = self.counts.get(stage, {}).get(dh, 0)
        else:
            s = self.sums_v.get(variant, {}).get(stage, {}).get(dh, {})
            v = self.valids_v.get(variant, {}).get(stage, {}).get(dh, {})
            c = self.counts_v.get(variant, {}).get(stage, {}).get(dh, 0)
        return s, v, c

    def mean_3d(
        self,
        field: str,
        stage: str,
        dh: int,
        *,
        variant: str | None = "original",
    ) -> np.ndarray | None:
        """Return the NaN-safe mean 3-D composite array."""
        s, v, _ = self._pick(variant, stage, dh)
        arr_sum = s.get(field)
        vcount = v.get(field)
        if arr_sum is None or vcount is None:
            return None
        arr = np.asarray(arr_sum, dtype=np.float64)
        vc = np.asarray(vcount, dtype=np.float64)
        out = np.full_like(arr, np.nan)
        mask = vc > 0
        np.divide(arr, vc, out=out, where=mask)
        return out

    def reduce_2d(
        self,
        field: str,
        stage: str,
        dh: int,
        *,
        variant: str | None = "original",
        level_mode: str | int | None = None,
    ) -> np.ndarray | None:
        """Reduce a 3-D composite to 2-D.

        ``level_mode=None|"all"|"3d"`` → return full 3-D array.
        ``level_mode="wavg"`` → ``exp(−z/H)`` weighted average over 300, 250, 200 hPa.
        ``level_mode=300`` → nearest pressure level slice.
        """
        arr3d = self.mean_3d(field, stage, dh, variant=variant)
        if arr3d is None:
            return None
        if level_mode in (None, "", "all", "3d"):
            return arr3d
        if isinstance(level_mode, str) and level_mode.lower() in {
            "wavg", "w-avg", "weighted",
        }:
            # exp(−z/H) weighted average over 300, 250, 200 hPa
            # (matches tendency.py vwm — canonical pvtend recipe)
            from .constants import WAVG_LEVELS as _WL, H_SCALE as _HS, G0 as _G0
            wavg_hpa = np.asarray(_WL, dtype=float)
            levels_arr = np.asarray(self.levels, dtype=float)
            indices = [int(np.nanargmin(np.abs(levels_arr - lv)))
                       for lv in wavg_hpa]
            slices = arr3d[indices]  # (3, NY, NX)
            z_name = "z_3d" if "z_3d" in self.fields_3d else "z"
            z3d = self.mean_3d(z_name, stage, dh, variant=variant)
            if z3d is None:
                raise ValueError("Need z_3d for wavg")
            z_m = z3d[indices] / _G0  # geopotential → metres
            h = float(self.h_scale) if self.h_scale is not None else _HS
            wt = np.exp(-z_m / h)
            num = np.nansum(slices * wt, axis=0)
            den = np.nansum(wt, axis=0)
            out = np.full(num.shape, np.nan, dtype=np.float64)
            m = den > 0
            out[m] = num[m] / den[m]
            return out
        try:
            lev_val = float(level_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported level_mode {level_mode!r}") from exc
        levels = np.asarray(self.levels, dtype=float)
        idx = int(np.nanargmin(np.abs(levels - lev_val)))
        return arr3d[idx]

    def available_dh(
        self, stage: str, *, variant: str | None = "original",
    ) -> list[int]:
        if not variant or str(variant).lower() == "original":
            return sorted(self.counts.get(stage, {}).keys())
        return sorted(
            self.counts_v.get(variant, {}).get(stage, {}).keys()
        )

    # ── I/O ──

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[saved] composite → {path}", flush=True)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "CompositeResult":
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj


# ── Builder ───────────────────────────────────────────────────────────

def build_composites(
    cfg: CompositeConfig,
    rwb: ClassifyResult | None = None,
) -> CompositeResult:
    """Accumulate NPZ fields into variant-aware composites.

    Args:
        cfg: Composite configuration (directories, stages).
        rwb: Optional RWB classification result. If *None*, only the
            ``original`` variant (all events) is produced.

    Returns:
        :class:`CompositeResult` with accumulated sums/counts.
    """
    excluded = _load_excluded(cfg.exclude_file)
    variant_trackset = rwb.variant_trackset if rwb is not None else {}
    variants = list(variant_trackset.keys())

    # ── accumulators ──
    sums: dict[str, dict[int, dict]] = defaultdict(_dd_dict)
    valids: dict[str, dict[int, dict]] = defaultdict(_dd_dict)
    counts: dict[str, dict[int, int]] = defaultdict(_dd_int)

    sums_v: dict[str, dict[str, dict[int, dict]]] = {
        v: defaultdict(_dd_dict) for v in variants
    }
    valids_v: dict[str, dict[str, dict[int, dict]]] = {
        v: defaultdict(_dd_dict) for v in variants
    }
    counts_v: dict[str, dict[str, dict[int, int]]] = {
        v: defaultdict(_dd_int) for v in variants
    }

    LEVELS: np.ndarray | None = None
    X_REL = Y_REL = None
    H_SCALE: float | None = None
    fields_3d: set[str] = set()

    print("\n[pass2] Accumulating composites ...", flush=True)

    for evt in cfg.stages:
        evt_dir = cfg.npz_dir / evt
        if not evt_dir.exists():
            continue

        dh_dirs = []
        for d in sorted(evt_dir.iterdir()):
            if not d.is_dir():
                continue
            dh_val = _parse_dh(d.name)
            if dh_val is not None:
                dh_dirs.append((dh_val, d))
        dh_dirs.sort(key=lambda x: x[0])

        for dh, dh_dir in dh_dirs:
            npz_files = sorted(dh_dir.glob("*.npz"))
            if not npz_files:
                continue

            n_total = n_loaded = 0
            for fp in npz_files:
                n_total += 1
                tid = _parse_track_id(fp)
                if tid is not None and tid in excluded:
                    continue

                try:
                    with np.load(fp, allow_pickle=False) as Z:
                        levels_file = Z["levels"]
                        # probe 3D field
                        if "pv_3d" in Z.files:
                            probe = Z["pv_3d"]
                        elif "z_3d" in Z.files:
                            probe = Z["z_3d"]
                        else:
                            continue

                        if LEVELS is None:
                            LEVELS = levels_file.astype(int).copy()
                            X_REL = Z["X_rel"]
                            Y_REL = Z["Y_rel"]
                        if H_SCALE is None and "H_SCALE" in Z.files:
                            H_SCALE = float(Z["H_SCALE"])

                        idx = _levels_indexer(levels_file, LEVELS)
                        if idx is None:
                            continue
                        if probe[idx].ndim != 3:
                            continue

                        # Discover & accumulate 3D fields
                        for k in Z.files:
                            if k in _META:
                                continue
                            a = Z[k]
                            if a.ndim != 3:
                                continue
                            # Skip LS-derived fields
                            if any(k.startswith(p)
                                   for p in ("prp__", "int__", "ax__",
                                             "ay__", "beta__")):
                                continue
                            fields_3d.add(k)
                            a3 = a[idx]
                            _accumulate(sums[evt][dh], valids[evt][dh], k, a3)
                            for var in variants:
                                if tid in variant_trackset[var]:
                                    _accumulate(
                                        sums_v[var][evt][dh],
                                        valids_v[var][evt][dh],
                                        k, a3,
                                    )

                        counts[evt][dh] += 1
                        for var in variants:
                            if tid in variant_trackset[var]:
                                counts_v[var][evt][dh] += 1
                        n_loaded += 1
                except Exception:
                    continue

            print(
                f"[{evt}] dh={dh:+d}: total={n_total} loaded={n_loaded}",
                flush=True,
            )

    print(f"[pass2] 3D fields discovered: {sorted(fields_3d)}", flush=True)

    all_variants = ["original"] + variants
    return CompositeResult(
        levels=LEVELS if LEVELS is not None else np.array([], dtype=int),
        x_rel=X_REL if X_REL is not None else np.array([]),
        y_rel=Y_REL if Y_REL is not None else np.array([]),
        h_scale=H_SCALE,
        stages=list(cfg.stages),
        fields_3d=sorted(fields_3d),
        sums=dict(sums),
        valids=dict(valids),
        counts=dict(counts),
        sums_v=dict(sums_v),
        valids_v=dict(valids_v),
        counts_v=dict(counts_v),
        variant_names=all_variants,
    )
