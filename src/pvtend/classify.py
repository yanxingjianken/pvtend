"""RWB (Rossby Wave Breaking) classification of tracked events.

Reads the dh=0 NPZ snapshots produced by :mod:`pvtend.tendency`,
classifies each event as AWB / CWB / NEUTRAL at multiple pressure
levels, and emits a "variant tracksets" PKL that the composite
builder can read.

This corresponds to **Pass 1** of the core script
``ss01_rwb_stage_multilevel_composites.py``.

Usage (via CLI)::

    pvtend-pipeline classify \\
        --npz-dir /path/to/composite_blocking_tempest \\
        --output  /path/to/outputs/rwb_variant_tracksets.pkl \\
        --stages  onset peak decay \\
        --levels  500 400 300 200 \\
        --threshold 3

Or programmatically::

    from pvtend.classify import run_pass1, ClassifyConfig
    cfg = ClassifyConfig(npz_dir=Path("..."))
    result = run_pass1(cfg)
    result.save("rwb_variant_tracksets.pkl")
"""

from __future__ import annotations

import csv
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .rwb import (
    RWBConfig,
    sampled_longest_contours,
    overturn_x_intervals,
    envelope_polygon,
    poly_area_centroid,
    classify_bay,
    centerline_tilt,
)

# ── regex helpers ────────────────────────────────────────────────────
_TRACK_RE = re.compile(r"track_(\d+)_")
_DH_RE = re.compile(r"^dh=([+\-]?\d+)$")


def _parse_track_id(fp: Path) -> int | None:
    m = _TRACK_RE.search(fp.name)
    return int(m.group(1)) if m else None


def _parse_dh(dirname: str) -> int | None:
    m = _DH_RE.match(dirname)
    return int(m.group(1)) if m else None


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class ClassifyConfig:
    """Configuration for Pass-1 RWB classification.

    Attributes:
        npz_dir: Root directory containing stage sub-directories
            (``onset/``, ``peak/``, ``decay/``), each with ``dh=±N``
            subdirectories that hold per-event NPZ files.
        output_path: Where to save the resulting variant-tracksets PKL.
        stages: List of event stages to process.
        classify_levels: Pressure levels [hPa] checked for RWB.
        classify_threshold: Number of levels that must agree.
        rwb_cfg: Fine-grained RWB bay-detection settings.
        exclude_file: Optional CSV listing track IDs to skip.
    """

    npz_dir: Path = Path(".")
    output_path: Path = Path("rwb_variant_tracksets.pkl")
    stages: list[str] = field(
        default_factory=lambda: ["onset", "peak", "decay"]
    )
    classify_levels: list[int | str] = field(
        default_factory=lambda: [500, 400, 300, 200]
    )
    classify_threshold: int = 3
    rwb_cfg: RWBConfig = field(
        default_factory=lambda: RWBConfig(area_min_deg2=20.0, try_levels=400)
    )
    exclude_file: Path | None = None


# ── Excluded track loader ─────────────────────────────────────────────

def _load_excluded(p: Path | None) -> set[int]:
    ids: set[int] = set()
    if p is None or not p.exists():
        return ids
    try:
        with open(p, "r", newline="") as f:
            sniff = f.read(1024)
            f.seek(0)
            if "," in sniff:
                reader = csv.DictReader(f)
                col = ("track_id" if reader.fieldnames
                       and "track_id" in reader.fieldnames else None)
                if col is not None:
                    for row in reader:
                        try:
                            ids.add(int(row[col]))
                        except (ValueError, KeyError):
                            pass
            else:
                for line in f:
                    m = re.search(r"\d+", line)
                    if m:
                        ids.add(int(m.group(0)))
    except Exception:
        pass
    return ids


# ── Single-level bay classifier ───────────────────────────────────────

def _classify_bays_z2d(
    z2d: np.ndarray,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    cfg: RWBConfig,
) -> tuple[bool, bool]:
    """Detect AWB / CWB bays on one 2-D Z field (relative coords)."""
    if not np.isfinite(z2d).any():
        return False, False
    x = x_rel[0, :] if x_rel.ndim == 2 else x_rel
    y = y_rel[:, 0] if y_rel.ndim == 2 else y_rel

    contours = sampled_longest_contours(
        z2d, x, y,
        try_levels=cfg.try_levels,
        max_keep=12,
        min_vertices=cfg.min_vertices,
    )
    if not contours:
        return False, False

    is_awb = is_cwb = False
    for c in contours:
        xline, yline = c["x"], c["y"]
        intervals = overturn_x_intervals(
            xline, yline,
            n_meridians=cfg.n_meridians,
            min_cross=cfg.min_cross,
        )
        for xa, xb in intervals:
            poly = envelope_polygon(
                xline, yline, xa, xb,
                n_samp=cfg.n_samp,
                min_points=cfg.min_points,
            )
            if poly is None:
                continue
            xp, yp, xm, y_min, y_max = poly
            area, _ = poly_area_centroid(xp, yp)
            if abs(area) <= cfg.area_min_deg2:
                continue

            wb_type, _ = classify_bay(
                xline, yline, xa, xb,
                n_samp=max(80, cfg.n_samp // 2),
            )
            if wb_type == "UNK":
                slope = centerline_tilt(xm, y_min, y_max)
                if not np.isfinite(slope):
                    continue
                wb_type = "AWB" if slope < 0 else "CWB"

            if wb_type == "AWB":
                is_awb = True
            if wb_type == "CWB":
                is_cwb = True
            if is_awb and is_cwb:
                return True, True
    return is_awb, is_cwb


def _classify_multilevel(
    z3d: np.ndarray | None,
    levels_file: np.ndarray | None,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    *,
    classify_levels: Sequence[int | str],
    threshold: int,
    cfg: RWBConfig,
    z2d_wavg: np.ndarray | None = None,
) -> tuple[bool, bool]:
    """Multi-level classification; require *threshold* levels to agree.

    *classify_levels* may contain integer hPa values or the string
    ``"wavg"``; the latter uses the pre-computed weighted-average 2-D
    Z field (*z2d_wavg*).
    """
    awb_count = cwb_count = 0
    for lev in classify_levels:
        if isinstance(lev, str) and lev.lower() == "wavg":
            if z2d_wavg is None:
                continue
            awb, cwb = _classify_bays_z2d(z2d_wavg, x_rel, y_rel, cfg)
        else:
            if z3d is None or levels_file is None:
                continue
            k = int(np.nanargmin(np.abs(levels_file - int(lev))))
            if k >= z3d.shape[0]:
                continue
            awb, cwb = _classify_bays_z2d(z3d[k], x_rel, y_rel, cfg)
        awb_count += int(awb)
        cwb_count += int(cwb)
    return awb_count >= threshold, cwb_count >= threshold


# ── Result container ──────────────────────────────────────────────────

@dataclass
class ClassifyResult:
    """Holds RWB variant classification results.

    Attributes:
        stage_all: ``{stage: set_of_track_ids}``
        stage_awb: ``{stage: set_of_AWB_track_ids}``
        stage_cwb: ``{stage: set_of_CWB_track_ids}``
        stage_neu: ``{stage: set_of_NEUTRAL_track_ids}``
        h_scale: Captured from the first NPZ file.
        stages: Ordered stage names.
        classify_levels: Pressure levels used.
        classify_threshold: Threshold used.
    """

    stage_all: dict[str, set[int]]
    stage_awb: dict[str, set[int]]
    stage_cwb: dict[str, set[int]]
    stage_neu: dict[str, set[int]]
    h_scale: float | None
    stages: list[str]
    classify_levels: list[int]
    classify_threshold: int

    # ── derived look-ups ──

    @property
    def variant_trackset(self) -> dict[str, frozenset[int]]:
        """Variant → frozenset mapping, e.g. ``AWB_onset``."""
        out: dict[str, frozenset[int]] = {}
        for evt in self.stages:
            out[f"AWB_{evt}"] = frozenset(self.stage_awb.get(evt, set()))
            out[f"CWB_{evt}"] = frozenset(self.stage_cwb.get(evt, set()))
            out[f"NEUTRAL_{evt}"] = frozenset(self.stage_neu.get(evt, set()))
        return out

    @property
    def stage_labels(self) -> dict[str, dict[int, str]]:
        """``{stage: {track_id: label}}`` where label ∈ AWB/CWB/NEUTRAL/Omega."""
        out: dict[str, dict[int, str]] = {}
        for evt in self.stages:
            lbl: dict[int, str] = {}
            amb = self.stage_awb.get(evt, set()) & self.stage_cwb.get(evt, set())
            for tid in sorted(self.stage_all.get(evt, set())):
                if tid in amb:
                    lbl[tid] = "Omega"
                elif tid in self.stage_awb.get(evt, set()):
                    lbl[tid] = "AWB"
                elif tid in self.stage_cwb.get(evt, set()):
                    lbl[tid] = "CWB"
                else:
                    lbl[tid] = "NEUTRAL"
            out[evt] = lbl
        return out

    @property
    def stage_tracksets(self) -> dict[str, dict[str, frozenset[int]]]:
        out: dict[str, dict[str, frozenset[int]]] = {}
        for evt in self.stages:
            out[evt] = {
                "ALL": frozenset(self.stage_all.get(evt, set())),
                "AWB": frozenset(self.stage_awb.get(evt, set())),
                "CWB": frozenset(self.stage_cwb.get(evt, set())),
                "NEUTRAL": frozenset(self.stage_neu.get(evt, set())),
                "Omega": frozenset(
                    self.stage_awb.get(evt, set())
                    & self.stage_cwb.get(evt, set())
                ),
            }
        return out

    # ── I/O ──

    def save(self, path: Path | str) -> Path:
        """Persist to pickle (same format as core script)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stage_ALL": {k: set(v) for k, v in self.stage_all.items()},
            "stage_AWB": {k: set(v) for k, v in self.stage_awb.items()},
            "stage_CWB": {k: set(v) for k, v in self.stage_cwb.items()},
            "stage_NEU": {k: set(v) for k, v in self.stage_neu.items()},
            "H_SCALE": self.h_scale,
            "variant_trackset": self.variant_trackset,
            "RWB_STAGE_LABELS": self.stage_labels,
            "RWB_STAGE_TRACKSETS": self.stage_tracksets,
            "CLASSIFY_LEVELS": self.classify_levels,
            "CLASSIFY_THRESHOLD": self.classify_threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[saved] RWB variant tracksets → {path}", flush=True)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ClassifyResult":
        """Load from a previously-saved PKL."""
        path = Path(path)
        with open(path, "rb") as f:
            d = pickle.load(f)
        stages = sorted(d["stage_ALL"].keys())
        return cls(
            stage_all={k: set(v) for k, v in d["stage_ALL"].items()},
            stage_awb={k: set(v) for k, v in d["stage_AWB"].items()},
            stage_cwb={k: set(v) for k, v in d["stage_CWB"].items()},
            stage_neu={k: set(v) for k, v in d["stage_NEU"].items()},
            h_scale=d.get("H_SCALE"),
            stages=stages,
            classify_levels=d.get("CLASSIFY_LEVELS", [500, 400, 300, 200]),
            classify_threshold=d.get("CLASSIFY_THRESHOLD", 3),
        )


# ── Main entry point ──────────────────────────────────────────────────

def run_pass1(cfg: ClassifyConfig) -> ClassifyResult:
    """Run Pass-1 RWB classification from NPZ files.

    Reads ``dh=0`` snapshots under ``cfg.npz_dir/{stage}/dh=+0/``
    and classifies each track as AWB / CWB / NEUTRAL.

    Returns:
        :class:`ClassifyResult` holding variant sets.
    """
    excluded = _load_excluded(cfg.exclude_file)
    if excluded:
        print(f"[exclude] {len(excluded)} track IDs", flush=True)

    h_scale: float | None = None
    stage_all: dict[str, set[int]] = {e: set() for e in cfg.stages}
    stage_awb: dict[str, set[int]] = {e: set() for e in cfg.stages}
    stage_cwb: dict[str, set[int]] = {e: set() for e in cfg.stages}
    stage_neu: dict[str, set[int]] = {}

    _need_wavg = any(
        isinstance(l, str) and l.lower() == "wavg" for l in cfg.classify_levels
    )
    _need_3d = any(
        not (isinstance(l, str) and l.lower() == "wavg") for l in cfg.classify_levels
    )

    print(f"\n[pass1] classifying at levels {cfg.classify_levels}  "
          f"(threshold={cfg.classify_threshold})", flush=True)

    for evt in cfg.stages:
        evt_dir = cfg.npz_dir / evt
        if not evt_dir.exists():
            continue

        # Find dh=0 directory
        dh0_dir = None
        for cand in ("dh=+0", "dh=0", "dh=-0"):
            d = evt_dir / cand
            if d.exists():
                dh0_dir = d
                break
        if dh0_dir is None:
            print(f"[warn] no dh=0 directory for {evt}", flush=True)
            continue

        npz_files = sorted(dh0_dir.glob("*.npz"))
        n_ok = n_fail = 0

        for fp in npz_files:
            tid = _parse_track_id(fp)
            if tid is None or tid in excluded:
                continue
            stage_all[evt].add(tid)
            try:
                with np.load(fp, allow_pickle=False) as Z:
                    if h_scale is None and "H_SCALE" in Z.files:
                        h_scale = float(Z["H_SCALE"])
                    x_rel = Z["X_rel"]
                    y_rel = Z["Y_rel"]

                    z3d = None
                    levels_file = None
                    z2d_wavg = None
                    if _need_3d and "z_3d" in Z.files:
                        z3d = Z["z_3d"]
                        levels_file = np.asarray(Z["levels"], dtype=float)
                    if _need_wavg and "z" in Z.files:
                        z2d_wavg = Z["z"]

                    if z3d is None and z2d_wavg is None:
                        continue

                    awb, cwb = _classify_multilevel(
                        z3d, levels_file, x_rel, y_rel,
                        classify_levels=cfg.classify_levels,
                        threshold=cfg.classify_threshold,
                        cfg=cfg.rwb_cfg,
                        z2d_wavg=z2d_wavg,
                    )
                    if awb:
                        stage_awb[evt].add(tid)
                    if cwb:
                        stage_cwb[evt].add(tid)
                    n_ok += 1
            except Exception:
                n_fail += 1
                continue

        print(f"[classify] {evt}: ok={n_ok}  fail={n_fail}", flush=True)

    for evt in cfg.stages:
        stage_neu[evt] = stage_all[evt] - (
            stage_awb.get(evt, set()) | stage_cwb.get(evt, set())
        )
        print(
            f"[classify] {evt}: ALL={len(stage_all[evt])}  "
            f"AWB={len(stage_awb.get(evt, set()))}  "
            f"CWB={len(stage_cwb.get(evt, set()))}  "
            f"NEU={len(stage_neu[evt])}",
            flush=True,
        )

    return ClassifyResult(
        stage_all=stage_all,
        stage_awb=stage_awb,
        stage_cwb=stage_cwb,
        stage_neu=stage_neu,
        h_scale=h_scale,
        stages=list(cfg.stages),
        classify_levels=list(cfg.classify_levels),
        classify_threshold=cfg.classify_threshold,
    )
