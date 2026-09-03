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
import functools
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .rwb import (
    RWBConfig,
    circumpolar_contours,
    crop_contour_to_patch,
    sampled_longest_contours,
    overturn_x_intervals,
    envelope_polygon,
    poly_area_centroid,
    classify_bay,
    centerline_tilt,
)

# ── regex helpers ────────────────────────────────────────────────────
# Accepts both naming schemes: ERA5 ``track_1234_...`` (numeric id) and
# CESM ``track_m091_t00002_...``.  The CESM id keeps its member prefix as a
# STRING -- a bare int would collide across members (every member has a
# t00002) and silently merge different events into one trackset entry.
_TRACK_RE = re.compile(r"track_((?:m\d+_t)?\d+)_")
_DH_RE = re.compile(r"^dh=([+\-]?\d+)$")


def _parse_track_id(fp: Path) -> int | str | None:
    m = _TRACK_RE.search(fp.name)
    if not m:
        return None
    g = m.group(1)
    return g if g.startswith("m") else int(g)


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
        n_workers: Parallel worker processes for the per-file
            classification loop (1 = serial). Files are independent, so
            this only shards the loop; results are merged in the parent.
        contours: Where the contours come from. ``"circumpolar"`` finds them
            on the hemisphere and crops them to the patch, which is what wave
            breaking means and needs *archive_dir*; ``"patch"`` contours the
            patch itself, which is all a caller without the archive can do;
            ``"auto"`` is the first when an archive is given and the second
            when it is not.
        source: Which archive the hemisphere field comes from, ``"era5"`` or
            ``"cesm"``.
        archive_dir: Root of that archive.
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
    n_workers: int = 1
    contours: str = "auto"
    source: str = "era5"
    archive_dir: Path | None = None

    def contour_source(self) -> str:
        """``"circumpolar"`` or ``"patch"``, with the reason a choice is refused."""
        if self.contours == "auto":
            return "circumpolar" if self.archive_dir else "patch"
        if self.contours == "circumpolar" and not self.archive_dir:
            raise ValueError(
                "circumpolar contours are found on the hemisphere, which is not in "
                "a per-event record: give archive_dir, or ask for contours='patch'"
            )
        if self.contours not in ("circumpolar", "patch"):
            raise ValueError(
                f"contours must be circumpolar, patch or auto, got {self.contours!r}")
        return self.contours


# ── Excluded track loader ─────────────────────────────────────────────

def _load_excluded(p: Path | None) -> set[int | str]:
    """Track ids to exclude. Numeric ids load as int (ERA5); ids like
    ``m091_t00002`` load as str — mirroring ``_parse_track_id``."""
    ids: set[int | str] = set()
    if p is None or not p.exists():
        return ids

    def _add(tok: str) -> None:
        tok = tok.strip()
        if not tok:
            return
        try:
            ids.add(int(tok))
        except ValueError:
            m = re.search(r"m\d+_t\d+", tok)
            if m:
                ids.add(m.group(0))

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
                            _add(row[col])
                        except KeyError:
                            pass
            else:
                for line in f:
                    m = re.search(r"m\d+_t\d+", line) or re.search(r"\d+", line)
                    if m:
                        _add(m.group(0))
    except Exception:
        pass
    return ids


# ── Single-level bay classifier ───────────────────────────────────────

def _patch_axes(x_rel: np.ndarray, y_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The patch's own relative axes, whether they arrive as vectors or as grids."""
    x = x_rel[0, :] if x_rel.ndim == 2 else x_rel
    y = y_rel[:, 0] if y_rel.ndim == 2 else y_rel
    return x, y


def _circumpolar_on_patch(
    field_nh: np.ndarray,
    lat_nh: np.ndarray,
    lon_nh: np.ndarray,
    centre_lat: float,
    centre_lon: float,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    cfg: RWBConfig,
    max_keep: int = 12,
) -> list[dict]:
    """Contours that encircle the pole, cropped to the event's patch.

    Wave breaking is the overturning of a contour that goes round the hemisphere,
    so the contour is found on the hemisphere and only then cut to the patch. A
    contour that does not span ``cfg.circumpolar_min_lon_span`` is not one of
    those and is dropped, and so is a crop that keeps too little of one to have a
    shape.

    The survivors are then thinned to *max_keep*, evenly across the levels they
    came from, exactly as the patch's own contours are. Without it the hemisphere
    hands over two to three hundred contours where the patch hands over twelve,
    and twenty times the contours is twenty times the chances of finding a bay --
    which would make this look better for a reason that has nothing to do with
    being circumpolar.
    """
    x, y = _patch_axes(x_rel, y_rel)
    circ = circumpolar_contours(
        field_nh, lat_nh, lon_nh,
        try_levels=cfg.try_levels,
        min_vertices=cfg.min_vertices,
        min_lon_span=cfg.circumpolar_min_lon_span,
    )
    half_dlat = float(np.max(np.abs(y)))
    half_dlon = float(np.max(np.abs(x)))
    out = []
    for cc in circ:
        cropped = crop_contour_to_patch(
            cc, float(centre_lat), float(centre_lon),
            half_dlat=half_dlat, half_dlon=half_dlon,
        )
        if cropped is not None:
            out.append(cropped)
    if len(out) <= max_keep:
        return out
    idx = np.linspace(0, len(out) - 1, num=max_keep).round().astype(int)
    return [out[i] for i in idx]


def _classify_bays(contours: list[dict], cfg: RWBConfig) -> tuple[bool, bool]:
    """Whether a set of contours overturns anticyclonically, cyclonically, or both.

    This is the whole of the classification: which contours it is given is the
    only thing that distinguishes a circumpolar run from a patch one.
    """
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


def _classify_bays_z2d(
    z2d: np.ndarray,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    cfg: RWBConfig,
) -> tuple[bool, bool]:
    """Detect AWB / CWB bays from the patch's own contours (relative coords)."""
    if not np.isfinite(z2d).any():
        return False, False
    x, y = _patch_axes(x_rel, y_rel)
    return _classify_bays(
        sampled_longest_contours(
            z2d, x, y,
            try_levels=cfg.try_levels,
            max_keep=12,
            min_vertices=cfg.min_vertices,
        ),
        cfg,
    )


def classify_z_field(
    z2d: np.ndarray,
    x_rel: np.ndarray,
    y_rel: np.ndarray,
    cfg: RWBConfig | None = None,
) -> tuple[bool, bool]:
    """Classify AWB / CWB Rossby-wave-breaking bays on one 2-D Z field.

    Public, side-effect-free wrapper around the internal bay-overturning
    detector :func:`_classify_bays_z2d`. Intended for callers that already
    hold a single 2-D geopotential-height patch — e.g. the weighted-average
    Z(300/250/200) "single-field, threshold=1" classification used outside
    the per-event NPZ tree (other grids/datasets, precomputed patch arrays).

    Args:
        z2d: 2-D geopotential-height field ``(NY, NX)`` [m] on a block- or
            event-relative grid. NaNs (e.g. polar padding) are tolerated; an
            all-NaN field returns ``(False, False)``.
        x_rel: Relative longitudes — 1-D ``(NX,)`` or 2-D ``(NY, NX)``.
        y_rel: Relative latitudes — 1-D ``(NY,)`` or 2-D ``(NY, NX)``,
            ascending (north at top).
        cfg: :class:`~pvtend.rwb.RWBConfig`; ``None`` uses the ``RWBConfig()``
            dataclass defaults (``area_min_deg2=30, try_levels=300``). Note the
            ``pvtend-pipeline classify`` CLI tunes these to
            ``area_min_deg2=20, try_levels=400`` for ERA5; pass an explicit
            ``cfg`` to reproduce that.

    Returns:
        ``(is_awb, is_cwb)`` — both ``True`` when the field exhibits *both*
        an anticyclonic and a cyclonic overturning bay (an Omega block); see
        :func:`label_from_flags` to collapse to a single class string.
    """
    if cfg is None:
        cfg = RWBConfig()
    return _classify_bays_z2d(z2d, x_rel, y_rel, cfg)


def label_from_flags(is_awb: bool, is_cwb: bool) -> str:
    """Collapse ``(AWB, CWB)`` flags to one physical-class string.

    Mirrors :attr:`ClassifyResult.stage_labels` semantics exactly:
    both → ``"Omega"``, AWB only → ``"AWB"``, CWB only → ``"CWB"``,
    neither → ``"NEUTRAL"``.
    """
    if is_awb and is_cwb:
        return "Omega"
    if is_awb:
        return "AWB"
    if is_cwb:
        return "CWB"
    return "NEUTRAL"


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
    hemisphere: tuple | None = None,
    centre: tuple[float, float] | None = None,
) -> tuple[bool, bool]:
    """Multi-level classification; require *threshold* levels to agree.

    *classify_levels* may contain integer hPa values or the string
    ``"wavg"``; the latter uses the pre-computed weighted-average 2-D
    Z field (*z2d_wavg*), and takes its contours from the hemisphere when
    *hemisphere* -- ``(field, lat, lon)`` -- and *centre* are given. The
    individual pressure levels are only in the record's own patch, so they are
    contoured there whatever the wavg level does.
    """
    awb_count = cwb_count = 0
    for lev in classify_levels:
        if isinstance(lev, str) and lev.lower() == "wavg":
            if z2d_wavg is None:
                continue
            if hemisphere is not None and centre is not None:
                awb, cwb = _classify_bays(
                    _circumpolar_on_patch(*hemisphere, centre[0], centre[1],
                                          x_rel, y_rel, cfg),
                    cfg,
                )
            else:
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

#: One hemisphere reader per worker process, keyed by the archive it reads.
#: A pool worker classifies many events and the reader holds the open files;
#: rebuilding it per event would reopen the archive every time.
_HEMISPHERE: dict = {}

_STAMP = re.compile(r"_(\d{10})_dh")


def _event_time(path: Path):
    """The event's own time, from the name the writer gave the record."""
    m = _STAMP.search(path.name)
    if m is None:
        return None
    return pd.Timestamp(
        f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]} {m.group(1)[8:10]}:00")


def _hemisphere_for(source, archive_dir, path, tid):
    """``(field, lat, lon)`` for this event, or ``None`` if it cannot be had.

    A record whose time the archive does not hold is an error worth seeing, not
    an event quietly classified from nothing, so the caller turns it into a
    failed file rather than a Neutral label.
    """
    from .hemisphere import HemisphereFields, member_of

    key = (source, str(archive_dir))
    reader = _HEMISPHERE.get(key)
    if reader is None:
        reader = _HEMISPHERE[key] = HemisphereFields(
            source=source, data_dir=Path(archive_dir))
    when = _event_time(path)
    if when is None:
        raise ValueError(f"no event time in {path.name}")
    return reader.wavg_height(when, member=member_of(tid))


def _classify_one_file(item, *, classify_levels, threshold, rwb_cfg,
                       need_3d, need_wavg, source=None, archive_dir=None):
    """Classify a single NPZ; picklable unit for the run_pass1 pool.

    With *archive_dir* the wavg level takes its contours from the hemisphere at
    the event's time; without it, from the patch.

    Returns ``(tid, awb, cwb, h_scale, status)`` with status ``"ok"``,
    ``"noz"`` (file lacks the requested Z fields — not an error) or
    ``"fail"``.
    """
    fp, tid = item
    try:
        with np.load(fp, allow_pickle=False) as Z:
            h_scale = float(Z["H_SCALE"]) if "H_SCALE" in Z.files else None
            x_rel = Z["X_rel"]
            y_rel = Z["Y_rel"]

            z3d = None
            levels_file = None
            z2d_wavg = None
            if need_3d and "z_3d" in Z.files:
                z3d = Z["z_3d"]
                levels_file = np.asarray(Z["levels"], dtype=float)
            if need_wavg and "z" in Z.files:
                z2d_wavg = Z["z"]

            if z3d is None and z2d_wavg is None:
                return (tid, False, False, h_scale, "noz")

            hemisphere = centre = None
            if archive_dir is not None and need_wavg:
                hemisphere = _hemisphere_for(source, archive_dir, Path(fp), tid)
                centre = (float(Z["center_lat"]), float(Z["center_lon"]))

            awb, cwb = _classify_multilevel(
                z3d, levels_file, x_rel, y_rel,
                classify_levels=classify_levels,
                threshold=threshold,
                cfg=rwb_cfg,
                z2d_wavg=z2d_wavg,
                hemisphere=hemisphere,
                centre=centre,
            )
            return (tid, awb, cwb, h_scale, "ok")
    except Exception:
        return (tid, False, False, None, "fail")


def run_pass1(cfg: ClassifyConfig) -> ClassifyResult:
    """Run Pass-1 RWB classification from NPZ files.

    Reads ``dh=0`` snapshots under ``cfg.npz_dir/{stage}/dh=+0/``
    and classifies each track as AWB / CWB / NEUTRAL.

    Returns:
        :class:`ClassifyResult` holding variant sets.
    """
    excluded = _load_excluded(cfg.exclude_file)
    _contours = cfg.contour_source()
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

        todo = []
        for fp in npz_files:
            tid = _parse_track_id(fp)
            if tid is None or tid in excluded:
                continue
            stage_all[evt].add(tid)
            todo.append((fp, tid))

        work = functools.partial(
            _classify_one_file,
            classify_levels=cfg.classify_levels,
            threshold=cfg.classify_threshold,
            rwb_cfg=cfg.rwb_cfg,
            need_3d=_need_3d,
            need_wavg=_need_wavg,
            source=cfg.source,
            archive_dir=cfg.archive_dir if _contours == "circumpolar" else None,
        )
        if cfg.n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
                results = ex.map(work, todo, chunksize=16)
                results = list(results)
        else:
            results = map(work, todo)

        for tid, awb, cwb, hs, status in results:
            if status == "fail":
                n_fail += 1
                continue
            if h_scale is None and hs is not None:
                h_scale = hs
            if status == "noz":
                continue
            if awb:
                stage_awb[evt].add(tid)
            if cwb:
                stage_cwb[evt].add(tid)
            n_ok += 1

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
