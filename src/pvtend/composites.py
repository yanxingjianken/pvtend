"""NPZ composite accumulation and PKL export/load.

Accumulates per-event NPZ patch fields into running sums and valid-point
counts, grouped by event stage (onset/peak/decay) and RWB variant
(original/AWB_onset/CWB_peak/etc.). The accumulated state can be exported
to a pickle file for rapid loading in analysis notebooks.
"""

from __future__ import annotations

import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

import numpy as np


@dataclass(frozen=True)
class CompositeState:
    """Lightweight wrapper around exported composite globals.

    Provides `composite_mean_3d()` and `composite_reduce()` methods to
    retrieve composite-mean fields from the accumulated sums.
    """

    globals_map: Mapping[str, object]

    def __post_init__(self) -> None:
        required = {
            "FIELDS3D", "LEVELS", "SUMS3D", "VALID3D",
            "FILECOUNT", "SUMS3D_V", "VALID3D_V",
            "FILECOUNT_V", "COMPOSITE_VARIANTS",
        }
        missing = sorted(name for name in required if name not in self.globals_map)
        if missing:
            raise KeyError(f"Composite globals missing: {missing}")

    def list_fields3d(self) -> tuple[str, ...]:
        """Return the tuple of 3D field names."""
        return tuple(self.globals_map.get("FIELDS3D", ()))

    def composite_mean_3d(
        self,
        field: str,
        stage: str,
        dh: int,
        *,
        variant: str | None = "original",
    ) -> np.ndarray | None:
        """Return composite mean of a 3D field.

        Parameters:
            field: Field name (e.g., 'pv_3d', 'z_3d').
            stage: Event stage name (e.g., 'onset', 'peak', 'decay').
            dh: Hour offset from stage reference time.
            variant: Composite variant key (default 'original').

        Returns:
            3D numpy array of composite-mean values, or None if unavailable.
        """
        variant_key = self._norm_variant(variant)
        stage_key = self._norm_stage(stage)
        sums, valids = self._pick_store3d(variant_key, stage_key, int(dh))
        if sums is None or valids is None:
            return None
        arr_sum = _safe_lookup(sums, field)
        vcount = _safe_lookup(valids, field)
        if arr_sum is None or vcount is None:
            return None
        arr_sum = np.asarray(arr_sum, dtype=np.float64)
        vcount = np.asarray(vcount, dtype=np.float64)
        out = np.full_like(arr_sum, np.nan, dtype=np.float64)
        mask = vcount > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            np.divide(arr_sum, vcount, out=out, where=mask)
        return out

    def composite_reduce(
        self,
        field: str,
        stage: str,
        dh: int,
        *,
        variant: str | None = "original",
        level_mode=None,
    ) -> np.ndarray | None:
        """Get composite-mean field, optionally reduced to 2D.

        Parameters:
            field: Field name (e.g., 'pv_3d', 'z_3d').
            stage: Event stage name.
            dh: Hour offset from stage reference time.
            variant: Composite variant key (default 'original').
            level_mode: One of None/'all'/'3d' (return full 3D),
                'wavg'/'weighted' (height-weighted vertical mean),
                or a numeric pressure level in hPa.

        Returns:
            Numpy array (3D or 2D depending on level_mode), or None.
        """
        arr3d = self.composite_mean_3d(field, stage, dh, variant=variant)
        if arr3d is None:
            return None
        if level_mode in (None, "", "all", "3d"):
            return np.array(arr3d, copy=True)
        if isinstance(level_mode, str) and level_mode.lower() in {"wavg", "weighted"}:
            z3d = self.composite_mean_3d(
                self._resolve_z_name(), stage, dh, variant=variant)
            h_scale = float(self.globals_map.get("H_SCALE", 7000.0))
            if z3d is None:
                raise ValueError("No z_3d for weighted average")
            w = np.exp(-z3d / h_scale)
            num = np.nansum(w * arr3d, axis=0)
            den = np.nansum(w, axis=0)
            out = np.full_like(num, np.nan)
            out[den > 0] = num[den > 0] / den[den > 0]
            return out
        level_val = float(level_mode)
        levels = np.asarray(self.globals_map.get("LEVELS", ()), dtype=float)
        idx = int(np.nanargmin(np.abs(levels - level_val)))
        return arr3d[idx]

    def _norm_stage(self, stage: str) -> str:
        """Normalize stage name to match stored keys (case-insensitive)."""
        for k in self._stage_names:
            if k.lower() == stage.strip().lower():
                return k
        raise KeyError(f"Unknown stage {stage!r}")

    def _norm_variant(self, variant: str | None) -> str:
        """Normalize variant name to match stored keys (case-insensitive)."""
        if not variant:
            return "original"
        for c in self.globals_map.get("COMPOSITE_VARIANTS", ()):
            if str(c).lower() == str(variant).strip().lower():
                return str(c)
        raise KeyError(f"Unknown variant {variant!r}")

    def _pick_store3d(self, variant, stage, dh):
        """Select the appropriate sums/valids dicts for a variant+stage+dh."""
        if variant == "original":
            sums_map = self.globals_map.get("SUMS3D", {})
            valids_map = self.globals_map.get("VALID3D", {})
        else:
            sums_map = self.globals_map.get("SUMS3D_V", {}).get(variant, {})
            valids_map = self.globals_map.get("VALID3D_V", {}).get(variant, {})
        s_stage = _safe_lookup(sums_map, stage)
        v_stage = _safe_lookup(valids_map, stage)
        if s_stage is None or v_stage is None:
            return None, None
        return _safe_lookup(s_stage, dh), _safe_lookup(v_stage, dh)

    def _resolve_z_name(self) -> str:
        """Resolve the geopotential height field name."""
        fields = self.list_fields3d()
        if "z_3d" in fields:
            return "z_3d"
        if "z" in fields:
            return "z"
        raise KeyError("No z_3d/z field found")

    @property
    def _stage_names(self) -> tuple[str, ...]:
        """Return stage names from the FILECOUNT dict keys."""
        fc = self.globals_map.get("FILECOUNT", {})
        return tuple(fc.keys()) if isinstance(fc, dict) else ()


def _safe_lookup(mapping, key):
    """Safely look up a key in a mapping-like object.

    Parameters:
        mapping: A dict, Mapping, or object with a .get() method.
        key: The key to look up.

    Returns:
        The value, or None if not found or mapping is None.
    """
    if mapping is None:
        return None
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    if hasattr(mapping, "get"):
        return mapping.get(key)
    return None


def load_composite_state(path: str | Path) -> CompositeState:
    """Load CompositeState from a pickle file.

    Parameters:
        path: Path to the exported pickle.

    Returns:
        CompositeState instance.
    """
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, dict) and "globals" in data and len(data) == 1:
        data = data["globals"]
    return CompositeState(globals_map=data)


def save_composite_state(
    state_dict: Mapping[str, object],
    path: str | Path,
) -> Path:
    """Save composite state as pickle (highest protocol).

    Parameters:
        state_dict: Dictionary of composite globals.
        path: Output path.

    Returns:
        Path to written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"globals": dict(state_dict)}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return path
