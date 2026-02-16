"""NPZ patch file I/O utilities."""

from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np


def load_npz_patch(path: str | Path) -> dict[str, np.ndarray]:
    """Load an event NPZ patch file.

    Parameters:
        path: Path to .npz file.

    Returns:
        Dict-like NpzFile mapping field names → arrays.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ not found: {path}")
    return dict(np.load(path, allow_pickle=True))


def list_npz_patches(
    root_dir: str | Path,
    stage: str | None = None,
    dh: int | None = None,
) -> list[Path]:
    """List all NPZ patch files under a directory.

    Parameters:
        root_dir: Root directory (e.g., composite_blocking_tempest/).
        stage: Filter by stage (onset/peak/decay).
        dh: Filter by hour offset.

    Returns:
        Sorted list of Path objects.
    """
    root = Path(root_dir)
    if stage and dh is not None:
        pattern = f"{stage}/dh={dh:+d}/*.npz"
    elif stage:
        pattern = f"{stage}/**/*.npz"
    else:
        pattern = "**/*.npz"
    return sorted(root.glob(pattern))


def npz_metadata(path: str | Path) -> dict:
    """Extract metadata (track_id, lat0, lon0, ts, dh) from NPZ.

    Parameters:
        path: Path to .npz file.

    Returns:
        Dict with scalar metadata fields.
    """
    npz = np.load(path, allow_pickle=True)
    meta = {}
    for key in ("track_id", "lat0", "lon0", "ts", "dh", "center_mode",
                "center_lat", "center_lon"):
        if key in npz.files:
            val = npz[key]
            meta[key] = val.item() if val.ndim == 0 else val
    return meta
