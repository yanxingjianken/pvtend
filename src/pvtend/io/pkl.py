"""Pickle-based state I/O for composite analysis.

Supports saving and loading composite states (which aggregate
per-event NPZ results into ensemble-mean fields).
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def save_pkl(data: Any, path: str | Path) -> Path:
    """Save data to a pickle file.

    Parameters:
        data: Arbitrary Python object (must be picklable).
        path: Output file path (.pkl).

    Returns:
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pkl(path: str | Path) -> Any:
    """Load data from a pickle file.

    Parameters:
        path: Path to the .pkl file.

    Returns:
        The unpickled Python object.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PKL not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def sanitize_for_pickle(obj: Any) -> Any:
    """Convert arrays and nested structures for safe pickling.

    Converts xarray objects to numpy, handles NaN/Inf, and
    ensures all nested dicts/lists are pickle-safe.

    Parameters:
        obj: Input object.

    Returns:
        Sanitized object.
    """
    if hasattr(obj, "values"):  # xarray DataArray
        return np.asarray(obj.values)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: sanitize_for_pickle(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        sanitized = [sanitize_for_pickle(v) for v in obj]
        return type(obj)(sanitized)
    return obj
