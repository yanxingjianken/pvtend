"""Reading model output and writing results.

The loaders assert the shape of what they read rather than trusting it.  A level
axis in the wrong order or a climatology on a different grid from the state costs
a whole batch, and both look like ordinary output until someone plots it.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

#: Variable names in the CESM wu9 files.
CESM_VARS = {"height": "Z3", "temperature": "T", "u": "U", "v": "V"}


@dataclass
class HemisphereFields:
    """One state on a northern-hemisphere latitude-longitude grid."""

    height: np.ndarray
    temperature: np.ndarray
    u: np.ndarray
    v: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    p_hpa: np.ndarray

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.height, self.temperature, self.u, self.v


def _check_axes(ds, p_hpa_expected: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)
    plev = np.asarray(ds["plev"].values, dtype=float)
    if lat[0] > lat[-1]:
        raise ValueError(
            "latitudes must ascend towards the pole; this file is stored "
            "north-to-south"
        )
    if lat[0] < -1.0:
        raise ValueError(
            f"expected a northern-hemisphere file, but latitudes start at {lat[0]}"
        )
    if plev[0] < plev[-1]:
        raise ValueError(
            "pressure levels must run bottom-up (1000 hPa first); this file is "
            "top-down"
        )
    if p_hpa_expected is not None and not np.allclose(plev, p_hpa_expected):
        raise ValueError(
            f"file levels {plev} do not match the configured set {p_hpa_expected}"
        )
    return lat, lon, plev


def load_cesm_state(path: str, index: int, p_hpa: np.ndarray | None = None) -> HemisphereFields:
    """Read one time from a CESM wu9 file.

    Args:
        path: NetCDF path.
        index: Position along the file's leading axis (``time`` for an event
            file, ``slot`` for a climatology).
        p_hpa: Expected pressure levels; checked when given.
    """
    import xarray as xr

    with xr.open_dataset(path) as ds:
        lat, lon, plev = _check_axes(ds, p_hpa)
        axis = "time" if "time" in ds.dims else "slot"
        if index < 0 or index >= ds.sizes[axis]:
            raise IndexError(
                f"index {index} is outside the file's {ds.sizes[axis]} {axis} steps"
            )
        fields = {}
        for key, name in CESM_VARS.items():
            if name not in ds:
                raise KeyError(f"{path} has no variable {name!r}")
            fields[key] = np.asarray(
                ds[name].isel({axis: index}).values, dtype=float
            )
            if fields[key].shape != (plev.size, lat.size, lon.size):
                raise ValueError(
                    f"{name} has shape {fields[key].shape}, expected "
                    f"{(plev.size, lat.size, lon.size)}"
                )
    return HemisphereFields(lat=lat, lon=lon, p_hpa=plev, **fields)


def climatology_slot(month: int, day: int, hour: int, path: str) -> int:
    """Position of a date in a six-hourly climatology file.

    Resolved by matching the file's own ``month``/``day``/``hour`` variables
    rather than by arithmetic on the day of year: the arithmetic differs between
    calendars, and getting it wrong shifts the whole climatology by a season
    without changing anything about how the output looks.
    """
    import xarray as xr

    with xr.open_dataset(path) as ds:
        for name in ("month", "day", "hour"):
            if name not in ds:
                raise KeyError(
                    f"{path} has no {name!r} variable, so a slot cannot be matched"
                )
        match = (
            (ds["month"].values == month)
            & (ds["day"].values == day)
            & (ds["hour"].values == hour)
        )
    hits = np.flatnonzero(match)
    if hits.size != 1:
        raise ValueError(
            f"{hits.size} climatology slots match {month:02d}-{day:02d} "
            f"{hour:02d}Z; expected exactly one"
        )
    return int(hits[0])


def save_npz(path: str, arrays: dict[str, np.ndarray], meta: dict) -> None:
    """Write results atomically, so an interrupted batch leaves no half file.

    A partially written file is worse than a missing one: a resumable batch skips
    what already exists, so a truncated result would be kept and never revisited.
    """
    payload = dict(arrays)
    for key, value in meta.items():
        payload[f"meta_{key}"] = np.asarray(value)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=directory, suffix=".npz.tmp", delete=False
    )
    try:
        np.savez_compressed(handle, **payload)
        handle.close()
        os.replace(handle.name, path)
    except BaseException:
        handle.close()
        if os.path.exists(handle.name):
            os.unlink(handle.name)
        raise
