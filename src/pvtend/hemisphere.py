"""The column-weighted height over the whole northern hemisphere, by event time.

Wave breaking is the overturning of a contour that encircles the pole, so the
contours have to be drawn on the hemisphere and only then cropped to the event's
patch. A per-event record holds the patch alone, so the hemisphere field is read
back from the archive here, and it is the *same* quantity the record's own
two-dimensional ``z`` holds: the average of geopotential height over the
upper-tropospheric levels, each level weighted by ``exp(-z/H)`` and the sum
divided by the weight of the levels that are valid at that point. Checked
against three ERA5 events of the finished store: cropped to the patch, this
field equals the stored key exactly, to the last bit.

The archives differ in what identifies a field. ERA5 is one file per variable and
month, so a time is enough; the CESM ensemble is one file per member and year and
two members share a timestamp, so the member is part of the key and of the open.

Caching buys less than it looks. The catalogues have 85,425 CESM events over
78,166 distinct (member, time) pairs and 10,071 ERA5 events over 9,744 distinct
times -- 1.09 and 1.03 events per field. So the cache here is small on purpose:
it catches the consecutive duplicate and nothing more, and the cost that matters
is one four-level hemisphere read per event, which is 442 kB on the f09 grid and
234 kB on ERA5's.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import H_SCALE

#: Levels averaged for the two-dimensional summary, and the field the
#: classification reads. The record's own ``wavg_levels`` key carries the same
#: list; a store written with another set has to say so.
WAVG_LEVELS: tuple[int, ...] = (400, 300, 250, 200)


@dataclass
class _ArchiveConfig:
    """The little of a tendency configuration that opening an archive needs."""

    source: str
    data_dir: Path
    member: int | None = None
    engine: str = "netcdf4"
    rel_hours: tuple[int, ...] = (0,)


@dataclass
class HemisphereFields:
    """Reader for the hemisphere field the circumpolar classification contours.

    Attributes:
        source: ``"era5"`` or ``"cesm"``.
        data_dir: Root of the archive: the directory of monthly per-variable
            files for ERA5, or of the per-member-year files for CESM.
        wavg_levels: Levels to average, in hPa.
        cache_size: How many fields to keep. Small by design; see the module
            docstring.
        engine: netCDF engine to open with.
    """

    source: str
    data_dir: Path
    wavg_levels: tuple[int, ...] = WAVG_LEVELS
    cache_size: int = 8
    engine: str = "netcdf4"
    _fields: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _datasets: dict = field(default_factory=dict, repr=False)
    reads: int = 0

    def wavg_height(
        self, when: pd.Timestamp, member: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(field, lat, lon)`` for one time: the weighted column-mean height.

        Args:
            when: The event's own time. It must be a time the archive holds; a
                time it does not is an error rather than a field of missing
                values, because a classification that quietly saw nothing would
                label the event Neutral and look like a result.
            member: LENS2 member, required for the CESM ensemble and refused for
                ERA5, where it would silently do nothing.

        Returns:
            The field on the archive's own northern-hemisphere grid, latitude and
            longitude in the archive's own order.
        """
        when = pd.Timestamp(when)
        if self.source == "cesm" and member is None:
            raise ValueError("the CESM ensemble needs the member: two members share a time")
        if self.source != "cesm" and member is not None:
            raise ValueError(f"member={member} means nothing for source={self.source!r}")

        key = (member, when)
        cached = self._fields.get(key)
        if cached is not None:
            self._fields.move_to_end(key)
            return cached

        field_nh, lat, lon = self._read(when, member)
        self._fields[key] = (field_nh, lat, lon)
        while len(self._fields) > max(1, self.cache_size):
            self._fields.popitem(last=False)
        return field_nh, lat, lon

    def close(self) -> None:
        """Release the archive handles a long-lived worker has accumulated."""
        for ds in self._datasets.values():
            try:
                ds.close()
            except Exception:  # noqa: BLE001 - a closed handle must not stop the rest
                pass
        self._datasets.clear()
        self._fields.clear()

    # ---- internals ----------------------------------------------------

    def _dataset(self, when: pd.Timestamp, member: int | None):
        """The open archive covering this time, kept per file rather than per event."""
        from .tendency import open_source_ds

        key = (member, when.year, when.month if self.source != "cesm" else 0)
        ds = self._datasets.get(key)
        if ds is None:
            cfg = _ArchiveConfig(
                source=self.source, data_dir=Path(self.data_dir),
                member=member, engine=self.engine,
            )
            ds = open_source_ds(cfg, when, chunks={"valid_time": 1}, var_list=["z"])
            self._datasets[key] = ds
        return ds

    def _read(self, when: pd.Timestamp, member: int | None):
        from .tendency import _plev_name, z_divisor

        ds = self._dataset(when, member)
        plev = _plev_name(ds)
        times = pd.to_datetime(ds.valid_time.values)
        if when not in times:
            raise KeyError(
                f"{when} is not in the archive window that {self.source} opened for it "
                f"({times[0]} to {times[-1]}); the event sits outside the record"
            )
        levels = list(self.wavg_levels)
        missing = [p for p in levels if p not in set(np.asarray(ds[plev].values).tolist())]
        if missing:
            raise KeyError(f"the archive has no {missing} hPa level to average")
        heights = (
            ds["z"].sel(valid_time=when).sel({plev: levels}).values.astype(np.float64)
            / z_divisor(_ArchiveConfig(source=self.source, data_dir=Path(self.data_dir)))
        )
        self.reads += 1
        return (
            weighted_column_mean(heights),
            np.asarray(ds["latitude"].values, dtype=float),
            np.asarray(ds["longitude"].values, dtype=float),
        )


def weighted_column_mean(heights: np.ndarray) -> np.ndarray:
    """Average heights over their first axis, weighted by ``exp(-z/H)``.

    The weight is the height itself, so this is the same self-weighted mean the
    record's two-dimensional keys carry, and the sum is divided by the weight of
    the levels that are finite at each point -- never by the whole weight, which
    would pull a point with a missing level towards zero while keeping it finite.
    """
    heights = np.asarray(heights, dtype=np.float64)
    weight = np.exp(-heights / H_SCALE)
    valid = np.isfinite(heights) & np.isfinite(weight)
    w = np.where(valid, weight, 0.0)
    numerator = np.sum(np.where(valid, heights, 0.0) * w, axis=0)
    denominator = np.sum(w, axis=0)
    out = np.full(numerator.shape, np.nan)
    good = denominator > 0
    out[good] = numerator[good] / denominator[good]
    return out


def member_of(track_id) -> int | None:
    """The LENS2 member a CESM track identifier carries, or ``None`` for ERA5.

    CESM identifiers are ``m091_t00002`` because the tracking restarts per
    member; ERA5's are bare integers.
    """
    text = str(track_id)
    if text.startswith("m") and "_" in text:
        head = text.split("_", 1)[0]
        if head[1:].isdigit():
            return int(head[1:])
    return None
