"""The hemisphere field the circumpolar classification contours.

The load-bearing test is the first one: cropped to an event's patch, this field
must be the same quantity the record's own two-dimensional ``z`` holds. If it is
not, the contours would be drawn on a different field from the one the
classification is calibrated against.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvtend.hemisphere import (
    HemisphereFields,
    WAVG_LEVELS,
    member_of,
    weighted_column_mean,
)

ERA5_DIR = Path("/net/flood/data2/users/x_yan/era")
STORE = Path("/net/flood/data2/users/x_yan/pvtend/outputs/era5_blocking/peak/dh=+0")


class TestWeightedColumnMean:
    def test_a_uniform_column_is_its_own_mean(self):
        heights = np.full((4, 3, 5), 9000.0)
        np.testing.assert_allclose(weighted_column_mean(heights), 9000.0)

    def test_the_weight_favours_the_lower_level(self):
        # exp(-z/H) is larger where z is smaller, so the mean sits below the
        # midpoint of a two-level column.
        heights = np.array([[[7000.0]], [[11000.0]]])
        got = float(weighted_column_mean(heights)[0, 0])
        assert 7000.0 < got < 9000.0

    def test_a_missing_level_is_dropped_not_counted_as_zero(self):
        heights = np.array([[[8000.0]], [[np.nan]], [[10000.0]]])
        both = weighted_column_mean(np.array([[[8000.0]], [[10000.0]]]))
        np.testing.assert_allclose(weighted_column_mean(heights), both)

    def test_a_column_with_nothing_valid_is_missing(self):
        assert np.isnan(weighted_column_mean(np.full((2, 1, 1), np.nan))[0, 0])


class TestMemberOf:
    @pytest.mark.parametrize("track,expected", [
        ("m091_t00002", 91), ("m100_t01234", 100), (42, None), ("42", None),
        ("track_7", None),
    ])
    def test_the_member_comes_from_a_cesm_identifier_only(self, track, expected):
        assert member_of(track) == expected


class TestArchiveContract:
    def test_the_ensemble_needs_its_member(self):
        fields = HemisphereFields(source="cesm", data_dir=Path("/nowhere"))
        with pytest.raises(ValueError, match="needs the member"):
            fields.wavg_height(pd.Timestamp("2000-01-01"))

    def test_a_member_means_nothing_for_era5(self):
        fields = HemisphereFields(source="era5", data_dir=Path("/nowhere"))
        with pytest.raises(ValueError, match="means nothing"):
            fields.wavg_height(pd.Timestamp("2000-01-01"), member=91)


@pytest.mark.skipif(not STORE.is_dir() or not any(STORE.glob("track_*.npz")),
                    reason="the ERA5 store is not on this machine")
class TestAgainstTheStore:
    """The field, cropped to a patch, against the key the record already holds."""

    @staticmethod
    def _events(n):
        files = sorted(glob.glob(str(STORE / "track_*.npz")))
        return [files[0], files[len(files) // 2], files[-1]][:n]

    def test_the_cropped_field_is_the_stored_key(self):
        fields = HemisphereFields(source="era5", data_dir=ERA5_DIR)
        for path in self._events(3):
            with np.load(path, allow_pickle=False) as z:
                stored = np.asarray(z["z"], dtype=float)
                lat_vec = np.asarray(z["lat_vec"], dtype=float)
                lon_vec = np.asarray(z["lon_vec_unwrapped"], dtype=float)
                assert [int(v) for v in np.asarray(z["wavg_levels"])] == list(WAVG_LEVELS)
            stamp = Path(path).name.split("_")[-2]
            when = pd.Timestamp(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:00")
            field, lat, lon = fields.wavg_height(when)
            rows = np.array([int(np.abs(lat - la).argmin()) if np.isfinite(la) else -1
                             for la in lat_vec])
            cols = np.array([int(np.abs((lon - lo + 180.0) % 360.0 - 180.0).argmin())
                             for lo in lon_vec])
            have = rows >= 0
            cropped = np.full(stored.shape, np.nan)
            cropped[have, :] = field[rows[have][:, None], cols[None, :]]
            both = np.isfinite(stored) & np.isfinite(cropped)
            assert both.sum() > 0.5 * stored.size
            np.testing.assert_array_equal(cropped[both], stored[both])
        fields.close()

    def test_a_repeated_time_is_not_read_twice(self):
        fields = HemisphereFields(source="era5", data_dir=ERA5_DIR, cache_size=4)
        path = self._events(1)[0]
        stamp = Path(path).name.split("_")[-2]
        when = pd.Timestamp(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:00")
        first, _, _ = fields.wavg_height(when)
        assert fields.reads == 1
        again, _, _ = fields.wavg_height(when)
        assert fields.reads == 1
        assert again is first
        fields.close()

    def test_a_time_outside_the_record_is_an_error(self):
        fields = HemisphereFields(source="era5", data_dir=ERA5_DIR)
        with pytest.raises((KeyError, FileNotFoundError, ValueError)):
            fields.wavg_height(pd.Timestamp("1889-01-01 00:00"))
        fields.close()


class TestContoursIgnoreTheFill:
    """A missing region must not manufacture contours along its own edge.

    The tracer cannot take a mask, so the field is filled to run it and the
    vertices beside filled cells are dropped afterwards. Without that, the step
    around the fill is itself a contour and the overturning test reads the bays
    it makes as wave breaking.
    """

    @staticmethod
    def _ramp(nlat=41, nlon=81):
        y = np.linspace(30.0, -30.0, nlat)[:, None]
        x = np.linspace(-60.0, 60.0, nlon)[None, :]
        return 9000.0 + 8.0 * y + 0.0 * x, x.ravel(), y.ravel()

    def test_a_hole_produces_no_contour_of_its_own(self):
        from pvtend.rwb import sampled_longest_contours

        field, x, y = self._ramp()
        whole = sampled_longest_contours(field, x, y, try_levels=60, min_vertices=10)
        holed = field.copy()
        holed[2:9, 30:50] = np.nan          # a block the archive did not cover
        after = sampled_longest_contours(holed, x, y, try_levels=60, min_vertices=10)
        # Every surviving vertex is away from the hole, and no contour runs
        # along its edge: the rows the hole occupies keep no vertex inside it.
        for c in after:
            inside = (c["y"] <= y[2]) & (c["y"] >= y[8]) & (c["x"] >= x[30]) & (c["x"] <= x[49])
            assert not inside.any(), "a vertex survived inside the missing block"
        assert whole, "the intact field must produce contours at all"

    def test_a_field_with_no_hole_is_untouched(self):
        from pvtend.rwb import sampled_longest_contours

        field, x, y = self._ramp()
        a = sampled_longest_contours(field, x, y, try_levels=40, min_vertices=10)
        b = sampled_longest_contours(field.copy(), x, y, try_levels=40, min_vertices=10)
        assert len(a) == len(b)
        for ca, cb in zip(a, b):
            np.testing.assert_array_equal(ca["x"], cb["x"])
            np.testing.assert_array_equal(ca["y"], cb["y"])
