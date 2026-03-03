"""Tests for variant-aware composite builder (Pass 2) module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pvtend.classify import ClassifyResult
from pvtend.composite_builder import (
    CompositeConfig,
    CompositeResult,
    _accumulate,
    _levels_indexer,
    build_composites,
)


# ── Helper fixtures ──────────────────────────────────────────────────

@pytest.fixture
def levels():
    return np.array([500, 300, 200], dtype=float)


@pytest.fixture
def sample_classify_result():
    """Minimal ClassifyResult for testing composite variants."""
    return ClassifyResult(
        stage_all={"onset": {1, 2, 3}},
        stage_awb={"onset": {1}},
        stage_cwb={"onset": {2}},
        stage_neu={"onset": {3}},
        h_scale=7000.0,
        stages=["onset"],
        classify_levels=[500, 300, 200],
        classify_threshold=2,
    )


# ── _levels_indexer ──────────────────────────────────────────────────

class TestLevelsIndexer:
    """Test mapping of file levels to reference levels."""

    def test_identical_levels(self, levels):
        idx = _levels_indexer(levels, levels)
        np.testing.assert_array_equal(idx, np.arange(3))

    def test_subset_mapping(self):
        file_lev = np.array([100, 200, 300, 500, 700, 850], dtype=float)
        ref_lev = np.array([500, 300, 200], dtype=float)
        idx = _levels_indexer(file_lev, ref_lev)
        assert idx is not None
        np.testing.assert_array_equal(idx, [3, 2, 1])  # 500→3, 300→2, 200→1

    def test_missing_level_returns_none(self):
        file_lev = np.array([500, 300], dtype=float)
        ref_lev = np.array([500, 300, 200], dtype=float)
        idx = _levels_indexer(file_lev, ref_lev)
        assert idx is None


# ── _accumulate ──────────────────────────────────────────────────────

class TestAccumulate:
    """Test NaN-safe in-place accumulation."""

    def test_first_accumulation(self):
        sums, valids = {}, {}
        arr = np.array([1.0, 2.0, np.nan, 4.0])
        _accumulate(sums, valids, "test", arr)
        np.testing.assert_array_equal(sums["test"], [1.0, 2.0, 0.0, 4.0])
        np.testing.assert_array_equal(valids["test"], [1, 1, 0, 1])

    def test_second_accumulation(self):
        sums, valids = {}, {}
        arr1 = np.array([1.0, np.nan, 3.0])
        arr2 = np.array([np.nan, 2.0, 3.0])
        _accumulate(sums, valids, "x", arr1)
        _accumulate(sums, valids, "x", arr2)
        np.testing.assert_array_equal(sums["x"], [1.0, 2.0, 6.0])
        np.testing.assert_array_equal(valids["x"], [1, 1, 2])

    def test_all_nan(self):
        sums, valids = {}, {}
        arr = np.array([np.nan, np.nan])
        _accumulate(sums, valids, "n", arr)
        np.testing.assert_array_equal(sums["n"], [0.0, 0.0])
        np.testing.assert_array_equal(valids["n"], [0, 0])


# ── CompositeResult ──────────────────────────────────────────────────

class TestCompositeResult:
    """Test CompositeResult methods and I/O."""

    @pytest.fixture
    def sample_result(self, levels):
        """Construct a minimal CompositeResult with known data."""
        # 3 levels × 5 y × 7 x
        shape = (3, 5, 7)
        arr = np.ones(shape, dtype=np.float64) * 10.0
        valid = np.ones(shape, dtype=np.uint16) * 2

        return CompositeResult(
            levels=levels,
            x_rel=np.linspace(-3, 3, 7),
            y_rel=np.linspace(-2, 2, 5),
            h_scale=7000.0,
            stages=["onset"],
            fields_3d=["z_3d", "pv_3d"],
            sums={"onset": {0: {"z_3d": arr * 2, "pv_3d": arr}}},
            valids={"onset": {0: {"z_3d": valid, "pv_3d": valid}}},
            counts={"onset": {0: 2}},
            sums_v={},
            valids_v={},
            counts_v={},
            variant_names=["original"],
        )

    def test_mean_3d(self, sample_result):
        m = sample_result.mean_3d("z_3d", "onset", 0)
        assert m is not None
        # 20 / 2 = 10
        np.testing.assert_allclose(m, 10.0)

    def test_mean_3d_missing_field(self, sample_result):
        m = sample_result.mean_3d("missing", "onset", 0)
        assert m is None

    def test_mean_3d_missing_dh(self, sample_result):
        m = sample_result.mean_3d("z_3d", "onset", 99)
        assert m is None

    def test_reduce_2d_level_select(self, sample_result):
        out = sample_result.reduce_2d("z_3d", "onset", 0, level_mode=500)
        assert out is not None
        assert out.ndim == 2
        assert out.shape == (5, 7)

    def test_reduce_2d_3d_passthrough(self, sample_result):
        out = sample_result.reduce_2d("z_3d", "onset", 0, level_mode="3d")
        assert out is not None
        assert out.ndim == 3

    def test_reduce_2d_invalid_raises(self, sample_result):
        with pytest.raises(ValueError, match="Unsupported"):
            sample_result.reduce_2d("z_3d", "onset", 0, level_mode="bogus")

    def test_available_dh(self, sample_result):
        dhs = sample_result.available_dh("onset")
        assert dhs == [0]

    def test_save_load_roundtrip(self, sample_result, tmp_path):
        pkl_path = tmp_path / "composite.pkl"
        sample_result.save(pkl_path)
        assert pkl_path.exists()

        loaded = CompositeResult.load(pkl_path)
        np.testing.assert_array_equal(loaded.levels, sample_result.levels)
        assert loaded.stages == sample_result.stages
        assert loaded.fields_3d == sample_result.fields_3d

        # Verify data integrity
        m = loaded.mean_3d("z_3d", "onset", 0)
        np.testing.assert_allclose(m, 10.0)


# ── Integration: build_composites on synthetic NPZ ──────────────────

class TestBuildComposites:
    """Integration test for build_composites with synthetic NPZ files."""

    @pytest.fixture
    def npz_tree(self, tmp_path, levels):
        """Write synthetic NPZ files for 3 events in onset/dh=0."""
        shape = (len(levels), 5, 7)
        x_rel = np.linspace(-3, 3, 7)
        y_rel = np.linspace(-2, 2, 5)
        onset_dir = tmp_path / "onset" / "dh=0"
        onset_dir.mkdir(parents=True)

        for tid in [1, 2, 3]:
            data = {
                "z_3d": np.full(shape, tid * 10.0),
                "pv_3d": np.full(shape, tid * 1.0),
                "X_rel": np.tile(x_rel, (5, 1)),
                "Y_rel": np.tile(y_rel, (7, 1)).T,
                "levels": levels,
                "H_SCALE": np.float64(7000.0),
                "track_id": np.int64(tid),
            }
            np.savez(onset_dir / f"evt_track_{tid}_test.npz", **data)
        return tmp_path

    def test_build_without_rwb(self, npz_tree):
        """Build composites without RWB variant information."""
        cfg = CompositeConfig(npz_dir=npz_tree, stages=["onset"])
        result = build_composites(cfg, rwb=None)
        assert isinstance(result, CompositeResult)
        assert result.counts["onset"][0] == 3

        # Mean z should be (10 + 20 + 30) / 3 = 20
        m = result.mean_3d("z_3d", "onset", 0)
        assert m is not None
        np.testing.assert_allclose(m, 20.0)

    def test_build_with_rwb(self, npz_tree, sample_classify_result):
        """Build composites WITH RWB variant stratification."""
        cfg = CompositeConfig(npz_dir=npz_tree, stages=["onset"])
        result = build_composites(cfg, rwb=sample_classify_result)
        assert isinstance(result, CompositeResult)

        # Overall "original" should still have 3 events
        assert result.counts["onset"][0] == 3

        # AWB_onset should only have track 1
        m_awb = result.mean_3d("z_3d", "onset", 0, variant="AWB_onset")
        if m_awb is not None:
            np.testing.assert_allclose(m_awb, 10.0)

    def test_exclude_file(self, npz_tree, tmp_path):
        """Excluded tracks should be skipped."""
        exclude = tmp_path / "exclude.txt"
        exclude.write_text("1\n")
        cfg = CompositeConfig(
            npz_dir=npz_tree,
            stages=["onset"],
            exclude_file=exclude,
        )
        result = build_composites(cfg, rwb=None)
        assert result.counts["onset"][0] == 2  # only tracks 2,3

        # Mean z should be (20 + 30) / 2 = 25
        m = result.mean_3d("z_3d", "onset", 0)
        np.testing.assert_allclose(m, 25.0)
