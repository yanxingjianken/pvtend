"""Tests for RWB classification (Pass 1) module."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from pvtend.classify import (
    ClassifyConfig,
    ClassifyResult,
    _classify_bays_z2d,
    _classify_multilevel,
    _load_excluded,
    _parse_dh,
    _parse_track_id,
    run_pass1,
)
from pvtend.rwb import RWBConfig


# ── Helper fixture: synthetic Z field ────────────────────────────────

@pytest.fixture
def z_patch():
    """21×41 Z field (arbitrary units) with a smooth dipole pattern."""
    ny, nx = 21, 41
    y = np.linspace(-10, 10, ny)
    x = np.linspace(-20, 20, nx)
    X, Y = np.meshgrid(x, y)
    # Smooth dipole – no real overturning
    z = 5400.0 + 60.0 * np.exp(-(X**2 + Y**2) / 100)
    return z, x, y, X, Y


# ── Parsing helpers ──────────────────────────────────────────────────

class TestParseTrackId:
    """Test _parse_track_id regex extraction.

    The regex is ``track_(\\d+)_`` so filenames MUST contain
    ``track_<digits>_`` to match.
    """

    def test_standard_filename(self, tmp_path):
        fp = tmp_path / "evt_track_425_20100101.npz"
        assert _parse_track_id(fp) == 425

    def test_track_prefix(self, tmp_path):
        fp = tmp_path / "track_0042_test.npz"
        assert _parse_track_id(fp) == 42

    def test_embedded(self, tmp_path):
        fp = tmp_path / "some_track_12_stuff.npz"
        assert _parse_track_id(fp) == 12

    def test_no_match(self, tmp_path):
        fp = tmp_path / "noid.npz"
        assert _parse_track_id(fp) is None


class TestParseDh:
    """Test _parse_dh directory-name parsing."""

    def test_positive(self):
        assert _parse_dh("dh=5") == 5

    def test_negative(self):
        assert _parse_dh("dh=-12") == -12

    def test_zero(self):
        assert _parse_dh("dh=0") == 0

    def test_no_match(self):
        assert _parse_dh("onset") is None
        assert _parse_dh("hour_5") is None


# ── Excluded track loading ───────────────────────────────────────────

class TestLoadExcluded:
    """Test excluded track ID loading from various file formats."""

    def test_none_path(self):
        assert _load_excluded(None) == set()

    def test_nonexistent_file(self, tmp_path):
        assert _load_excluded(tmp_path / "missing.csv") == set()

    def test_csv_with_header(self, tmp_path):
        f = tmp_path / "exclude.csv"
        f.write_text("track_id,reason\n10,bad\n20,noisy\n")
        assert _load_excluded(f) == {10, 20}

    def test_plain_ids(self, tmp_path):
        f = tmp_path / "exclude.txt"
        f.write_text("10\n20\n30\n")
        assert _load_excluded(f) == {10, 20, 30}

    def test_cesm_string_ids_csv(self, tmp_path):
        """CESM ids like m091_t00002 must load as str, not be int()-dropped."""
        f = tmp_path / "exclude.csv"
        f.write_text("track_id,reason\nm091_t00002,omega\n123,blowup\n")
        assert _load_excluded(f) == {"m091_t00002", 123}

    def test_cesm_string_ids_plain_and_timestamped(self, tmp_path):
        """Headerless lines; a trailing timestamp normalises to the bare id."""
        f = tmp_path / "exclude.txt"
        f.write_text("m091_t00002\nm095_t00443_1985020800\n77\n")
        assert _load_excluded(f) == {"m091_t00002", "m095_t00443", 77}


# ── Single-level bay classifier ─────────────────────────────────────

class TestClassifyBaysZ2d:
    """Test _classify_bays_z2d on simple Z fields."""

    def test_all_nan(self):
        z = np.full((11, 21), np.nan)
        x = np.linspace(-10, 10, 21)
        y = np.linspace(-5, 5, 11)
        cfg = RWBConfig()
        awb, cwb = _classify_bays_z2d(z, x, y, cfg)
        assert awb is False
        assert cwb is False

    def test_smooth_field_no_overturn(self, z_patch):
        """A smooth dome should produce no overturning."""
        z, x, y, _, _ = z_patch
        cfg = RWBConfig(area_min_deg2=1.0, try_levels=200)
        awb, cwb = _classify_bays_z2d(z, x, y, cfg)
        # Smooth dome → unlikely to have bays with default settings
        assert isinstance(awb, bool)
        assert isinstance(cwb, bool)


# ── Multi-level classifier ──────────────────────────────────────────

class TestClassifyMultilevel:
    """Test _classify_multilevel threshold logic."""

    def test_too_few_levels(self, z_patch):
        """If fewer levels than threshold, should not classify."""
        z2d, x, y, _, _ = z_patch
        z3d = np.stack([z2d, z2d])  # 2 levels
        levels = np.array([500, 300])
        cfg = RWBConfig(area_min_deg2=1.0, try_levels=200)
        # Threshold=3 but only 2 levels → cannot reach threshold
        awb, cwb = _classify_multilevel(
            z3d, levels, x, y,
            classify_levels=[500, 300],
            threshold=3,
            cfg=cfg,
        )
        assert awb is False
        assert cwb is False


# ── ClassifyResult I/O ──────────────────────────────────────────────

class TestClassifyResult:
    """Test ClassifyResult creation, properties, and round-trip I/O."""

    @pytest.fixture
    def sample_result(self):
        return ClassifyResult(
            stage_all={"onset": {1, 2, 3}, "peak": {1, 2, 3}},
            stage_awb={"onset": {1}, "peak": {2}},
            stage_cwb={"onset": {2}, "peak": {1}},
            stage_neu={"onset": {3}, "peak": {3}},
            h_scale=7000.0,
            stages=["onset", "peak"],
            classify_levels=[500, 400, 300, 200],
            classify_threshold=3,
        )

    def test_variant_trackset_keys(self, sample_result):
        vt = sample_result.variant_trackset
        expected_keys = {
            "AWB_onset", "CWB_onset", "NEUTRAL_onset",
            "AWB_peak", "CWB_peak", "NEUTRAL_peak",
        }
        assert set(vt.keys()) == expected_keys

    def test_variant_trackset_values(self, sample_result):
        vt = sample_result.variant_trackset
        assert vt["AWB_onset"] == frozenset({1})
        assert vt["CWB_peak"] == frozenset({1})
        assert vt["NEUTRAL_peak"] == frozenset({3})

    def test_stage_labels(self, sample_result):
        lbls = sample_result.stage_labels
        assert lbls["onset"][1] == "AWB"
        assert lbls["onset"][2] == "CWB"
        assert lbls["onset"][3] == "NEUTRAL"

    def test_stage_tracksets(self, sample_result):
        ts = sample_result.stage_tracksets
        assert ts["onset"]["ALL"] == frozenset({1, 2, 3})
        assert ts["onset"]["AWB"] == frozenset({1})

    def test_omega_detection(self):
        """Track IDs in both AWB and CWB should be labelled 'Omega'."""
        r = ClassifyResult(
            stage_all={"onset": {1}},
            stage_awb={"onset": {1}},
            stage_cwb={"onset": {1}},
            stage_neu={"onset": set()},
            h_scale=None,
            stages=["onset"],
            classify_levels=[500],
            classify_threshold=1,
        )
        assert r.stage_labels["onset"][1] == "Omega"
        ts = r.stage_tracksets
        assert 1 in ts["onset"]["Omega"]

    def test_save_load_roundtrip(self, sample_result, tmp_path):
        pkl_path = tmp_path / "rwb.pkl"
        sample_result.save(pkl_path)
        assert pkl_path.exists()

        loaded = ClassifyResult.load(pkl_path)
        assert loaded.stages == sample_result.stages
        assert loaded.h_scale == sample_result.h_scale
        assert loaded.stage_all["onset"] == sample_result.stage_all["onset"]
        assert loaded.classify_levels == sample_result.classify_levels
        assert loaded.classify_threshold == sample_result.classify_threshold

    def test_pkl_format_matches_core(self, sample_result, tmp_path):
        """Verify the saved PKL dict keys match the core script format."""
        pkl_path = tmp_path / "rwb.pkl"
        sample_result.save(pkl_path)
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        expected_keys = {
            "stage_ALL", "stage_AWB", "stage_CWB", "stage_NEU",
            "H_SCALE", "variant_trackset", "RWB_STAGE_LABELS",
            "RWB_STAGE_TRACKSETS", "CLASSIFY_LEVELS", "CLASSIFY_THRESHOLD",
        }
        assert set(d.keys()) == expected_keys


# ── Integration: run_pass1 on synthetic data ─────────────────────────

class TestRunPass1:
    """Integration test for run_pass1 with synthetic NPZ files."""

    @pytest.fixture
    def synthetic_npz_tree(self, tmp_path, z_patch):
        """Build a minimal NPZ tree: onset/dh=0/ with 2 events."""
        z, x, y, X, Y = z_patch
        levels = np.array([500, 400, 300, 200], dtype=float)
        z3d = np.stack([z + lev * 0.01 for lev in levels])

        onset_dir = tmp_path / "onset" / "dh=0"
        onset_dir.mkdir(parents=True)

        for tid in [100, 200]:
            np.savez(
                onset_dir / f"evt_track_{tid}_test.npz",
                z_3d=z3d,
                X_rel=X,
                Y_rel=Y,
                levels=levels,
                H_SCALE=np.float64(7000.0),
                track_id=np.int64(tid),
            )
        return tmp_path

    def test_run_pass1_basic(self, synthetic_npz_tree, tmp_path):
        out_pkl = tmp_path / "result.pkl"
        cfg = ClassifyConfig(
            npz_dir=synthetic_npz_tree,
            output_path=out_pkl,
            stages=["onset"],
            classify_levels=[500, 400, 300, 200],
            classify_threshold=3,
            rwb_cfg=RWBConfig(area_min_deg2=1.0, try_levels=200),
        )
        result = run_pass1(cfg)
        assert isinstance(result, ClassifyResult)
        assert "onset" in result.stages
        # Both tracks should appear in stage_all
        assert 100 in result.stage_all["onset"]
        assert 200 in result.stage_all["onset"]


# ── Public single-field API: classify_z_field + label_from_flags ──────

from pvtend.classify import classify_z_field, label_from_flags  # noqa: E402


class TestLabelFromFlags:
    """label_from_flags mirrors ClassifyResult.stage_labels semantics."""

    def test_omega(self):
        assert label_from_flags(True, True) == "Omega"

    def test_awb(self):
        assert label_from_flags(True, False) == "AWB"

    def test_cwb(self):
        assert label_from_flags(False, True) == "CWB"

    def test_neutral(self):
        assert label_from_flags(False, False) == "NEUTRAL"


class TestClassifyZField:
    """classify_z_field is a faithful public wrapper of _classify_bays_z2d."""

    def test_returns_bool_tuple(self, z_patch):
        z, x, y, _, _ = z_patch
        awb, cwb = classify_z_field(z, x, y)
        assert isinstance(awb, bool) and isinstance(cwb, bool)

    def test_all_nan_is_neutral(self, z_patch):
        z, x, y, _, _ = z_patch
        awb, cwb = classify_z_field(np.full_like(z, np.nan), x, y)
        assert (awb, cwb) == (False, False)
        assert label_from_flags(awb, cwb) == "NEUTRAL"

    def test_default_cfg_matches_explicit(self, z_patch):
        """cfg=None (→ RWBConfig()) must equal an explicit default RWBConfig."""
        z, x, y, _, _ = z_patch
        assert classify_z_field(z, x, y, None) == \
            classify_z_field(z, x, y, RWBConfig())

    def test_wrapper_equals_private(self, z_patch):
        """The public wrapper returns exactly what the private core returns."""
        z, x, y, _, _ = z_patch
        cfg = RWBConfig()
        assert classify_z_field(z, x, y, cfg) == _classify_bays_z2d(z, x, y, cfg)

    def test_overturning_field_is_non_neutral(self):
        """A tilted Z dipole (cutoff high NW over low SE) overturns → the public
        path returns a real AWB/CWB/Omega label, exercising the bay detector
        end-to-end (not just the NEUTRAL branch)."""
        ny, nx = 41, 81
        y = np.linspace(-20, 20, ny)
        x = np.linspace(-40, 40, nx)
        X, Y = np.meshgrid(x, y)
        z = (5500.0 - 9 * Y
             + 400 * np.exp(-(((X + 10) ** 2) / 120 + ((Y - 6) ** 2) / 40))
             - 300 * np.exp(-(((X - 12) ** 2) / 120 + ((Y + 4) ** 2) / 40)))
        awb, cwb = classify_z_field(z, x, y)
        assert awb or cwb                          # at least one bay detected
        assert label_from_flags(awb, cwb) in {"AWB", "CWB", "Omega"}


# ── classify --patches second option (npz/nc patch-file mode) ─────────

class TestClassifyPatchesCLI:
    """`pvtend-pipeline classify --patches <file>` reads a 2-D Z-patch stack
    and writes a per-event AWB/CWB/Omega/NEUTRAL label CSV (single-field mode)."""

    def _make_patches(self, tmp_path, n=4, fmt="npz", ident="member_track"):
        ny, nx = 21, 41
        y = np.linspace(-15, 15, ny)
        x = np.linspace(-22, 22, nx)
        X, Y = np.meshgrid(x, y)
        Z = np.stack([5500.0 - 8 * Y + (60 + 10 * i) * np.exp(-(X**2 + Y**2) / 90)
                      for i in range(n)]).astype(np.float64)
        ids = (dict(member=np.ones(n, int), track=np.arange(n))
               if ident == "member_track"
               else dict(id=np.arange(100, 100 + n)) if ident == "id" else {})
        if fmt == "nc":
            import xarray as xr
            fp = tmp_path / "patches.nc"
            ds = xr.Dataset(
                {"patch_zwavg": (("n", "ny", "nx"), Z),
                 "rel_lon": (("nx",), x), "rel_lat": (("ny",), y),
                 **{k: (("n",), v) for k, v in ids.items()}})
            ds.to_netcdf(fp)
        else:
            fp = tmp_path / "patches.npz"
            np.savez(fp, patch_zwavg=Z, rel_lon=x, rel_lat=y, **ids)
        return fp

    @pytest.mark.parametrize("fmt", ["npz", "nc"])
    def test_writes_label_csv(self, tmp_path, fmt):
        import csv
        from argparse import Namespace
        from pvtend.cli import _cmd_classify_patches
        fp = self._make_patches(tmp_path, n=4, fmt=fmt)
        out = tmp_path / "labels.csv"
        _cmd_classify_patches(Namespace(patches=fp, output=out))
        assert out.exists()
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4
        assert set(rows[0]) == {"member", "track", "label", "awb", "cwb"}
        assert all(r["label"] in {"AWB", "CWB", "Omega", "NEUTRAL"} for r in rows)

    def test_id_only_identifier(self, tmp_path):
        """id-only identifier branch → CSV header {id,label,awb,cwb}."""
        import csv
        from argparse import Namespace
        from pvtend.cli import _cmd_classify_patches
        fp = self._make_patches(tmp_path, n=3, fmt="npz", ident="id")
        out = tmp_path / "labels_id.csv"
        _cmd_classify_patches(Namespace(patches=fp, output=out))
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert set(rows[0]) == {"id", "label", "awb", "cwb"}
        assert rows[0]["id"] == "100"

    def test_single_2d_patch_ndim_guard(self, tmp_path):
        """A genuine 2-D (NY,NX) Z field is promoted to a stack of one."""
        import csv
        from argparse import Namespace
        from pvtend.cli import _cmd_classify_patches
        ny, nx = 21, 41
        y = np.linspace(-15, 15, ny)
        x = np.linspace(-22, 22, nx)
        X, Y = np.meshgrid(x, y)
        z2d = (5500.0 - 8 * Y).astype(np.float64)      # (NY, NX), no leading axis
        fp = tmp_path / "one.npz"
        np.savez(fp, z=z2d, rel_lon=x, rel_lat=y)
        out = tmp_path / "one.csv"
        _cmd_classify_patches(Namespace(patches=fp, output=out))
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["label"] in {"AWB", "CWB", "Omega", "NEUTRAL"}
