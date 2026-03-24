"""Tests for the tendency computation pipeline module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvtend.tendency import (
    TendencyConfig,
    load_climatology,
    month_keys_for_window,
)


# ── TendencyConfig defaults ─────────────────────────────────────────

class TestTendencyConfig:
    """Test TendencyConfig dataclass defaults and overrides."""

    def test_default_event_type(self):
        cfg = TendencyConfig()
        assert cfg.event_type == "blocking"

    def test_default_qg_method(self):
        cfg = TendencyConfig()
        assert cfg.qg_omega_method == "log20"

    def test_prp_config(self):
        cfg = TendencyConfig(event_type="prp", qg_omega_method="sp19")
        assert cfg.event_type == "prp"
        assert cfg.qg_omega_method == "sp19"

    def test_default_levels(self):
        cfg = TendencyConfig()
        assert 500 in cfg.levels
        assert 200 in cfg.levels

    def test_rel_hours_range(self):
        cfg = TendencyConfig(rel_hours=list(range(-25, 25)))
        assert len(cfg.rel_hours) == 50
        assert cfg.rel_hours[0] == -25
        assert cfg.rel_hours[-1] == 24

    def test_wavg_levels_subset(self):
        cfg = TendencyConfig()
        # wavg_levels should be a subset of levels
        for lev in cfg.wavg_levels:
            assert lev in cfg.levels

    def test_center_mode(self):
        cfg = TendencyConfig(center_mode="lagrangian")
        assert cfg.center_mode == "lagrangian"

    def test_skip_existing(self):
        cfg = TendencyConfig(skip_existing=True)
        assert cfg.skip_existing is True


# ── month_keys_for_window ────────────────────────────────────────────

class TestMonthKeysForWindow:
    """Test (year, month) computation for event time windows."""

    def test_single_month(self):
        ts = pd.Timestamp("2010-06-15")
        keys = month_keys_for_window(ts, hmin=-12, hmax=12)
        assert len(keys) == 1
        assert keys[0] == (2010, 6)

    def test_cross_month_boundary(self):
        ts = pd.Timestamp("2010-01-01 06:00")
        keys = month_keys_for_window(ts, hmin=-12, hmax=12)
        # -12h from Jan 1 06:00 → Dec 31 18:00 (2009,12)
        # +12h from Jan 1 06:00 → Jan 1 18:00  (2010,1)
        assert (2009, 12) in keys
        assert (2010, 1) in keys

    def test_three_months(self):
        ts = pd.Timestamp("2010-02-01 00:00")
        keys = month_keys_for_window(ts, hmin=-49, hmax=24)
        # -49h → Jan 30; +24h → Feb 2 → should cover Jan and Feb
        months = {m for _, m in keys}
        assert 1 in months
        assert 2 in months

    def test_default_blocking_range(self):
        ts = pd.Timestamp("2010-07-15 12:00")
        keys = month_keys_for_window(ts)
        assert len(keys) >= 1
        assert all(isinstance(k, tuple) and len(k) == 2 for k in keys)


# ── load_climatology ────────────────────────────────────────────────

class TestLoadClimatology:
    """Test climatology auto-detection logic."""


# ── Cross-term catalog completeness (v2.0, 53 terms) ────────────────

# 52 named cross-term NPZ keys + Q (stored as a field) = 53 budget terms.
EXPECTED_CROSS_TERM_KEYS = {
    # 12 base (bar/anom × bar/anom)
    "u_anom_pv_bar_dx", "u_anom_pv_anom_dx",
    "u_bar_pv_anom_dx", "u_bar_pv_bar_dx",
    "v_anom_pv_bar_dy", "v_anom_pv_anom_dy",
    "v_bar_pv_anom_dy", "v_bar_pv_bar_dy",
    "w_anom_pv_bar_dp", "w_anom_pv_anom_dp",
    "w_bar_pv_anom_dp", "w_bar_pv_bar_dp",
    # 16 Helmholtz (anom + bar rot/div)
    "u_anom_rot_pv_bar_dx", "u_anom_rot_pv_anom_dx",
    "u_anom_div_pv_bar_dx", "u_anom_div_pv_anom_dx",
    "u_rot_bar_pv_bar_dx", "u_rot_bar_pv_anom_dx",
    "u_div_bar_pv_bar_dx", "u_div_bar_pv_anom_dx",
    "v_anom_rot_pv_bar_dy", "v_anom_rot_pv_anom_dy",
    "v_anom_div_pv_bar_dy", "v_anom_div_pv_anom_dy",
    "v_rot_bar_pv_bar_dy", "v_rot_bar_pv_anom_dy",
    "v_div_bar_pv_bar_dy", "v_div_bar_pv_anom_dy",
    # 16 divergent dry/moist horizontal
    "u_div_moist_pv_bar_dx", "u_div_moist_pv_anom_dx",
    "u_div_dry_pv_bar_dx", "u_div_dry_pv_anom_dx",
    "v_div_moist_pv_bar_dy", "v_div_moist_pv_anom_dy",
    "v_div_dry_pv_bar_dy", "v_div_dry_pv_anom_dy",
    "u_div_qg_moist_pv_bar_dx", "u_div_qg_moist_pv_anom_dx",
    "v_div_qg_moist_pv_bar_dy", "v_div_qg_moist_pv_anom_dy",
    "u_div_emanuel_moist_pv_bar_dx", "u_div_emanuel_moist_pv_anom_dx",
    "v_div_emanuel_moist_pv_bar_dy", "v_div_emanuel_moist_pv_anom_dy",
    # 8 alt vertical
    "w_dry_pv_bar_dp", "w_dry_pv_anom_dp",
    "w_moist_pv_bar_dp", "w_moist_pv_anom_dp",
    "w_qg_moist_pv_bar_dp", "w_qg_moist_pv_anom_dp",
    "w_emanuel_moist_pv_bar_dp", "w_emanuel_moist_pv_anom_dp",
}


class TestCrossTermCatalog:
    """Verify the 53-term v2.0 cross-term catalog is complete."""

    def test_expected_count(self):
        """52 named cross-term keys + Q (stored as field) = 53."""
        assert len(EXPECTED_CROSS_TERM_KEYS) == 52

    def test_no_duplicates(self):
        """All keys are unique (set length matches list-of-elements length)."""
        keys_list = list(EXPECTED_CROSS_TERM_KEYS)
        assert len(keys_list) == len(set(keys_list))

    def test_base_terms_present(self):
        base_u = {k for k in EXPECTED_CROSS_TERM_KEYS if k.startswith("u_anom_pv") or k.startswith("u_bar_pv")}
        assert len(base_u) == 4

    def test_helmholtz_bar_terms_present(self):
        bar_helm = {k for k in EXPECTED_CROSS_TERM_KEYS
                    if "rot_bar" in k or "div_bar" in k}
        assert len(bar_helm) == 8  # 4 u-dir + 4 v-dir

    def test_no_harmonic_cross_terms(self):
        """Harmonic absorbed into residual — no u_har/v_har cross-terms."""
        har_terms = {k for k in EXPECTED_CROSS_TERM_KEYS if "_har_" in k}
        assert len(har_terms) == 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Climatology missing"):
            load_climatology(tmp_path / "nonexistent.nc")

    def test_missing_parent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_climatology(tmp_path / "no_such_dir" / "clim.nc")
