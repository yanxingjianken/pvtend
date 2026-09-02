"""Tests for the input-grid profiles."""

from __future__ import annotations

import pytest

from pvtend.grid import ERA5_1P5_NH, CESM_F09, GRID_PROFILES


class TestGridProfile:
    def test_era5_default_isotropic(self):
        assert ERA5_1P5_NH.isotropic is True
        assert ERA5_1P5_NH.z_is_height is False     # ERA5 z is geopotential

    def test_cesm_f09_anisotropic(self):
        assert CESM_F09.isotropic is False          # 0.942 != 1.25
        assert CESM_F09.z_is_height is True         # CESM z is height [m]
        assert CESM_F09.nlat == 192 and CESM_F09.nlon == 288

    def test_registry(self):
        assert GRID_PROFILES["ERA5_1P5_NH"] is ERA5_1P5_NH
        assert GRID_PROFILES["CESM_F09"] is CESM_F09

    def test_frozen(self):
        with pytest.raises(Exception):
            CESM_F09.dlat = 1.0  # type: ignore[misc]
