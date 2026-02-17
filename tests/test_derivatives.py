"""Tests for derivative operators."""

from __future__ import annotations

import numpy as np
import pytest

from pvtend.derivatives import ddx, ddy, ddp, ddt
from pvtend.constants import R_EARTH


class TestDdx:
    """Test zonal derivative with periodic boundary."""

    def test_sine_wave(self, small_grid):
        """ddx of sin(2*lon) should approximate 2*cos(2*lon)."""
        lat = small_grid["lat"]
        lon = small_grid["lon"]
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        dlon_rad = np.deg2rad(small_grid["dlon"])
        cos_lat = np.cos(np.deg2rad(lat2d))

        # dx_arr: zonal grid spacing per latitude row [m]
        dx_arr = R_EARTH * np.cos(np.deg2rad(lat)) * dlon_rad  # (nlat,)

        k = 2
        f = np.sin(np.deg2rad(lon2d) * k)
        expected = k * np.cos(np.deg2rad(lon2d) * k) / (R_EARTH * cos_lat)

        dfdx = ddx(f, dx_arr, periodic=False)
        # Exclude boundary rows AND columns (non-periodic limited-area grid)
        np.testing.assert_allclose(dfdx[1:-1, 1:-1], expected[1:-1, 1:-1],
                                   atol=0.1 * np.abs(expected).max())

    def test_constant_field(self, small_grid):
        """ddx of a constant should be zero."""
        f = np.ones((small_grid["nlat"], small_grid["nlon"])) * 42.0
        dlon_rad = np.deg2rad(small_grid["dlon"])
        dx_arr = R_EARTH * np.cos(np.deg2rad(small_grid["lat"])) * dlon_rad
        result = ddx(f, dx_arr)
        np.testing.assert_allclose(result, 0.0, atol=1e-12)


class TestDdy:
    """Test meridional derivative."""

    def test_linear_in_lat(self, small_grid):
        """ddy of a linearly varying field in lat."""
        lat = small_grid["lat"]
        lon = small_grid["lon"]
        lat2d, _ = np.meshgrid(lat, lon, indexing="ij")
        dlat_rad = np.deg2rad(small_grid["dlat"])
        dy = R_EARTH * dlat_rad  # meridional spacing in metres

        f = np.deg2rad(lat2d)  # f = lat_rad
        expected = np.ones_like(f) / R_EARTH

        dfdy = ddy(f, dy)
        # Interior points (not boundaries)
        np.testing.assert_allclose(dfdy[2:-2, :], expected[2:-2, :],
                                   rtol=0.05)


class TestDdp:
    """Test pressure derivative."""

    def test_linear_in_p(self, pressure_levels):
        """ddp of a linear function in p."""
        nlev = len(pressure_levels)
        f = np.tile(pressure_levels[:, None, None], (1, 5, 5))
        result = ddp(f, pressure_levels)
        # d(p)/dp = 1 everywhere
        np.testing.assert_allclose(result[1:-1], 1.0, rtol=0.01)


class TestDdt:
    """Test time derivative."""

    def test_linear_in_time(self):
        """ddt of linearly increasing field."""
        nt = 5
        dt = 3600.0  # 1 hour
        f = np.arange(nt, dtype=float)[:, None, None] * np.ones((nt, 4, 4))
        result = ddt(f, dt)
        expected = np.ones_like(f) / dt
        np.testing.assert_allclose(result[1:-1], expected[1:-1], rtol=1e-10)
