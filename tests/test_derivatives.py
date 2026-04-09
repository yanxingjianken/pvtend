"""Tests for derivative operators."""

from __future__ import annotations

import numpy as np
import pytest

from pvtend.derivatives import ddx, ddy, ddp, ddt, compute_dx_arr, compute_dy, ddx_dx, ddy_dy, ddx_dy
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
        """ddt of linearly increasing field — 2nd order everywhere."""
        nt = 5
        dt = 3600.0  # 1 hour
        f = np.arange(nt, dtype=float)[:, None, None] * np.ones((nt, 4, 4))
        result = ddt(f, dt)
        expected = np.ones_like(f) / dt
        # All points (including boundaries) should be exact for linear f
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestComputeDxArr:
    """Test the compute_dx_arr helper."""

    def test_equator(self):
        """At equator, dx = deg2rad(dlon) * R_EARTH."""
        y_rel = np.array([-1.5, 0.0, 1.5])
        dx = compute_dx_arr(1.5, center_lat=0.0, y_rel=y_rel)
        # At center row (equator), cos(0) = 1
        expected_center = np.deg2rad(1.5) * R_EARTH
        np.testing.assert_allclose(dx[1], expected_center, rtol=1e-10)

    def test_60N(self):
        """At 60°N, dx should be halved vs equator."""
        y_rel = np.array([0.0])
        dx_eq = compute_dx_arr(1.5, center_lat=0.0, y_rel=y_rel)
        dx_60 = compute_dx_arr(1.5, center_lat=60.0, y_rel=y_rel)
        np.testing.assert_allclose(dx_60[0] / dx_eq[0], 0.5, rtol=1e-10)

    def test_shape(self):
        """Output shape matches y_rel."""
        y_rel = np.linspace(-15, 15, 21)
        dx = compute_dx_arr(1.5, center_lat=55.0, y_rel=y_rel)
        assert dx.shape == (21,)


class TestComputeDy:
    """Test the compute_dy helper."""

    def test_value(self):
        """1.5° → ~167 km."""
        dy = compute_dy(1.5)
        np.testing.assert_allclose(dy, np.deg2rad(1.5) * R_EARTH, rtol=1e-10)


class TestDdxDx:
    """Test second zonal derivative."""

    def test_quadratic(self, small_grid):
        """ddx_dx of x² ≈ 2 at interior points."""
        lat = small_grid["lat"]
        lon = small_grid["lon"]
        dlon_rad = np.deg2rad(small_grid["dlon"])
        dx_arr = R_EARTH * np.cos(np.deg2rad(lat)) * dlon_rad

        _, lon2d = np.meshgrid(lat, lon, indexing="ij")
        # f(x) = (lon_rad)² — physical distance
        lon_rad = np.deg2rad(lon2d)
        cos_lat = np.cos(np.deg2rad(lat))[:, None]
        x_m = lon_rad * R_EARTH * cos_lat
        f = x_m ** 2

        result = ddx_dx(f, dx_arr, periodic=False)
        # d²(x²)/dx² = 2 everywhere
        # Only check far interior (2nd-order boundary effects)
        np.testing.assert_allclose(result[3:-3, 3:-3], 2.0, atol=0.5)


class TestDdyDy:
    """Test second meridional derivative."""

    def test_quadratic(self, small_grid):
        """ddy_dy of y² ≈ 2 at interior."""
        lat = small_grid["lat"]
        lon = small_grid["lon"]
        dy = R_EARTH * np.deg2rad(small_grid["dlat"])

        lat2d, _ = np.meshgrid(lat, lon, indexing="ij")
        y_m = np.deg2rad(lat2d) * R_EARTH
        f = y_m ** 2

        result = ddy_dy(f, dy)
        np.testing.assert_allclose(result[2:-2, :], 2.0, atol=0.5)


class TestDdxDy:
    """Test mixed second derivative."""

    def test_product(self, small_grid):
        """ddx_dy of (x * y) ≈ 1 at interior."""
        lat = small_grid["lat"]
        lon = small_grid["lon"]
        dlon_rad = np.deg2rad(small_grid["dlon"])
        dx_arr = R_EARTH * np.cos(np.deg2rad(lat)) * dlon_rad
        dy = R_EARTH * np.deg2rad(small_grid["dlat"])

        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        cos_lat = np.cos(np.deg2rad(lat))[:, None]
        x_m = np.deg2rad(lon2d) * R_EARTH * cos_lat
        y_m = np.deg2rad(lat2d) * R_EARTH
        f = x_m * y_m

        result = ddx_dy(f, dx_arr, dy, periodic=False)
        # d²(xy)/dxdy = 1
        np.testing.assert_allclose(result[3:-3, 3:-3], 1.0, atol=0.5)
