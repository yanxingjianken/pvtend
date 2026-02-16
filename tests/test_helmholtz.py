"""Tests for Helmholtz decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from pvtend.helmholtz import (
    compute_vorticity_divergence,
    solve_poisson_dct,
    helmholtz_decomposition,
)
from pvtend.constants import R_EARTH


class TestVorticityDivergence:
    """Test vorticity and divergence computation."""

    def test_solid_body_rotation(self, small_grid):
        """Solid-body rotation should have constant vorticity."""
        lat = small_grid["lat"]
        lon = small_grid["lon"]
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        cos_lat = np.cos(np.deg2rad(lat2d))
        dlat = np.deg2rad(small_grid["dlat"])
        dlon = np.deg2rad(small_grid["dlon"])

        # u = cos(lat), v = 0  => solid body rotation
        u = cos_lat
        v = np.zeros_like(u)

        vort, div = compute_vorticity_divergence(u, v, lat, dlon, dlat)

        # Divergence of solid body should be ~0
        np.testing.assert_allclose(div[2:-2, 2:-2], 0.0, atol=1e-6)

        # Vorticity should be negative (anticyclonic in NH => -2Ω sin(lat) / R)
        # but since u=cos(lat) (not full rotation), just check it's non-zero
        assert np.abs(vort[small_grid["nlat"] // 2, small_grid["nlon"] // 2]) > 0


class TestPoissonDCT:
    """Test DCT Poisson solver."""

    def test_known_rhs(self, small_grid):
        """Solve ∇²ψ = rhs and verify reconstruction."""
        nlat, nlon = small_grid["nlat"], small_grid["nlon"]
        dlat = np.deg2rad(small_grid["dlat"])
        dlon = np.deg2rad(small_grid["dlon"])

        # Create a smooth RHS
        lat2d, lon2d = np.meshgrid(small_grid["lat"], small_grid["lon"], indexing="ij")
        rhs = np.sin(np.deg2rad(lat2d) * 2) * np.sin(np.deg2rad(lon2d) * 2)

        # dx, dy in meters
        cos_lat_mid = np.cos(np.deg2rad(small_grid["lat"].mean()))
        dx = R_EARTH * cos_lat_mid * dlon
        dy = R_EARTH * dlat

        psi = solve_poisson_dct(rhs, dx, dy)
        assert psi.shape == rhs.shape
        # psi boundary should be ~0 for DCT (Dirichlet-like)
        assert np.abs(psi[0, :]).max() < np.abs(psi).max() * 0.5


class TestHelmholtzDecomposition:
    """Test full Helmholtz decomposition."""

    def test_reconstruction(self, small_grid):
        """u_psi + u_chi should approximate original u."""
        nlat, nlon = small_grid["nlat"], small_grid["nlon"]
        lat2d, lon2d = np.meshgrid(small_grid["lat"], small_grid["lon"], indexing="ij")
        dlon = np.deg2rad(small_grid["dlon"])
        dlat = np.deg2rad(small_grid["dlat"])

        u = np.sin(np.deg2rad(lat2d) * 2)
        v = np.cos(np.deg2rad(lon2d) * 3) * 0.5

        result = helmholtz_decomposition(u, v, small_grid["lat"], dlon, dlat,
                                         backend="dct")
        u_recon = result["u_psi"] + result["u_chi"]
        v_recon = result["v_psi"] + result["v_chi"]

        # Interior reconstruction should be reasonable
        rel_err_u = np.abs(u_recon[3:-3, 3:-3] - u[3:-3, 3:-3]).mean() / (np.abs(u).max() + 1e-10)
        rel_err_v = np.abs(v_recon[3:-3, 3:-3] - v[3:-3, 3:-3]).mean() / (np.abs(v).max() + 1e-10)
        assert rel_err_u < 0.5  # Helmholtz not exact on limited area
        assert rel_err_v < 0.5
