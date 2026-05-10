"""Tests for spherical-harmonic Helmholtz operators (sh_ops)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("spharm")

from pvtend.sh_ops import (
    gradient_sh,
    laplacian_sh,
    invert_laplacian_sh,
    helmholtz_sh,
    second_derivs_sh,
    parity_mirror,
    is_nh_grid,
)


def _make_grid(nlat=73, nlon=144):
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    return lat, lon


def test_gradient_constant_is_zero():
    lat, lon = _make_grid()
    f = np.ones((lat.size, lon.size), dtype=np.float64) * 3.14
    fx, fy = gradient_sh(f, lat, lon)
    assert np.max(np.abs(fx)) < 1e-6
    assert np.max(np.abs(fy)) < 1e-6


def test_laplacian_inverse_roundtrip_band_limited():
    """∇²(∇⁻²f) ≈ f for a field projected into pyspharm's band-limited
    regular-grid representation."""
    lat, lon = _make_grid()
    LAT, LON = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    f0 = (np.sin(LAT) * np.cos(LON)
          + np.sin(2 * LAT) * np.cos(2 * LON))
    # Project into band-limited representation via Lap(InvLap(f0))
    # which forces f into the spectrally representable subspace.
    f = laplacian_sh(invert_laplacian_sh(f0, lat, lon), lat, lon)
    f -= f.mean()

    chi = invert_laplacian_sh(f, lat, lon)
    f_back = laplacian_sh(chi, lat, lon)
    err = np.max(np.abs(f_back - f)) / (np.max(np.abs(f)) + 1e-30)
    assert err < 5e-2, f"roundtrip rel err {err}"


def test_helmholtz_pure_rotational():
    """If u, v are pure rotational, divergent component ≈ 0."""
    lat, lon = _make_grid()
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    # Pure solid-body rotation: u = U cos(lat), v = 0
    u = 10.0 * np.cos(np.deg2rad(LAT))
    v = np.zeros_like(u)
    out = helmholtz_sh(u, v, lat, lon)
    # Mask pole rows where pyspharm zeros velocity
    pole_mask = np.abs(lat) < 89.5
    u_div = out["u_div"][pole_mask]
    v_div = out["v_div"][pole_mask]
    rms = np.sqrt(np.mean(u_div**2 + v_div**2))
    rms_total = np.sqrt(np.mean(u[pole_mask]**2 + v[pole_mask]**2))
    assert rms / (rms_total + 1e-30) < 5e-2


def test_second_derivs_finite():
    lat, lon = _make_grid(nlat=37, nlon=72)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    f = np.sin(np.deg2rad(LAT)) * np.cos(np.deg2rad(LON))
    fxx, fxy, fyy = second_derivs_sh(f, lat, lon)
    assert np.all(np.isfinite(fxx))
    assert np.all(np.isfinite(fxy))
    assert np.all(np.isfinite(fyy))


def test_parity_mirror_scalar_even():
    lat_nh = np.linspace(0, 90, 37)
    lon = np.linspace(0, 360, 72, endpoint=False)
    f_nh = np.ones((lat_nh.size, lon.size))
    f_glob, lat_glob = parity_mirror(f_nh, lat_nh, kind="scalar")
    assert lat_glob.size == 2 * lat_nh.size - 1
    assert np.allclose(lat_glob, np.linspace(-90, 90, lat_glob.size))
    # Even reflection
    nlat_g = lat_glob.size
    eq = nlat_g // 2
    for j in range(1, eq + 1):
        assert np.allclose(f_glob[eq - j], f_glob[eq + j])


def test_parity_mirror_v_odd_zero_at_equator():
    lat_nh = np.linspace(0, 90, 37)
    lon = np.linspace(0, 360, 72, endpoint=False)
    rng = np.random.default_rng(0)
    v_nh = rng.standard_normal((lat_nh.size, lon.size))
    v_glob, lat_glob = parity_mirror(v_nh, lat_nh, kind="v")
    eq = lat_glob.size // 2
    assert np.allclose(v_glob[eq], 0.0)
    # Odd reflection
    for j in range(1, eq + 1):
        assert np.allclose(v_glob[eq - j], -v_glob[eq + j])


def test_is_nh_grid():
    lat_nh = np.linspace(0, 90, 37)
    lat_global = np.linspace(-90, 90, 73)
    assert is_nh_grid(lat_nh)
    assert not is_nh_grid(lat_global)
