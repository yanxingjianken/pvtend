"""Shared test fixtures for pvtend."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def small_grid():
    """Small lat/lon grid for unit tests."""
    nlat, nlon = 21, 41  # ~10°×20° at 0.5° spacing
    lat = np.linspace(30, 50, nlat)
    lon = np.linspace(-20, 20, nlon)
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]
    return {"lat": lat, "lon": lon, "dlat": dlat, "dlon": dlon,
            "nlat": nlat, "nlon": nlon}


@pytest.fixture
def pressure_levels():
    """Standard ERA5 pressure levels (subset for tests)."""
    return np.array([100, 200, 300, 500, 700, 850, 925, 1000], dtype=float) * 100  # Pa


@pytest.fixture
def synthetic_field(rng, small_grid):
    """A smooth synthetic 2D field."""
    lat, lon = np.meshgrid(small_grid["lat"], small_grid["lon"], indexing="ij")
    field = np.sin(np.deg2rad(lat) * 3) * np.cos(np.deg2rad(lon) * 2)
    field += rng.normal(0, 0.01, field.shape)
    return field
