"""Tests for Helmholtz climatology compute/load round-trip."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pvtend.climatology import (
    compute_helmholtz_climatology,
    load_helmholtz_climatology,
)


class TestHelmholtzClimatologyRoundTrip:
    """Write synthetic clim files, compute Helmholtz, then load them."""

    @pytest.fixture
    def clim_env(self, tmp_path):
        """Create a tiny synthetic climatology for month=1 (jan) only."""
        nlat, nlon, nlev, nhour, nday = 21, 41, 3, 2, 2
        lat = np.linspace(30, 50, nlat)
        lon = np.linspace(-20, 20, nlon)
        rng = np.random.default_rng(7)

        stem = "test_clim"
        clim_dir = tmp_path / "clim"
        clim_dir.mkdir()
        out_dir = tmp_path / "helm_clim"

        # Write synthetic u / v clim NetCDFs for month 1 (jan)
        # Real files are 6-D: (month, day, hour, pressure_level, lat, lon)
        for var in ("u", "v"):
            data = rng.standard_normal(
                (1, nday, nhour, nlev, nlat, nlon)
            ).astype(np.float32)
            dims = ["month", "day", "hour", "pressure_level",
                    "latitude", "longitude"]
            ds = xr.Dataset({var: (dims, data)})
            ds.to_netcdf(clim_dir / f"{stem}_jan_{var}.nc")

        return {
            "clim_dir": clim_dir,
            "out_dir": out_dir,
            "lat": lat,
            "lon": lon,
            "stem": stem,
            "nhour": nhour,
            "nlev": nlev,
            "nlat": nlat,
            "nlon": nlon,
        }

    def test_compute_writes_files(self, clim_env):
        paths = compute_helmholtz_climatology(
            clim_dir=clim_env["clim_dir"],
            output_dir=clim_env["out_dir"],
            lat=clim_env["lat"],
            lon=clim_env["lon"],
            clim_stem=clim_env["stem"],
        )
        # Only month 1 exists → 2 files (u_helmholtz + v_helmholtz)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_load_returns_correct_shapes(self, clim_env):
        compute_helmholtz_climatology(
            clim_dir=clim_env["clim_dir"],
            output_dir=clim_env["out_dir"],
            lat=clim_env["lat"],
            lon=clim_env["lon"],
            clim_stem=clim_env["stem"],
        )
        result = load_helmholtz_climatology(
            clim_env["out_dir"], month=1, clim_stem=clim_env["stem"],
        )
        nday = 2  # synthetic fixture has nday=2
        nhour, nlev = clim_env["nhour"], clim_env["nlev"]
        nlat, nlon = clim_env["nlat"], clim_env["nlon"]
        for key in ("u_rot_bar", "u_div_bar", "v_rot_bar", "v_div_bar"):
            assert key in result
            assert result[key].shape == (nday, nhour, nlev, nlat, nlon)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Helmholtz climatology not found"):
            load_helmholtz_climatology(tmp_path, month=1, clim_stem="no_such")
