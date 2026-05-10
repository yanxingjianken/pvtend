"""Tests for pvtend.blowup.scan_omega_blowup (hard ±5 Pa/s cutoff)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pvtend.blowup import scan_omega_blowup


def _make_synthetic_omega(
    path: Path,
    times: pd.DatetimeIndex,
    levels_hpa: list[int] = [200, 300, 500, 700, 850],
    nlat: int = 8,
    nlon: int = 16,
    blowup_idx: list[int] | None = None,
    blowup_value: float = 7.5,
) -> Path:
    """Write a tiny ERA5-shaped ω file with optional blowup spikes."""
    rng = np.random.default_rng(0)
    nt, nlev = len(times), len(levels_hpa)
    w = rng.normal(0.0, 0.2, size=(nt, nlev, nlat, nlon)).astype("float32")
    if blowup_idx:
        lev_idx = levels_hpa.index(300)
        for t in blowup_idx:
            w[t, lev_idx, nlat // 2, nlon // 2] = blowup_value
    ds = xr.Dataset(
        {"w": (("valid_time", "pressure_level", "latitude", "longitude"), w)},
        coords={
            "valid_time": times,
            "pressure_level": np.array(levels_hpa, dtype="int32"),
            "latitude": np.linspace(90, -90, nlat, dtype="float32"),
            "longitude": np.linspace(0, 360, nlon, endpoint=False, dtype="float32"),
        },
    )
    ds.to_netcdf(path)
    return path


def test_scan_flags_only_exceedances(tmp_path: Path) -> None:
    times = pd.date_range("2010-01-01", periods=10, freq="h")
    nc = _make_synthetic_omega(
        tmp_path / "era5_w_test.nc",
        times,
        blowup_idx=[3, 7],
        blowup_value=7.5,
    )
    out_csv = tmp_path / "blowups.csv"
    df = scan_omega_blowup(
        era5_w_glob=str(nc),
        level_pa=30000.0,
        threshold=5.0,
        out_csv=out_csv,
    )
    assert len(df) == 2
    assert set(df["timestamp"]) == {times[3], times[7]}
    assert (df["max_abs_omega"] > 5.0).all()
    assert (df["exceedance_ratio"] > 1.0).all()
    assert out_csv.exists()


def test_scan_returns_empty_when_no_blowup(tmp_path: Path) -> None:
    times = pd.date_range("2010-01-01", periods=5, freq="h")
    nc = _make_synthetic_omega(tmp_path / "era5_w_quiet.nc", times)
    df = scan_omega_blowup(era5_w_glob=str(nc), threshold=5.0)
    assert df.empty


def test_scan_threshold_is_pa_per_s_not_sigma(tmp_path: Path) -> None:
    """A spike at 4 Pa/s must NOT be flagged when threshold=5 Pa/s."""
    times = pd.date_range("2010-01-01", periods=5, freq="h")
    nc = _make_synthetic_omega(
        tmp_path / "era5_w_marginal.nc",
        times,
        blowup_idx=[2],
        blowup_value=4.0,
    )
    df = scan_omega_blowup(era5_w_glob=str(nc), threshold=5.0)
    assert df.empty
