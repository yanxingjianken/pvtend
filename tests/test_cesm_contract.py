"""CESM reader-boundary contract: units, member matching, missing variables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pvtend.tendency import cesm_to_pipeline_names, open_cesm_years_ds
from pvtend.cli import _assert_member_matches_events


def _tiny_cam_ds(pv_units: str = "PVU", drop: tuple[str, ...] = ()) -> xr.Dataset:
    data = {}
    dims = ("time", "plev", "lat", "lon")
    shape = (2, 3, 4, 5)
    for name in ("U", "V", "OMEGA", "PV", "Z3", "T", "Q"):
        if name in drop:
            continue
        da = xr.DataArray(np.full(shape, 2.0, dtype=np.float32), dims=dims)
        if name == "PV":
            da.attrs["units"] = pv_units
        data[name] = da
    return xr.Dataset(
        data,
        coords={
            "time": pd.date_range("1990-01-01", periods=2, freq="6h"),
            "plev": [1000.0, 500.0, 100.0],
            "lat": np.linspace(0.5, 90.0, 4),
            "lon": np.linspace(0.0, 288.0, 5, endpoint=False),
        },
    )


class TestPvUnitsConversion:
    """PVU archives are converted to SI at the reader boundary, exactly once."""

    def test_pvu_converts_to_si(self):
        out = cesm_to_pipeline_names(_tiny_cam_ds(pv_units="PVU"))
        np.testing.assert_allclose(out["pv"].values, 2.0e-6, rtol=1e-6)
        assert out["pv"].attrs["units"] == "K m**2 kg**-1 s**-1"

    def test_si_archive_untouched(self):
        out = cesm_to_pipeline_names(
            _tiny_cam_ds(pv_units="K m**2 kg**-1 s**-1"))
        np.testing.assert_allclose(out["pv"].values, 2.0, rtol=1e-6)

    def test_no_double_conversion(self):
        once = cesm_to_pipeline_names(_tiny_cam_ds(pv_units="PVU"))
        twice = cesm_to_pipeline_names(once)
        np.testing.assert_allclose(twice["pv"].values, 2.0e-6, rtol=1e-6)

    def test_other_variables_unscaled(self):
        out = cesm_to_pipeline_names(_tiny_cam_ds(pv_units="PVU"))
        for v in ("u", "v", "w", "z", "t", "q"):
            np.testing.assert_allclose(out[v].values, 2.0, rtol=1e-6)


class TestMissingVariableFailsAtBoundary:
    def test_missing_q_raises_with_names(self, tmp_path):
        ds = _tiny_cam_ds(drop=("Q",))
        fp = tmp_path / "lens2_smbb_m91_1990_plev.nc"
        ds.to_netcdf(fp)
        with pytest.raises(KeyError, match=r"\['q'\]"):
            open_cesm_years_ds(tmp_path, 91, [1990])


class TestMemberCsvCrossCheck:
    _EVENTS_M91 = [("peak", "m091_t00002", 50.0, 10.0, None),
                   ("onset", "m091_t00007", 55.0, 20.0, None)]

    def test_matching_member_passes(self):
        _assert_member_matches_events(self._EVENTS_M91, 91)

    def test_wrong_member_raises(self):
        with pytest.raises(SystemExit, match="m091"):
            _assert_member_matches_events(self._EVENTS_M91, 92)

    def test_era5_int_ids_ignored(self):
        _assert_member_matches_events(
            [("peak", 123, 50.0, 10.0, None)], 92)

    def test_no_member_is_a_noop(self):
        _assert_member_matches_events(self._EVENTS_M91, None)
