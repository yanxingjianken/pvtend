"""The spherical PPVI engine inside this pipeline's own conventions.

The solver is tested where it lives (pv_inversion_spherical).  What is tested
here is the adapter: that an archive stored north-to-south, or on a -180..180
longitude axis, reaches the engine in its order and comes back on the record's
own patch; that rows past the pole are NaN and the pole row too; that the
residual is the record's observed anomaly minus the pieces, exactly; and that
the key contract the rest of the pipeline reads is written in full.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pvtend.ppvi import spherical_engine as sph
from pvtend.tendency import TendencyComputer, TendencyConfig, _plev_name

G0 = 9.80665
RD = 287.0
WU = sph.WU_PLEVS


class TestHemisphereAxes:
    def test_era5_layout_is_reversed_into_ascending_rows(self):
        lat = np.arange(90.0, -0.1, -1.5)           # 61 rows, north first
        lon = np.arange(0.0, 360.0, 1.5)
        axes = sph.hemisphere_axes(lat, lon)
        np.testing.assert_array_equal(axes.nh_rows, np.arange(60, -1, -1))
        assert axes.lat_nh[0] == 0.0 and axes.lat_nh[-1] == 90.0
        np.testing.assert_array_equal(axes.lon_order, np.arange(lon.size))
        np.testing.assert_array_equal(axes.col_of, np.arange(lon.size))
        assert axes.dlat == pytest.approx(1.5)

    def test_cesm_layout_rolls_the_longitude_axis(self):
        lat = -90.0 + np.arange(192) * (180.0 / 191.0)   # south first
        lon = np.arange(-180.0, 180.0, 1.25)             # 288 columns
        axes = sph.hemisphere_axes(lat, lon)
        np.testing.assert_array_equal(axes.nh_rows, np.arange(96, 192))
        assert axes.lat_nh[0] == pytest.approx(0.47120419)
        assert axes.lat_nh[-1] == pytest.approx(90.0)
        assert axes.lon[0] == 0.0 and axes.lon[-1] == pytest.approx(358.75)
        assert axes.lon_order[0] == 144            # archive column of lon 0
        assert axes.col_of[144] == 0
        assert axes.col_of[0] == 144               # -180 is 180, mid-axis
        # Reordering a cube's columns puts lon 0 first.
        cube = np.broadcast_to(lon, (2, 96, 288))
        rolled = sph.to_engine_order(cube, axes)
        np.testing.assert_allclose(np.mod(rolled[0, 0], 360.0), axes.lon)

    def test_southern_hemisphere_only_is_refused(self):
        with pytest.raises(ValueError, match="northern"):
            sph.hemisphere_axes(np.arange(-90.0, -1.0, 5.0), np.arange(0.0, 360.0, 5.0))


class TestCrop:
    LAT_NH = np.arange(0.0, 90.1, 5.0)   # 19 rows
    LON = np.arange(0.0, 360.0, 5.0)     # 72 columns

    def test_rows_by_value_columns_by_index_nan_past_the_pole(self):
        field = (self.LAT_NH[:, None] * 1000.0 + self.LON[None, :])[None]
        # A patch stored north to south whose first two rows lie past the pole.
        lat_vec = np.array([np.nan, np.nan, 90.0, 85.0, 80.0, 75.0])
        row_for = sph.patch_row_index(lat_vec, self.LAT_NH, 5.0)
        np.testing.assert_array_equal(row_for, [-1, -1, 18, 17, 16, 15])
        cols = np.array([70, 71, 0, 1, 2])          # across the seam
        out = sph.crop_to_patch(field, row_for, cols)
        assert out.shape == (1, 6, 5)
        assert np.isnan(out[0, :2]).all()
        np.testing.assert_array_equal(out[0, 2], 90000.0 + np.array([350, 355, 0, 5, 10]))
        np.testing.assert_array_equal(out[0, 5], 75000.0 + np.array([350, 355, 0, 5, 10]))

    def test_off_grid_row_is_unmatched(self):
        row_for = sph.patch_row_index(np.array([42.0, 40.0]), self.LAT_NH, 5.0)
        np.testing.assert_array_equal(row_for, [-1, 8])


def _synthetic(lat, lon, ts, factor=1.0, vortex=None):
    """A hydrostatic hemisphere on a 5-degree ERA5-like grid, as a dataset.

    ``vortex=(lat0, lon0)`` adds an anticyclone there whose streamfunction grows
    with height: a negative upper-level potential-vorticity anomaly for the
    scale split to find, as a blocking event has.
    """
    lon2d, lat2d = np.meshgrid(lon, lat)
    nlev = len(WU)
    p_hpa = np.array(WU, dtype=float)
    t = np.stack([288.0 - 6.5 * k + 4.0 * np.cos(np.radians(lat2d)) for k in range(nlev)])
    t = t * factor
    h = np.empty_like(t)
    h[0] = 100.0 + 150.0 * np.cos(np.radians(lat2d)) + 60.0 * np.sin(np.radians(lon2d))
    for k in range(1, nlev):
        t_bar = 0.5 * (t[k] + t[k - 1])
        h[k] = h[k - 1] + (RD * t_bar / G0) * np.log(p_hpa[k - 1] / p_hpa[k])
    u = np.stack([(6.0 + 2.0 * k) * np.cos(np.radians(lat2d)) for k in range(nlev)]) * factor
    v = np.stack([2.0 * np.sin(np.radians(2 * lon2d)) * np.cos(np.radians(lat2d))
                  for _ in range(nlev)])
    if vortex is not None:
        lat0, lon0 = vortex
        a = 6.371e6
        cosd = (np.sin(np.radians(lat2d)) * np.sin(np.radians(lat0))
                + np.cos(np.radians(lat2d)) * np.cos(np.radians(lat0))
                * np.cos(np.radians(lon2d - lon0)))
        dist = np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))
        bump = np.exp(-0.5 * (dist / 12.0) ** 2)
        for k in range(nlev):
            psi = 4.0e6 * (k / (nlev - 1)) * bump
            dpsi_dphi = np.gradient(psi, np.radians(lat), axis=0)
            dpsi_dlam = np.gradient(psi, np.radians(lon), axis=1)
            cos = np.cos(np.radians(lat2d))
            ok = cos > 1e-3
            u[k] = u[k] - dpsi_dphi / a
            v[k] = v[k] + np.where(ok, dpsi_dlam / (a * np.where(ok, cos, 1.0)), 0.0)
    pv = np.stack([(0.5 + 0.3 * k) * 1e-6 * np.sin(np.radians(lat2d)) for k in range(nlev)])
    data = {"z": h * G0, "t": t, "u": u, "v": v, "pv": pv}
    return data


def _event_ds(lat, lon, ts, vortex=None):
    d = _synthetic(lat, lon, ts, factor=1.01, vortex=vortex)
    return xr.Dataset(
        {k: (("valid_time", "level", "latitude", "longitude"), v[None]) for k, v in d.items()},
        coords={"valid_time": [ts], "level": WU, "latitude": lat, "longitude": lon},
    )


def _clim_ds(lat, lon, ts):
    d = _synthetic(lat, lon, ts, factor=1.0)
    return xr.Dataset(
        {k: (("month", "day", "hour", "level", "latitude", "longitude"), v[None, None, None])
         for k, v in d.items()},
        coords={"month": [ts.month], "day": [ts.day], "hour": [ts.hour],
                "level": WU, "latitude": lat, "longitude": lon},
    )


class TestSphericalKeys:
    """End to end through ``_ppvi_compute_keys`` on a tiny solver grid."""

    LAT = np.arange(90.0, -0.1, -5.0)    # ERA5 order, 19 rows
    LON = np.arange(0.0, 360.0, 5.0)     # 72 columns
    TS = pd.Timestamp("2001-01-15 12:00")

    def _store(self, center_lat, center_lon, yp, xp):
        # The base patch as process_event cuts it: rows by index about the
        # centre, north first on this archive, NaN past the pole.
        ilat = int(np.argmin(np.abs(self.LAT - center_lat)))
        pad = (yp - 1) // 2
        lat_vec = np.full(yp, np.nan)
        for j, k in enumerate(range(ilat - pad, ilat + pad + 1)):
            if 0 <= k < self.LAT.size:
                lat_vec[j] = self.LAT[k]
        rng = np.random.default_rng(0)
        return {
            "center_lat": np.float64(center_lat),
            "center_lon": np.float64(center_lon),
            "lat_vec": lat_vec,
            "levels": np.array(WU),
            "u_rot_anom_3d": rng.normal(0.0, 3.0, (len(WU), yp, xp)),
            "v_rot_anom_3d": rng.normal(0.0, 3.0, (len(WU), yp, xp)),
        }

    def _run(self, pieces, center_lat=80.0, center_lon=357.5):
        cfg = TendencyConfig(
            source="era5", ppvi_engine="spherical", ppvi_pieces=pieces,
            ppvi_solver_nlat=32, ppvi_solver_nlon=64, ppvi_lmax=20,
            ppvi_newton_max_steps=3,
        )
        tc = TendencyComputer(cfg)
        ds = _event_ds(self.LAT, self.LON, self.TS, vortex=(center_lat, center_lon))
        clim = _clim_ds(self.LAT, self.LON, self.TS)
        geom = tc._ppvi_geom(ds, 90.0)
        yp, xp = 13, 25                      # +-30 x +-60 degrees at 5 degrees
        store = self._store(center_lat, center_lon, yp, xp)
        new = tc._ppvi_compute_keys(store, self.TS, ds, clim, _plev_name(clim), geom)
        return tc, store, new

    def test_scale_pieces_close_on_the_record_and_blank_past_the_pole(self):
        tc, store, new = self._run("scale")
        for key in tc._ppvi_piece_keys():
            assert key in new
        assert "u_rot_anom_ppvi_wall_3d" not in new
        pieces = sph.piece_keys("scale")
        summed = sum(new[f"u_rot_anom_ppvi_{p}_3d"] for p in pieces)
        resid = new["u_rot_anom_residual_ppvi_3d"]
        assert resid.shape == (len(WU), 13, 25)
        # Rows 0-3 lie past the pole, row 4 is the pole itself: no components.
        assert np.isnan(resid[:, :5, :]).all()
        finite = np.isfinite(resid)
        assert finite[:, 5:, :].all()
        np.testing.assert_array_equal(
            resid[finite], (store["u_rot_anom_3d"] - summed)[finite])
        # The engine's own anomaly and its column average are delivered.
        assert np.isfinite(new["u_rot_anom_sph_3d"][:, 5:, :]).all()
        assert new["u_rot_anom_sph"].shape == (13, 25)
        assert np.isnan(new["u_rot_anom_sph"][:5]).all()
        assert np.isfinite(new["u_rot_anom_sph"][5:]).all()
        # Potential vorticity: interior levels only.
        pv = new["pv_anom_wu_3d"]
        assert np.isnan(pv[0]).all() and np.isnan(pv[-1]).all()
        assert np.isfinite(pv[1:-1, 5:, :]).all()
        # The convergence record.
        assert str(new["ppvi_engine"]) == "spherical"
        assert str(new["ppvi_pieces"]) == "scale"
        assert int(new["ppvi_newton_steps"]) >= 1
        assert "ppvi_newton_final_increment_m" in new
        assert "ppvi_split_q_min" in new and "ppvi_split_mask_frac" in new
        for p in pieces:
            assert np.isfinite(new[f"max_abs_u_rot_anom_ppvi_{p}"])
        assert np.isfinite(new["max_abs_u_rot_anom_residual_ppvi"])

    def test_per_level_pieces_are_named_by_pressure(self):
        tc, store, new = self._run("per_level", center_lat=45.0, center_lon=10.0)
        for p in WU:
            assert f"u_rot_anom_ppvi_{p}_3d" in new
            assert new[f"v_rot_anom_ppvi_{p}"].shape == (13, 25)
        assert np.isfinite(new["u_rot_anom_residual_ppvi_3d"]).all()
        assert "ppvi_split_q_min" not in new

    def test_engine_is_built_once_per_worker(self):
        tc, store, new = self._run("scale")
        engine = tc._sph_engine
        assert engine is not None
        ds = _event_ds(self.LAT, self.LON, self.TS)
        assert tc._spherical_engine(tc._ppvi_geom(ds, 90.0)["sph_axes"]) is engine

    def test_nested_is_refused_on_the_sphere(self):
        cfg = TendencyConfig(ppvi_engine="spherical", ppvi_pieces="scale", ppvi_nested=True)
        tc = TendencyComputer(cfg)
        ds = _event_ds(self.LAT, self.LON, self.TS)
        geom = tc._ppvi_geom(ds, 90.0)
        with pytest.raises(ValueError, match="windowed engine only"):
            tc._ppvi_compute_keys({}, self.TS, ds, ds, "level", geom)
