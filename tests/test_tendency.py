"""Tests for the tendency computation pipeline module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pvtend.tendency import (
    TendencyComputer,
    TendencyConfig,
    _band_row_index,
    _circ_nearest_lon,
    load_climatology,
    month_keys_for_window,
)


# ── TendencyConfig defaults ─────────────────────────────────────────

class TestTendencyConfig:
    """Test TendencyConfig dataclass defaults and overrides."""

    def test_default_event_type(self):
        cfg = TendencyConfig()
        assert cfg.event_type == "blocking"

    def test_default_qg_method(self):
        cfg = TendencyConfig()
        assert cfg.qg_omega_method == "log20"

    def test_prp_config(self):
        cfg = TendencyConfig(event_type="prp", qg_omega_method="sp19")
        assert cfg.event_type == "prp"
        assert cfg.qg_omega_method == "sp19"

    def test_default_levels(self):
        cfg = TendencyConfig()
        assert 500 in cfg.levels
        assert 200 in cfg.levels

    def test_rel_hours_range(self):
        cfg = TendencyConfig(rel_hours=list(range(-25, 25)))
        assert len(cfg.rel_hours) == 50
        assert cfg.rel_hours[0] == -25
        assert cfg.rel_hours[-1] == 24

    def test_wavg_levels_subset(self):
        cfg = TendencyConfig()
        # wavg_levels should be a subset of levels
        for lev in cfg.wavg_levels:
            assert lev in cfg.levels

    def test_center_mode(self):
        cfg = TendencyConfig(center_mode="lagrangian")
        assert cfg.center_mode == "lagrangian"

    def test_skip_existing(self):
        cfg = TendencyConfig(skip_existing=True)
        assert cfg.skip_existing is True


# ── month_keys_for_window ────────────────────────────────────────────

class TestMonthKeysForWindow:
    """Test (year, month) computation for event time windows."""

    def test_single_month(self):
        ts = pd.Timestamp("2010-06-15")
        keys = month_keys_for_window(ts, hmin=-12, hmax=12)
        assert len(keys) == 1
        assert keys[0] == (2010, 6)

    def test_cross_month_boundary(self):
        ts = pd.Timestamp("2010-01-01 06:00")
        keys = month_keys_for_window(ts, hmin=-12, hmax=12)
        # -12h from Jan 1 06:00 → Dec 31 18:00 (2009,12)
        # +12h from Jan 1 06:00 → Jan 1 18:00  (2010,1)
        assert (2009, 12) in keys
        assert (2010, 1) in keys

    def test_three_months(self):
        ts = pd.Timestamp("2010-02-01 00:00")
        keys = month_keys_for_window(ts, hmin=-49, hmax=24)
        # -49h → Jan 30; +24h → Feb 2 → should cover Jan and Feb
        months = {m for _, m in keys}
        assert 1 in months
        assert 2 in months

    def test_default_blocking_range(self):
        ts = pd.Timestamp("2010-07-15 12:00")
        keys = month_keys_for_window(ts)
        assert len(keys) >= 1
        assert all(isinstance(k, tuple) and len(k) == 2 for k in keys)


# ── load_climatology ────────────────────────────────────────────────

class TestLoadClimatology:
    """Test climatology auto-detection logic."""


# ── Cross-term catalog completeness (v2.0, 53 terms) ────────────────

# 52 named cross-term NPZ keys + Q (stored as a field) = 53 budget terms.
EXPECTED_CROSS_TERM_KEYS = {
    # 12 base (bar/anom × bar/anom)
    "u_anom_pv_bar_dx", "u_anom_pv_anom_dx",
    "u_bar_pv_anom_dx", "u_bar_pv_bar_dx",
    "v_anom_pv_bar_dy", "v_anom_pv_anom_dy",
    "v_bar_pv_anom_dy", "v_bar_pv_bar_dy",
    "w_anom_pv_bar_dp", "w_anom_pv_anom_dp",
    "w_bar_pv_anom_dp", "w_bar_pv_bar_dp",
    # 16 Helmholtz (anom + bar rot/div)
    "u_anom_rot_pv_bar_dx", "u_anom_rot_pv_anom_dx",
    "u_anom_div_pv_bar_dx", "u_anom_div_pv_anom_dx",
    "u_rot_bar_pv_bar_dx", "u_rot_bar_pv_anom_dx",
    "u_div_bar_pv_bar_dx", "u_div_bar_pv_anom_dx",
    "v_anom_rot_pv_bar_dy", "v_anom_rot_pv_anom_dy",
    "v_anom_div_pv_bar_dy", "v_anom_div_pv_anom_dy",
    "v_rot_bar_pv_bar_dy", "v_rot_bar_pv_anom_dy",
    "v_div_bar_pv_bar_dy", "v_div_bar_pv_anom_dy",
    # 16 divergent adiabatic/diabatic horizontal
    "u_div_diabatic_pv_bar_dx", "u_div_diabatic_pv_anom_dx",
    "u_div_adiabatic_pv_bar_dx", "u_div_adiabatic_pv_anom_dx",
    "v_div_diabatic_pv_bar_dy", "v_div_diabatic_pv_anom_dy",
    "v_div_adiabatic_pv_bar_dy", "v_div_adiabatic_pv_anom_dy",
    "u_div_qg_diabatic_pv_bar_dx", "u_div_qg_diabatic_pv_anom_dx",
    "v_div_qg_diabatic_pv_bar_dy", "v_div_qg_diabatic_pv_anom_dy",
    "u_div_lhr_moist_pv_bar_dx", "u_div_lhr_moist_pv_anom_dx",
    "v_div_lhr_moist_pv_bar_dy", "v_div_lhr_moist_pv_anom_dy",
    # 8 alt vertical
    "w_adiabatic_pv_bar_dp", "w_adiabatic_pv_anom_dp",
    "w_diabatic_pv_bar_dp", "w_diabatic_pv_anom_dp",
    "w_qg_diabatic_pv_bar_dp", "w_qg_diabatic_pv_anom_dp",
    "w_lhr_moist_pv_bar_dp", "w_lhr_moist_pv_anom_dp",
}


class TestCrossTermCatalog:
    """Verify the 53-term v2.0 cross-term catalog is complete."""

    def test_expected_count(self):
        """52 named cross-term keys + Q (stored as field) = 53."""
        assert len(EXPECTED_CROSS_TERM_KEYS) == 52

    def test_no_duplicates(self):
        """All keys are unique (set length matches list-of-elements length)."""
        keys_list = list(EXPECTED_CROSS_TERM_KEYS)
        assert len(keys_list) == len(set(keys_list))

    def test_base_terms_present(self):
        base_u = {k for k in EXPECTED_CROSS_TERM_KEYS if k.startswith("u_anom_pv") or k.startswith("u_bar_pv")}
        assert len(base_u) == 4

    def test_helmholtz_bar_terms_present(self):
        bar_helm = {k for k in EXPECTED_CROSS_TERM_KEYS
                    if "rot_bar" in k or "div_bar" in k}
        assert len(bar_helm) == 8  # 4 u-dir + 4 v-dir

    def test_no_harmonic_cross_terms(self):
        """Harmonic absorbed into residual — no u_har/v_har cross-terms."""
        har_terms = {k for k in EXPECTED_CROSS_TERM_KEYS if "_har_" in k}
        assert len(har_terms) == 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Climatology missing"):
            load_climatology(tmp_path / "nonexistent.nc")

    def test_missing_parent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_climatology(tmp_path / "no_such_dir" / "clim.nc")


# ── PPVI inversion-band geometry ─────────────────────────────────────

class TestPPVIBandGeometry:
    """The Wu cores never receive latitudes — they rebuild row I's as
    ``lat_n-(I-1)*dlat`` and key cos(phi) and the Coriolis f off that index.
    Row 0 must therefore be the northernmost, whatever order the source file
    happens to use. ERA5 is stored N->S and satisfies it for free; CESM f09 is
    stored S->N, which fed the inversion an upside-down f and left every
    cropped PPVI field 100 % NaN.
    """

    @staticmethod
    def _ds(lat, dlon=1.25):
        lon = np.arange(0.0, 360.0, dlon)
        return xr.Dataset(coords={"latitude": lat, "longitude": lon,
                                  "level": np.array([1000, 500, 100])})

    # f09 NH subset as written by 07_isobaric_nh: ascending, 0.47 -> 90.
    F09_LAT = 0.47120419 + np.arange(96) * (180.0 / 191.0)
    # ERA5 1.5 deg NH: descending, 90 -> 0.
    ERA5_LAT = np.arange(90.0, -0.1, -1.5)

    def _geom(self, lat):
        tc = TendencyComputer(TendencyConfig())
        return tc._ppvi_geom(self._ds(lat), 90.0)

    def test_ascending_grid_is_reordered_north_first(self):
        g = self._geom(self.F09_LAT)
        assert g["band_lats"][0] > g["band_lats"][-1], "band must run N->S"
        assert g["ny"] == 80

    def test_descending_grid_is_left_alone(self):
        g = self._geom(self.ERA5_LAT)
        assert g["band_lats"][0] > g["band_lats"][-1]
        np.testing.assert_allclose(g["band_lats"][0], 85.5)

    def test_band_latitudes_are_grid_latitudes(self):
        """Reordering must permute the band, never resample it."""
        for lat in (self.F09_LAT, self.ERA5_LAT):
            g = self._geom(lat)
            np.testing.assert_array_equal(np.sort(g["band_lats"]),
                                          np.sort(lat[np.sort(g["band_idx"])]))

    def test_zhdr_edges_match_actual_band(self):
        """zhdr's span contract (lat_n-lat_s == (ny-1)*dlat) must hold, or
        invert_piecewise refuses the header."""
        for lat in (self.F09_LAT, self.ERA5_LAT):
            g = self._geom(lat)
            lat_s, lat_n, dlat, ny = (g["zhdr"][0], g["zhdr"][2],
                                      g["zhdr"][4], int(g["zhdr"][7]))
            assert abs((lat_n - lat_s) - (ny - 1) * dlat) < 0.5 * dlat


class TestBandRowIndex:
    """Regression for the all-NaN PPVI crop: patch rows are matched to band
    rows by value, so neither ordering nor an off-nominal band edge can leave
    the mapping empty.
    """

    DLAT = 180.0 / 191.0
    BAND_DESC = (0.47120419 + np.arange(11, 91) * DLAT)[::-1]   # 85.29 -> 10.84

    def _patch(self, ilat=66, pad=32):
        lat = 0.47120419 + np.arange(96) * self.DLAT
        out = np.full(2 * pad + 1, np.nan)
        for j, k in enumerate(range(ilat - pad, ilat + pad + 1)):
            if 0 <= k < lat.size:
                out[j] = lat[k]
        return out

    def test_f09_patch_rows_are_matched(self):
        """The bug: 0 of 62 finite rows matched, so the crop was pure NaN."""
        row_for = _band_row_index(self._patch(), self.BAND_DESC, self.DLAT)
        assert (row_for >= 0).sum() == 57

    def test_rows_beyond_the_band_edge_are_unmatched(self):
        lat_vec = self._patch()
        unmatched = lat_vec[_band_row_index(lat_vec, self.BAND_DESC,
                                            self.DLAT) < 0]
        unmatched = unmatched[np.isfinite(unmatched)]
        assert unmatched.min() > self.BAND_DESC[0], \
            "only rows poleward of the band's north edge may go unmatched"

    def test_no_two_patch_rows_share_a_band_row(self):
        row_for = _band_row_index(self._patch(), self.BAND_DESC, self.DLAT)
        hit = row_for[row_for >= 0]
        assert hit.size == np.unique(hit).size

    def test_matches_are_exact(self):
        lat_vec = self._patch()
        row_for = _band_row_index(lat_vec, self.BAND_DESC, self.DLAT)
        m = row_for >= 0
        np.testing.assert_array_equal(self.BAND_DESC[row_for[m]], lat_vec[m])

    def test_ordering_agnostic(self):
        """The same patch rows resolve to the same latitudes whichever way the
        band is stored; only the row numbers differ."""
        lat_vec = self._patch()
        asc = self.BAND_DESC[::-1]
        r_desc = _band_row_index(lat_vec, self.BAND_DESC, self.DLAT)
        r_asc = _band_row_index(lat_vec, asc, self.DLAT)
        np.testing.assert_array_equal(r_desc >= 0, r_asc >= 0)
        m = r_desc >= 0
        np.testing.assert_array_equal(self.BAND_DESC[r_desc[m]], asc[r_asc[m]])

    def test_nan_patch_rows_are_unmatched(self):
        lat_vec = np.array([np.nan, self.BAND_DESC[3], np.nan])
        np.testing.assert_array_equal(
            _band_row_index(lat_vec, self.BAND_DESC, self.DLAT), [-1, 3, -1])


class TestPPVIPieceKeys:
    """The key set that ``--skip-existing`` tests and that
    ``compute_ppvi_for_event`` asserts on afterwards must follow the configured
    decomposition. Hardcoding ``u_rot_anom_ppvi_250_3d`` made "scale" runs both
    unresumable (nothing ever matched, so nothing was skipped) and unable to
    write at all (the post-condition raised on every event).
    """

    @staticmethod
    def _keys(mode):
        return TendencyComputer(
            TendencyConfig(ppvi_pieces=mode))._ppvi_piece_keys()

    def test_per_level_covers_all_nine_wu_levels(self):
        keys = self._keys("per_level")
        assert len(keys) == 9
        assert "u_rot_anom_ppvi_250_3d" in keys
        assert "u_rot_anom_ppvi_1000_3d" in keys

    def test_scale_covers_the_four_scale_pieces(self):
        assert set(self._keys("scale")) == {
            "u_rot_anom_ppvi_surface_3d", "u_rot_anom_ppvi_lower_3d",
            "u_rot_anom_ppvi_upper_p_3d", "u_rot_anom_ppvi_upper_e_3d"}

    def test_scale_claims_no_per_level_key(self):
        """The exact confusion behind the bug."""
        assert "u_rot_anom_ppvi_250_3d" not in self._keys("scale")

    def test_default_is_per_level(self):
        assert self._keys("per_level") == TendencyComputer(
            TendencyConfig())._ppvi_piece_keys()


class TestCircNearestLon:
    """The longitude axis wraps; a plain argmin does not know that. Events near
    the dateline were centred on the wrong side of the seam — 176 of the 85,425
    catalogue rows, all North Pacific.
    """

    LON = np.arange(-180.0, 180.0, 1.25)   # f09 after the 0-360 -> -180..180 fix

    def test_picks_across_the_seam(self):
        """179.7 is 0.3 deg from -180 and 0.95 from 178.75."""
        assert _circ_nearest_lon(self.LON, 179.7) == 0

    def test_agrees_with_plain_argmin_away_from_the_seam(self):
        for lon0 in (-140.0, -26.95, 0.3, 87.6, 150.0):
            assert (_circ_nearest_lon(self.LON, lon0)
                    == int(np.abs(self.LON - lon0).argmin()))

    def test_never_worse_than_plain_argmin(self):
        """Circular distance to the chosen point is minimal, everywhere."""
        for lon0 in np.arange(-180.0, 180.0, 0.37):
            i = _circ_nearest_lon(self.LON, lon0)
            d = np.abs((self.LON - lon0 + 180.0) % 360.0 - 180.0)
            assert d[i] == pytest.approx(d.min())

    def test_equivalent_on_a_0_360_axis(self):
        """Wrapping is what matters, not which convention the axis uses."""
        lon360 = np.arange(0.0, 360.0, 1.25)
        assert lon360[_circ_nearest_lon(lon360, -0.4)] == pytest.approx(0.0)
        assert lon360[_circ_nearest_lon(lon360, 359.8)] == pytest.approx(0.0)


class TestOpenSourceDsDispatch:
    """`compute_ppvi_for_event` used to open its window with open_months_ds,
    which is the ERA5 one-file-per-(variable, month) layout. On CESM -- one file
    per (member, year) -- every event died with "No files for z in months
    [(1985, 2)]", so `pvtend-pipeline ppvi --skip-existing` could not append to
    the very catalogue it exists to fill.
    """

    @staticmethod
    def _cfg(**kw):
        return TendencyConfig(rel_hours=[0], engine="netcdf4", **kw)

    def test_cesm_goes_to_the_year_reader(self, monkeypatch, tmp_path):
        import pvtend.tendency as T
        seen = {}
        monkeypatch.setattr(T, "open_cesm_years_ds",
                            lambda *a, **k: seen.update(args=a, kw=k) or "CESM")
        monkeypatch.setattr(T, "open_months_ds",
                            lambda *a, **k: pytest.fail("used the ERA5 reader"))
        got = T.open_source_ds(
            self._cfg(source="cesm", member=91, data_dir=tmp_path),
            pd.Timestamp("1985-02-08"), var_list=["z", "t", "u", "v"])
        assert got == "CESM"
        assert seen["args"][1] == 91

    def test_era5_still_goes_to_the_month_reader(self, monkeypatch, tmp_path):
        import pvtend.tendency as T
        seen = {}
        monkeypatch.setattr(T, "open_months_ds",
                            lambda *a, **k: seen.update(args=a) or "ERA5")
        got = T.open_source_ds(self._cfg(source="era5", data_dir=tmp_path),
                               pd.Timestamp("2010-06-15"))
        assert got == "ERA5"
        assert seen["args"][1] == ["u", "v", "w", "pv", "z", "t", "q"]

    def test_var_list_narrows_the_era5_open(self):
        import pvtend.tendency as T
        seen = {}
        orig = T.open_months_ds
        try:
            T.open_months_ds = lambda *a, **k: seen.update(args=a) or "ERA5"
            T.open_source_ds(self._cfg(source="era5", data_dir=Path(".")),
                             pd.Timestamp("2010-06-15"),
                             var_list=["z", "t", "u", "v"])
        finally:
            T.open_months_ds = orig
        assert seen["args"][1] == ["z", "t", "u", "v"]

    def test_cesm_chunk_keys_are_translated_to_raw_dims(self, monkeypatch, tmp_path):
        """`chunks` is applied to the raw file, whose dims are still CAM's."""
        import pvtend.tendency as T
        (tmp_path / "lens2_smbb_m91_1985_plev.nc").write_bytes(b"")
        seen = {}

        def _capture(*a, **k):
            seen.update(k)
            raise RuntimeError("stop here")

        monkeypatch.setattr(T.xr, "open_mfdataset", _capture)
        with pytest.raises(RuntimeError, match="stop here"):
            T.open_cesm_years_ds(tmp_path, 91, [1985],
                                 chunks={"valid_time": 1, "latitude": 4})
        assert seen["chunks"] == {"time": 1, "lat": 4}


class TestArchiveSplitKeys:
    """The archive-PV split keys (pv_anom_p/e) added 2026-08-19."""

    def test_track_id_parser_handles_both_schemes(self):
        from pathlib import Path
        from pvtend.classify import _parse_track_id
        assert _parse_track_id(
            Path("track_m091_t00002_1985020800_dh+0.npz")) == "m091_t00002"
        assert _parse_track_id(
            Path("track_1234_2001010100_dh+0.npz")) == 1234
        # the member prefix must survive: a bare int would collide across
        # members (every member has a t00002)
        a = _parse_track_id(Path("track_m091_t00002_x_dh+0.npz"))
        b = _parse_track_id(Path("track_m092_t00002_x_dh+0.npz"))
        assert a != b

    def test_arch_keys_identity_and_shapes(self):
        """p + e == stored total exactly; mask uint8; wavg keys present."""
        import pvtend.tendency as T
        ny, nx, nlev = 21, 33, 9
        levels = np.array([1000, 850, 700, 500, 400, 300, 250, 200, 100])
        band_lats = 80.0 - np.arange(60) * 0.9375          # N->S band
        lat_vec = 60.0 - np.arange(ny) * 0.9375            # patch rows in band
        rng = np.random.default_rng(0)
        total = rng.standard_normal((nlev, ny, nx))
        store = dict(
            pv_anom_3d=total,
            lat_vec=lat_vec,
            levels=levels,
            wavg_levels=np.array([400, 300, 250, 200]),
            center_lon=100.0,
            z_3d=np.linspace(100, 16000, nlev)[:, None, None]
            * np.ones((nlev, ny, nx)),
        )
        geom = dict(band_lats=band_lats, dlat=0.9375, dlon=1.25,
                    lon_all=np.arange(0.0, 360.0, 1.25), nlon=288)
        arch = dict(
            q_p=rng.standard_normal((nlev, len(band_lats), 288)),
            mask=(rng.random((nlev, len(band_lats), 288)) > 0.9),
            q_min=-5.0, thresh=-1.75)
        cfg = T.TendencyConfig(source="cesm", member=91)
        comp = T.TendencyComputer.__new__(T.TendencyComputer)
        comp.cfg = cfg
        new = comp._arch_keys_from_split(arch, store, geom)
        assert new["pv_split_mask_3d"].dtype == np.uint8
        s = new["pv_anom_p_3d"] + new["pv_anom_e_3d"]
        np.testing.assert_allclose(s, total, atol=1e-12)
        assert new["pv_anom_p"].shape == (ny, nx)
        assert np.isfinite(new["pv_anom_p"]).all()


# ── vertical-weight convention of the archive split ──────────────────

_ARCHIVE = Path("/net/flood/data2/users/x_yan/pvtend/outputs/blocking")


def _one_real_npz() -> Path | None:
    """A real catalogue NPZ, or None where the archive is not mounted."""
    if not _ARCHIVE.is_dir():
        return None
    return next(iter(sorted(_ARCHIVE.glob("*/dh=*/track_*.npz"))), None)


@pytest.mark.skipif(_one_real_npz() is None,
                    reason="local ERA5 NPZ catalogue not mounted")
class TestWavgWeightConvention:
    """``store["z_3d"]`` is height in metres — never divide it by g again.

    ``_arch_keys_from_split`` used to build its ``exp(-z/H)`` weights from
    ``store["z_3d"] / z_divisor(cfg)``. But ``z_3d`` is written as ``z_m_3d``,
    which was already divided where it was built, so on ERA5 the weights came
    out as ``exp(-z/gH)``: z/9.81 is ~1 km where the height is ~10 km, exp()
    barely varies over the layer, and the weighted average collapsed to nearly
    flat. CESM never showed it because ``z_divisor`` returns 1.0 there.

    The test is anchored on a real NPZ rather than a constructed one because
    the whole point is the *units actually on disk*, which a fixture would
    simply restate. Weighting ``pv_anom_3d`` with the stored ``z_3d`` must
    reproduce the stored 2-D ``pv_anom`` exactly — that is the same
    ``exp(-z/H)`` average the writer applies — and the doubly-divided form
    must not.
    """

    @staticmethod
    def _wavg(a3, z3):
        from pvtend.constants import H_SCALE
        wt = np.exp(-z3 / H_SCALE)
        num = np.nansum(a3 * wt, axis=0)
        den = np.nansum(np.where(np.isfinite(a3), wt, 0.0), axis=0)
        out = np.full(num.shape, np.nan)
        m = den > 0
        out[m] = num[m] / den[m]
        return out

    def _parts(self):
        with np.load(_one_real_npz(), allow_pickle=True) as z:
            lev = [int(v) for v in z["levels"]]
            widx = [lev.index(int(p)) for p in z["wavg_levels"]]
            return (np.asarray(z["pv_anom_3d"], float)[widx],
                    np.asarray(z["z_3d"], float)[widx],
                    np.asarray(z["pv_anom"], float))

    def test_z_3d_is_height_not_geopotential(self):
        _, z3, _ = self._parts()
        # Geopotential would be ~g x larger; 300-200 hPa heights are ~9-13 km.
        assert np.nanmax(z3) < 3.0e4

    def test_stored_z_reproduces_the_stored_2d_field(self):
        a3, z3, stored = self._parts()
        assert np.allclose(self._wavg(a3, z3), stored,
                           equal_nan=True, rtol=1e-6, atol=1e-9)

    def test_dividing_by_g_again_does_not(self):
        from pvtend.constants import G0
        a3, z3, stored = self._parts()
        assert not np.allclose(self._wavg(a3, z3 / G0), stored,
                               equal_nan=True, rtol=1e-6, atol=1e-9)
