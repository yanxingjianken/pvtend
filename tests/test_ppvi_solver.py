"""Tests for the Wu PPVI solver's grid-agnostic support (pvtend ≥2.13).

Covers the two f09 (anisotropic, terrain-following) fixes:
  * ``fill_below_ground`` — hydrostatic gap-fill of below-ground NaN (pure
    Python; CI-safe).
  * the ``zhdr`` Δlat/Δlon reordering — exercised by the gated end-to-end
    inversion test, which is skipped when the locally-built ``_wuppvi``
    Fortran extension is unavailable (it is not shipped in the PyPI wheel).
Plus the ``GridProfile`` config dataclass.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvtend.grid import GridProfile, ERA5_1P5_NH, CESM_F09, GRID_PROFILES
from pvtend.ppvi.solver import fill_below_ground, PR


# ── GridProfile ───────────────────────────────────────────────────────
class TestGridProfile:
    def test_era5_default_isotropic(self):
        assert ERA5_1P5_NH.isotropic is True
        assert ERA5_1P5_NH.z_is_height is False     # ERA5 z is geopotential

    def test_cesm_f09_anisotropic(self):
        assert CESM_F09.isotropic is False          # 0.942 ≠ 1.25
        assert CESM_F09.z_is_height is True          # CESM z is height [m]
        assert CESM_F09.nlat == 192 and CESM_F09.nlon == 288
        assert (CESM_F09.inv_band_s, CESM_F09.inv_band_n) == (10.5, 85.5)

    def test_registry(self):
        assert GRID_PROFILES["ERA5_1P5_NH"] is ERA5_1P5_NH
        assert GRID_PROFILES["CESM_F09"] is CESM_F09

    def test_frozen(self):
        with pytest.raises(Exception):
            CESM_F09.dlat = 1.0  # type: ignore[misc]


# ── fill_below_ground (pure Python) ───────────────────────────────────
class TestFillBelowGround:
    def _cubes(self, nl=9, ny=6, nx=7):
        rng = np.random.default_rng(0)
        # plausible magnitudes: z [m], T [K], winds [m/s]
        z = np.linspace(1000, 16000, nl)[:, None, None] + rng.standard_normal((nl, ny, nx))
        t = np.full((nl, ny, nx), 250.0) + rng.standard_normal((nl, ny, nx))
        u = rng.standard_normal((nl, ny, nx)) * 10
        v = rng.standard_normal((nl, ny, nx)) * 10
        return z, t, u, v

    def test_noop_on_finite_returns_same_object(self):
        z, t, u, v = self._cubes()
        out = fill_below_ground(z, t, u, v)
        assert out[0] is z and out[1] is t  # untouched (ERA5 byte-exact path)

    def test_fills_below_ground_nan(self):
        z, t, u, v = self._cubes()
        z[0, :2, :2] = np.nan       # 1000 hPa below ground
        t[0, :2, :2] = np.nan
        u[0, :2, :2] = np.nan
        v[0, :2, :2] = np.nan
        Z, T, U, V = fill_below_ground(z, t, u, v)
        assert np.isfinite(Z).all() and np.isfinite(T).all()
        assert np.isfinite(U).all() and np.isfinite(V).all()

    def test_tuv_constant_downward(self):
        z, t, u, v = self._cubes()
        t[0] = np.nan
        _, T, _, _ = fill_below_ground(z, t, u, v)
        # surface (idx0) filled from 850 (idx1) by construction
        np.testing.assert_allclose(T[0], t[1], rtol=0, atol=0)

    def test_z_hydrostatic_monotonic(self):
        """Filled surface height must be LOWER than the level above (height
        decreases downward), i.e. dz < 0."""
        z, t, u, v = self._cubes()
        z[0] = np.nan
        Z, _, _, _ = fill_below_ground(z, t, u, v)
        assert np.all(Z[0] < Z[1])

    def test_pr_levels_match_solver(self):
        # PR is σ = p/p0 descending [1.0 … 0.1]; surface is index 0.
        assert PR[0] == pytest.approx(1.0)
        assert PR[-1] == pytest.approx(0.1)


# ── Gated end-to-end inversion on an anisotropic (f09-like) grid ──────
def _ext_available():
    try:
        from pvtend.ppvi._ext import load_ext
        load_ext()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ext_available(),
                    reason="_wuppvi Fortran extension not built (PyPI wheel)")
class TestInvertAnisotropic:
    """invert_piecewise must produce finite, sane output on an anisotropic
    grid with below-ground NaN — the f09 regime."""

    def test_finite_output_anisotropic_with_nan(self):
        """On an anisotropic (f09-like) grid, a stratified mean state + a modest
        perturbation + below-ground NaN must invert to finite interior output —
        proving the v2.13 below-ground fill + zhdr Δlat/Δlon reordering work
        (with the OLD code this regime gave ~all-NaN garbage)."""
        from pvtend.ppvi.solver import invert_piecewise, PR
        # ny chosen so that WITHOUT the zhdr Δlat/Δlon swap the Fortran would
        # reconstruct row latitudes with Δlon=1.25 → 80−59·1.25 = 6.25°N (the
        # singular low-lat band → blow-up), while WITH the swap (Δlat=0.942) the
        # band stays 24–80°N. So finite interior output here proves BOTH the
        # swap and the below-ground fill are active.
        nl, ny, nx = 9, 60, 60
        dlat, dlon = 180.0 / 191.0, 1.25         # anisotropic, f09-like
        lat_n = 80.0
        latv = lat_n - np.arange(ny) * dlat
        p = np.asarray(PR) * 1000.0              # hPa [1000…100]
        # realistic US-standard-atmosphere-ish T profile (non-zero stratification)
        T_prof = np.array([288., 279., 269., 252., 240., 229., 222., 217., 210.])
        # hydrostatic mean height from the T profile (surface 100 m, up)
        Rd, g = 287.05, 9.80665
        Hp = np.empty(nl)
        Hp[0] = 100.0
        for k in range(1, nl):
            Tbar = 0.5 * (T_prof[k] + T_prof[k - 1])
            Hp[k] = Hp[k - 1] + Rd * Tbar / g * np.log(p[k - 1] / p[k])
        ones = np.ones((ny, nx))
        H_m = (Hp[:, None, None] * ones)
        T_m = (T_prof[:, None, None] * ones).copy()
        rng = np.random.default_rng(1)
        U_m = 10.0 * np.cos(np.deg2rad(latv))[None, :, None] * ones + 0.1 * rng.standard_normal((nl, ny, nx))
        V_m = 0.1 * rng.standard_normal((nl, ny, nx))
        # event = mean + a localized upper-level ridge (geostrophically modest)
        xc = np.exp(-((np.arange(nx) - nx / 2) ** 2) / 80.0)[None, None, :]
        yc = np.exp(-((latv - latv[ny // 2]) ** 2) / 120.0)[None, :, None]
        H_e = H_m + 60.0 * (np.linspace(0, 1, nl)[:, None, None]) * xc * yc
        T_e, U_e, V_e = T_m.copy(), U_m.copy(), V_m.copy()
        # inject below-ground NaN at the surface (idx 0) in a corner
        for A in (H_e, T_e, U_e, V_e, H_m, T_m, U_m, V_m):
            A[0, :3, :3] = np.nan
        zhdr = np.array([latv[-1], 0.0, latv[0], (nx - 1) * dlon,
                         dlat, dlon, nx, ny], dtype=np.float32)
        res = invert_piecewise(H_m, T_m, U_m, V_m, H_e, T_e, U_e, V_e, zhdr)
        # documented keys + correct anisotropic shapes (default per-level pieces)
        assert res["psi_pieces"]["6"].shape == (nl, ny, nx)
        assert {"psi_pieces", "psi_total", "Q_event", "Q_mean"} <= set(res)
        # interior level (250 hPa, idx 6) substantially finite — the fill + zhdr
        # fix prevented the all-NaN blow-up the old anisotropic path produced.
        p250 = res["psi_pieces"]["6"][6]
        assert np.isfinite(p250).mean() > 0.5


# --- planetary/eddy scale split -------------------------------------------

class TestScaleSplit:
    """The split must be exactly additive, and must not need tuning."""

    @staticmethod
    def _field(seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        NL, NY, NX = 9, 60, 288
        q = rng.standard_normal((NL, NY, NX)) * 0.4
        jj, ii = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
        blob = -3.0 * np.exp(-(((jj - 25) / 6.) ** 2
                               + (((ii - 100 + 144) % 288 - 144) / 10.) ** 2))
        q[4:9] += blob
        th = rng.standard_normal((NY, NX)) * 2.0
        return q, th

    def test_pieces_sum_back_exactly(self):
        """Pass D is linear in its source, so the split is only legitimate if
        the pieces sum to the unsplit anomaly to machine precision."""
        import numpy as np
        from pvtend.ppvi import scale_split as ss

        q, th = self._field()
        upper, top = [4, 5, 6, 7], 8
        out = ss.split_at_box_minimum(q, th, upper, top,
                                      np.arange(5, 55), np.arange(60, 150))
        np.testing.assert_allclose((out["q_p"] + out["q_e"])[upper], q[upper],
                                   atol=1e-12)
        np.testing.assert_allclose(out["th_p"] + out["th_e"], th, atol=1e-12)

    def test_box_minimum_seed_never_needs_a_threshold(self):
        """The box holds a tracked anticyclone, so its minimum is negative and
        the flood fill is unconditional — there is no threshold to tune."""
        import numpy as np
        from pvtend.ppvi import scale_split as ss

        q, th = self._field(seed=3)
        upper, top = [4, 5, 6, 7], 8
        bl, bo = np.arange(5, 55), np.arange(60, 150)
        out = ss.split_at_box_minimum(q, th, upper, top, bl, bo)
        assert out["seed_source"] == "box_min"
        lev, j, i = out["seed"]
        assert out["mask"][lev, j, i], "the seed must lie inside its own object"

    def test_object_cannot_leave_the_box(self):
        """Cropping before the flood fill is what stops the k<=4 negative side
        linking through the subtropics into a hemispheric blob."""
        import numpy as np
        from pvtend.ppvi import scale_split as ss

        q, th = self._field(seed=5)
        upper, top = [4, 5, 6, 7], 8
        bl, bo = np.arange(5, 55), np.arange(60, 150)
        out = ss.split_at_box_minimum(q, th, upper, top, bl, bo)
        outside = np.ones(q.shape[-1], bool)
        outside[bo] = False
        assert not out["mask"][..., outside].any()

    def test_planetary_part_is_wavenumber_limited(self):
        """The re-filter after masking is the point: q'_k4 * M is not itself
        k<=4, because a sharp mask edge puts power back into every wavenumber."""
        import numpy as np
        from pvtend.ppvi import scale_split as ss

        q, th = self._field(seed=7)
        upper, top = [4, 5, 6, 7], 8
        out = ss.split_at_box_minimum(q, th, upper, top,
                                      np.arange(5, 55), np.arange(60, 150))
        F = np.abs(np.fft.rfft(out["q_p"][upper], axis=-1)) ** 2
        keep = F[..., ss.KMIN:ss.KMAX + 1].sum()
        assert keep / F.sum() > 0.999

    def test_contour_is_a_fraction_of_the_events_own_minimum(self):
        """The contour scales with the event, and the seed stays inside it.

        `frac * q_min` is negative and `q_min < frac * q_min` for any frac < 1,
        so the box minimum is inside its own contour by construction -- which is
        what keeps the flood fill unconditional after adding a threshold back.
        """
        import numpy as np
        from pvtend.ppvi import scale_split as ss

        q, th = self._field(seed=11)
        upper, top = [4, 5, 6, 7], 8
        bl, bo = np.arange(5, 55), np.arange(60, 150)

        out = ss.split_at_box_minimum(q, th, upper, top, bl, bo)
        assert out["thresh_used"] == pytest.approx(ss.OBJ_FRAC * out["q_min"])
        assert out["mask"][out["seed"]]

        # a tighter contour must give a strictly smaller object
        tight = ss.split_at_box_minimum(q, th, upper, top, bl, bo, frac=0.7)
        loose = ss.split_at_box_minimum(q, th, upper, top, bl, bo, frac=0.0)
        assert tight["mask"].sum() < out["mask"].sum() < loose["mask"].sum()


# ── the wrap seam ────────────────────────────────────────────────────
class TestFillSeamColumns:
    """pvpialln flags all four lateral boundaries (wuppvi.f:93-102).  On a pass
    over the whole circle the first and last columns are *neighbours* across the
    dateline, so the anomaly comes back with a notch there and zonal_filter
    transforms across it.
    """

    @staticmethod
    def _wave(nx=288, k=3):
        lam = np.linspace(0, 2 * np.pi, nx, endpoint=False)
        return np.sin(k * lam)[None, None, :] * np.ones((2, 3, 1))

    def test_flagged_columns_are_replaced(self):
        from pvtend.ppvi.scale_split import fill_seam_columns
        q = self._wave().copy()
        q[..., 0] = 0.0
        q[..., -1] = 0.0
        out = fill_seam_columns(q)
        assert np.abs(out[..., 0]).min() > 0
        assert np.abs(out[..., -1]).min() > 0

    def test_interior_is_untouched(self):
        from pvtend.ppvi.scale_split import fill_seam_columns
        q = self._wave().copy()
        q[..., 0] = q[..., -1] = 0.0
        np.testing.assert_array_equal(fill_seam_columns(q)[..., 1:-1], q[..., 1:-1])

    def test_recovers_a_smooth_wave_closely(self):
        """The notch is the error; the fill should mostly remove it."""
        from pvtend.ppvi.scale_split import fill_seam_columns
        truth = self._wave()
        notched = truth.copy()
        notched[..., 0] = notched[..., -1] = 0.0
        err_before = np.abs(notched - truth).max()
        err_after = np.abs(fill_seam_columns(notched) - truth).max()
        assert err_after < 0.25 * err_before

    def test_planetary_amplitude_stops_being_inflated(self):
        """The reason this matters: k<=4 power from a notch that is not real."""
        from pvtend.ppvi.scale_split import fill_seam_columns, zonal_filter
        truth = self._wave(k=3)
        notched = truth.copy()
        notched[..., 0] = notched[..., -1] = 0.0
        rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
        e_notched = rms(zonal_filter(notched) - zonal_filter(truth))
        e_filled = rms(zonal_filter(fill_seam_columns(notched)) - zonal_filter(truth))
        assert e_filled < e_notched

    def test_returns_a_copy(self):
        from pvtend.ppvi.scale_split import fill_seam_columns
        q = self._wave().copy()
        fill_seam_columns(q)
        assert q[..., 0] == pytest.approx(self._wave()[..., 0])

    def test_degenerate_width_is_a_noop(self):
        from pvtend.ppvi.scale_split import fill_seam_columns
        q = np.ones((1, 1, 3))
        np.testing.assert_array_equal(fill_seam_columns(q), q)


class TestDatelineBoxIsContiguous:
    """`split_at_box_minimum` floods with ndimage.label, which is not periodic.
    A box spanning the dateline must therefore arrive ordered around the circle,
    not sorted by index, or the object is cut in half and a false adjacency is
    created between longitudes 240 deg apart.
    """

    NLON, DLON, LON_HALF = 288, 1.25, 60.0

    def _box(self, clon):
        from pvtend.tendency import _circ_nearest_lon, _wrapped_lon_index
        lon = np.arange(-180.0, 180.0, self.DLON)
        return lon, _wrapped_lon_index(
            _circ_nearest_lon(lon, clon),
            LON_PAD=int(round(self.LON_HALF / self.DLON)), nlon=self.NLON)

    def test_box_is_circularly_contiguous_at_the_dateline(self):
        _, box = self._box(175.0)
        step = np.diff(box) % self.NLON
        assert set(np.unique(step)) == {1}, "columns must be adjacent on the circle"

    def test_box_holds_the_right_longitudes(self):
        lon, box = self._box(175.0)
        d = np.abs((lon[box] - 175.0 + 180.0) % 360.0 - 180.0)
        assert d.max() <= self.LON_HALF + 0.5 * self.DLON

    def test_a_seam_crossing_object_stays_one_component(self):
        from pvtend.ppvi.scale_split import component_containing
        lon, box = self._box(178.0)
        q = np.zeros((1, 3, self.NLON))
        blob = np.abs((lon - 178.0 + 180.0) % 360.0 - 180.0) <= 10.0
        q[:, :, blob] = -1.0
        sub = q[:, :, box] < -0.5
        seed = int(np.nonzero(box == int(np.nonzero(blob)[0][len(np.nonzero(blob)[0]) // 2]))[0][0])
        comp = component_containing(sub, 0, 1, seed)
        assert comp.sum() == sub.sum(), "the blob must not be split by the seam"

    def test_sorted_box_would_have_split_it(self):
        """Guard the guard: the old boolean-mask ordering really does fail."""
        from pvtend.ppvi.scale_split import component_containing
        lon = np.arange(-180.0, 180.0, self.DLON)
        old_box = np.nonzero(
            np.abs((lon - 178.0 + 180.0) % 360.0 - 180.0) <= self.LON_HALF)[0]
        q = np.zeros((1, 3, self.NLON))
        blob = np.abs((lon - 178.0 + 180.0) % 360.0 - 180.0) <= 10.0
        q[:, :, blob] = -1.0
        sub = q[:, :, old_box] < -0.5
        seed = int(np.nonzero(old_box == int(np.nonzero(blob)[0][0]))[0][0])
        comp = component_containing(sub, 0, 1, seed)
        assert comp.sum() < sub.sum(), "expected the old ordering to split the object"


class TestSeedNearCentre:
    """The object must be the TRACKED feature, not whatever is deepest in the
    +/-60 x +/-30 box. Measured on 30+30 m091 events: a blocking high IS its box
    minimum, a propagating high is not, and the old policy then inverted an
    unrelated system a median 4193 km away while every sum still balanced.
    """

    NL, NY, NX = 9, 40, 288
    UPPER, TOP = [4, 5, 6, 7], 8

    def _field(self):
        """A shallow anomaly at the tracked centre, a DEEPER one far away."""
        q = np.zeros((self.NL, self.NY, self.NX))
        lon = np.arange(self.NX) * (360.0 / self.NX) - 180.0
        for k in self.UPPER:
            for j in range(self.NY):
                # the event: modest, centred at lon 0, row 20
                q[k] += -60.0 * np.exp(-(((lon - 0.0) / 18.0) ** 2))[None, :] \
                    * np.exp(-(((np.arange(self.NY) - 20) / 5.0) ** 2))[:, None]
                # a deeper distractor 100 deg away, inside the same box
                q[k] += -200.0 * np.exp(-(((lon + 100.0) / 18.0) ** 2))[None, :] \
                    * np.exp(-(((np.arange(self.NY) - 22) / 5.0) ** 2))[:, None]
                break
        for k in self.UPPER[1:]:
            q[k] = q[self.UPPER[0]]
        return q

    @property
    def _box(self):
        return np.arange(5, 35), np.arange(self.NX)   # box holds both features

    def test_box_min_seed_lands_on_the_distractor(self):
        """Guard the guard: the old policy really does pick the wrong feature."""
        from pvtend.ppvi.scale_split import seed_from_box_min, zonal_filter
        q = self._field(); bl, bo = self._box
        _, _, i = seed_from_box_min(zonal_filter(q, 1, 4), self.UPPER, bl, bo)
        lon_i = i * (360.0 / self.NX) - 180.0
        assert abs(lon_i + 100.0) < 25.0, f"expected the distractor, got lon {lon_i}"

    def test_near_centre_seed_lands_on_the_event(self):
        from pvtend.ppvi.scale_split import seed_near_centre, zonal_filter
        q = self._field(); bl, bo = self._box
        ic = int(np.argmin(np.abs(np.arange(self.NX)*(360.0/self.NX)-180.0)))
        _, _, i = seed_near_centre(zonal_filter(q, 1, 4), self.UPPER, bl, bo,
                                   centre_lat=20, centre_lon=ic,
                                   halo_lat=10, halo_lon=16)
        lon_i = i * (360.0 / self.NX) - 180.0
        assert abs(lon_i) < 25.0, f"expected the tracked event, got lon {lon_i}"

    def test_contour_scales_to_the_LOCAL_minimum(self):
        """0.35 x box min would be deeper than the event and reject it."""
        from pvtend.ppvi.scale_split import split_near_centre, split_at_box_minimum
        q = self._field(); bl, bo = self._box
        th = np.zeros((self.NY, self.NX))
        ic = int(np.argmin(np.abs(np.arange(self.NX)*(360.0/self.NX)-180.0)))
        near = split_near_centre(q, th, self.UPPER, self.TOP, bl, bo,
                                 centre_lat=20, centre_lon=ic,
                                 halo_lat=10, halo_lon=16)
        boxm = split_at_box_minimum(q, th, self.UPPER, self.TOP, bl, bo)
        assert abs(near["q_min"]) < abs(boxm["q_min"]), "local min must be shallower"
        assert near["seed_source"] == "near_centre"

    def test_mask_covers_the_tracked_centre(self):
        from pvtend.ppvi.scale_split import split_near_centre
        q = self._field(); th = np.zeros((self.NY, self.NX)); bl, bo = self._box
        ic = int(np.argmin(np.abs(np.arange(self.NX)*(360.0/self.NX)-180.0)))
        out = split_near_centre(q, th, self.UPPER, self.TOP, bl, bo,
                                centre_lat=20, centre_lon=ic,
                                halo_lat=10, halo_lon=16)
        assert out["mask"][self.UPPER][:, 20, ic].any()

    def test_pieces_still_sum_to_the_total(self):
        from pvtend.ppvi.scale_split import split_near_centre
        q = self._field(); th = np.zeros((self.NY, self.NX)); bl, bo = self._box
        ic = int(np.argmin(np.abs(np.arange(self.NX)*(360.0/self.NX)-180.0)))
        out = split_near_centre(q, th, self.UPPER, self.TOP, bl, bo,
                                centre_lat=20, centre_lon=ic,
                                halo_lat=10, halo_lon=16)
        np.testing.assert_allclose(out["q_p"] + out["q_e"], q, atol=1e-9)

    def test_centre_outside_the_box_is_refused(self):
        from pvtend.ppvi.scale_split import seed_near_centre, zonal_filter
        q = self._field(); bl, bo = self._box
        with pytest.raises(ValueError, match="not inside the event box"):
            seed_near_centre(zonal_filter(q, 1, 4), self.UPPER, bl, bo,
                             centre_lat=999, centre_lon=0,
                             halo_lat=10, halo_lon=16)

    def test_no_negative_anomaly_near_the_centre_is_refused(self):
        from pvtend.ppvi.scale_split import seed_near_centre
        q = np.ones((self.NL, self.NY, self.NX)) * 5.0
        bl, bo = self._box
        with pytest.raises(ValueError, match="no negative"):
            seed_near_centre(q, self.UPPER, bl, bo, centre_lat=20, centre_lon=10,
                             halo_lat=4, halo_lon=4)


# ── wall piece (lateral-boundary contribution, scale-mode production) ──
@pytest.mark.skipif(not _ext_available(),
                    reason="locally built _wuppvi extension unavailable")
class TestWallPiece:
    """The zero-source ibc=1 solve isolates the shared wall response W.

    On a cropped window the lateral ψ′ boundary values carry the far-field
    (outside-PV) flow. With ibc=1 every piece solve carries those values, so
    the corrected decomposition is piece_k(ibc=1) − W, with W stored once —
    the construction the production scale mode uses.
    """

    @staticmethod
    def _states():
        nl, ny, nx = 9, 40, 50
        dlat = dlon = 1.5
        lat_n = 78.5
        latv = lat_n - np.arange(ny) * dlat
        p = np.asarray(PR) * 1000.0
        T_prof = np.array([288., 279., 269., 252., 240., 229., 222., 217., 210.])
        Rd, g = 287.05, 9.80665
        Hp = np.empty(nl)
        Hp[0] = 100.0
        for k in range(1, nl):
            Hp[k] = Hp[k - 1] + Rd * 0.5 * (T_prof[k] + T_prof[k - 1]) / g \
                * np.log(p[k - 1] / p[k])
        ones = np.ones((ny, nx))
        H_m = Hp[:, None, None] * ones
        T_m = T_prof[:, None, None] * ones
        U_m = np.broadcast_to(
            10.0 * np.cos(np.deg2rad(latv))[None, :, None],
            (nl, ny, nx)).astype(float).copy()
        V_m = np.zeros((nl, ny, nx))
        # Dynamically consistent perturbation with a deliberate nonzero
        # lateral-boundary component: streamfunction = centred vortex + a
        # meridional ramp (band-mean zonal-flow anomaly reaching the walls),
        # winds geostrophic from it, H from f0*psi/g, T hydrostatic from H.
        xg = np.exp(-((np.arange(nx) - nx / 2) ** 2) / 60.0)[None, :]
        yg = np.exp(-((latv - latv[ny // 2]) ** 2) / 100.0)[:, None]
        ramp = np.linspace(0.0, 1.0, ny)[:, None]
        psi2d = 4.0e6 * (xg * yg + 0.5 * ramp)          # [m^2/s]
        vshape = np.linspace(0.2, 1.0, nl)[:, None, None]
        psi = vshape * psi2d[None, :, :]
        dy = dlat * 111.2e3
        dx = dlon * 111.2e3 * np.cos(np.deg2rad(latv))[None, :, None]
        up = -np.gradient(psi, axis=1) / (-dy)          # rows run N->S
        vp = np.gradient(psi, axis=2) / dx
        f0 = 2 * 7.292e-5 * np.sin(np.deg2rad(latv))[None, :, None]
        Hp3 = f0 * psi / g
        lnp = np.log(p)
        Tp = np.zeros_like(Hp3)
        for k in range(1, nl - 1):
            Tp[k] = -(g / Rd) * (Hp3[k + 1] - Hp3[k - 1]) / (lnp[k + 1] - lnp[k - 1])
        Tp[0], Tp[-1] = Tp[1], Tp[-2]
        H_e = H_m + Hp3
        T_e = T_m + Tp
        U_e = U_m + up
        V_e = V_m + vp
        zhdr = np.array([latv[-1], 0.0, latv[0], (nx - 1) * dlon,
                         dlat, dlon, nx, ny], dtype=np.float32)
        return H_m, T_m, U_m, V_m, H_e, T_e, U_e, V_e, zhdr

    _PIECES = {"low": [1, 2, 3, 4], "up": [5, 6, 7, 8, 9]}

    @classmethod
    def _invert(cls, ibc, with_wall):
        from pvtend.ppvi.solver import invert_piecewise, PassDParams
        H_m, T_m, U_m, V_m, H_e, T_e, U_e, V_e, zhdr = cls._states()
        pieces = dict(cls._PIECES)
        qp = th = None
        if with_wall:
            pieces["wall"] = [2]
            qp = {"wall": np.zeros_like(H_m)}
            th = {"wall": np.zeros(H_m.shape[1:] + (2,))}
        return invert_piecewise(
            H_m, T_m, U_m, V_m, H_e, T_e, U_e, V_e, zhdr,
            pieces=pieces, qp_anoms=qp, th_anoms=th,
            pd=PassDParams(ibc=ibc))

    def test_zero_source_ibc0_solve_is_zero(self):
        res = self._invert(ibc=0, with_wall=True)
        wall = res["psi_pieces"]["wall"]
        ref = res["psi_pieces"]["up"]
        assert np.nanmax(np.abs(wall)) < 1e-3 * np.nanmax(np.abs(ref))

    def test_wall_correction_improves_closure(self):
        """The corrected sum Σpieces(ibc=1) − (N−1)·W beats ibc=0 against the
        constructed truth, and recovers the band-mean flow ibc=0 cannot carry.

        The constructed perturbation is known exactly (geostrophic winds from
        the prescribed streamfunction), so closure is measured against truth,
        not against another solve.
        """
        from pvtend.ppvi.winds import psi_to_winds
        H_m, T_m, U_m, V_m, H_e, T_e, U_e, V_e, zhdr = self._states()
        ny = H_m.shape[1]
        dlat = dlon = 1.5
        latv = 78.5 - np.arange(ny) * dlat
        res0 = self._invert(ibc=0, with_wall=False)
        res1 = self._invert(ibc=1, with_wall=True)
        W = res1["psi_pieces"]["wall"]
        assert np.nanmax(np.abs(W)) > 0
        n = len(self._PIECES)
        s0 = sum(res0["psi_pieces"][k] for k in self._PIECES)
        s1c = sum(res1["psi_pieces"][k] for k in self._PIECES) - (n - 1) * W
        u_t = U_e - U_m
        u0, _ = psi_to_winds(s0, latv, dlat, dlon)
        u1, _ = psi_to_winds(s1c, latv, dlat, dlon)
        core = (slice(2, 7), slice(8, -8), slice(8, -8))

        def _nrms(a):
            return float(np.sqrt(np.nanmean((a[core] - u_t[core]) ** 2))
                         / np.sqrt(np.nanmean(u_t[core] ** 2)))

        # measured on this fixture: 0.30 → 0.06
        assert _nrms(u1) < 0.15
        assert _nrms(u1) < 0.5 * _nrms(u0)
        # band-mean (ramp) flow: ibc=0 loses it entirely (measured −8%),
        # the corrected sum restores it (measured 102%)
        mt = float(np.nanmean(u_t[core]))
        assert abs(float(np.nanmean(u0[core])) / mt) < 0.5
        assert 0.7 < float(np.nanmean(u1[core])) / mt < 1.3
