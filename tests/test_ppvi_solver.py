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
