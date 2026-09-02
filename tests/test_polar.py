"""The continuation of a patch across the pole (pvtend.polar)."""
from __future__ import annotations

import numpy as np
import pytest

from pvtend.polar import (
    far_side_rows,
    geographic_latitude,
    negates_across_pole,
    plan_continuation,
)


class TestSignRule:
    @pytest.mark.parametrize("key", [
        "u", "v", "u_3d", "u_bar", "v_anom_3d", "u_rot", "v_rot_anom", "u_div_adiabatic",
        "v_har_anom", "u_rot_anom_ppvi_upper_p_3d", "v_rot_anom_residual_ppvi",
        "u_rot_anom_sph_3d", "pv_bar_dx", "pv_anom_dy_3d", "pv_total_dx", "pv_dx_3d",
    ])
    def test_vector_components_and_first_derivatives_negate(self, key):
        assert negates_across_pole(key)

    @pytest.mark.parametrize("key", [
        "pv", "pv_anom_3d", "z", "t", "q", "w", "w_anom", "w_adiabatic_3d", "Q",
        "theta", "theta_dot", "pv_bar_dp", "pv_anom_dt", "pv_dx_dx_3d", "pv_dx_dy",
        "pv_dy_dy", "u_anom_pv_bar_dx_3d", "v_bar_pv_anom_dy", "u_div_adiabatic_pv_anom_dx",
        "w_anom_pv_bar_dp", "u_rot_anom_pv_bar_dx_3d", "pv_split_mask_3d", "phi_3d", "z_m_3d",
    ])
    def test_scalars_products_and_second_derivatives_do_not(self, key):
        assert not negates_across_pole(key)


class TestPlan:
    ERA5 = np.arange(90.0, -0.1, -1.5)                       # north first, 61 rows
    F09 = -90.0 + np.arange(192) * (180.0 / 191.0)          # south first, pole last

    def test_no_rows_past_the_pole_means_no_plan(self):
        assert plan_continuation(self.ERA5, 240, 20, 20, 20, 1.5) is None

    def test_era5_rows_are_mirrored_onto_the_antimeridian(self):
        # Centre at 82.5 N: 5 rows to the pole, 15 rows past it.
        plan = plan_continuation(self.ERA5, 240, 20, 5, 20, 1.5)
        assert plan is not None
        assert plan.col_shift == 120
        np.testing.assert_array_equal(plan.slots, np.arange(26, 41))
        np.testing.assert_allclose(plan.nominal_lat, 90.0 + 1.5 * np.arange(1, 16))
        np.testing.assert_allclose(self.ERA5[plan.src_rows], 90.0 - 1.5 * np.arange(1, 16))
        np.testing.assert_array_equal(plan.far_columns(np.array([0, 5, 239]), 240),
                                      [120, 125, 119])

    def test_f09_rows_are_mirrored_too(self):
        dlat = 180.0 / 191.0
        ilat = 185                                            # 84.35 N
        eff_north = 191 - ilat
        plan = plan_continuation(self.F09, 288, 32, eff_north, 32, dlat)
        assert plan is not None and plan.col_shift == 144
        assert plan.slots.size == 32 - eff_north
        np.testing.assert_allclose(self.F09[plan.src_rows],
                                   90.0 - dlat * np.arange(1, plan.slots.size + 1))

    def test_a_grid_without_a_pole_row_is_not_continued(self):
        lat = np.arange(0.5, 90.0, 1.0)                       # 0.5 .. 89.5
        assert plan_continuation(lat, 360, 30, 5, 30, 1.0) is None

    def test_odd_longitude_count_is_not_continued(self):
        assert plan_continuation(self.ERA5, 241, 20, 5, 20, 1.5) is None


def test_geographic_latitude_and_far_rows():
    lat_vec = np.array([np.nan, 60.0, 88.5, 90.0, 91.5, 100.5])
    np.testing.assert_allclose(geographic_latitude(lat_vec)[1:], [60.0, 88.5, 90.0, 88.5, 79.5])
    np.testing.assert_array_equal(far_side_rows(lat_vec), [False, False, False, False, True, True])
