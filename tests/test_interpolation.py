"""Tests for lerp_fields."""

from __future__ import annotations

import numpy as np
import pytest

from pvtend.decomposition.interpolation import lerp_fields
from pvtend.decomposition.basis import compute_orthogonal_basis


class TestLerpFields:
    """Linear interpolation between two snapshots."""

    def test_alpha_zero_returns_prev(self):
        prev = {"pv_anom": np.ones((5, 5)), "pv_dx": np.zeros((5, 5)),
                "pv_dy": np.full((5, 5), 2.0)}
        curr = {"pv_anom": np.full((5, 5), 10.0), "pv_dx": np.ones((5, 5)),
                "pv_dy": np.full((5, 5), 8.0)}
        out = lerp_fields(prev, curr, alpha=0.0)
        np.testing.assert_array_equal(out["pv_anom"], prev["pv_anom"])
        np.testing.assert_array_equal(out["pv_dx"], prev["pv_dx"])
        np.testing.assert_array_equal(out["pv_dy"], prev["pv_dy"])

    def test_alpha_one_returns_curr(self):
        prev = {"pv_anom": np.zeros((5, 5)), "pv_dx": np.zeros((5, 5)),
                "pv_dy": np.zeros((5, 5))}
        curr = {"pv_anom": np.ones((5, 5)), "pv_dx": np.full((5, 5), 3.0),
                "pv_dy": np.full((5, 5), 7.0)}
        out = lerp_fields(prev, curr, alpha=1.0)
        np.testing.assert_array_equal(out["pv_anom"], curr["pv_anom"])

    def test_alpha_075_linear(self):
        prev = {"pv_anom": np.zeros((3, 3)), "pv_dx": np.zeros((3, 3)),
                "pv_dy": np.zeros((3, 3))}
        curr = {"pv_anom": np.full((3, 3), 4.0), "pv_dx": np.full((3, 3), 8.0),
                "pv_dy": np.full((3, 3), 12.0)}
        out = lerp_fields(prev, curr, alpha=0.75)
        np.testing.assert_allclose(out["pv_anom"], 3.0)
        np.testing.assert_allclose(out["pv_dx"], 6.0)
        np.testing.assert_allclose(out["pv_dy"], 9.0)

    def test_invalid_alpha_raises(self):
        d = {"pv_anom": np.zeros((2, 2)), "pv_dx": np.zeros((2, 2)),
             "pv_dy": np.zeros((2, 2))}
        with pytest.raises(ValueError, match="alpha must be in"):
            lerp_fields(d, d, alpha=1.5)
        with pytest.raises(ValueError, match="alpha must be in"):
            lerp_fields(d, d, alpha=-0.1)

    def test_custom_keys(self):
        prev = {"foo": np.array([1.0, 2.0])}
        curr = {"foo": np.array([3.0, 4.0])}
        out = lerp_fields(prev, curr, alpha=0.5, keys=("foo",))
        np.testing.assert_allclose(out["foo"], [2.0, 3.0])


class TestBasisNextAPI:
    """Tests for _next temporal interpolation in compute_orthogonal_basis."""

    @staticmethod
    def _make_fields(ny=29, nx=49):
        """Create plausible synthetic PV fields for testing."""
        rng = np.random.default_rng(42)
        x_rel = np.linspace(-36, 36, nx)
        y_rel = np.linspace(-21, 21, ny)
        X, Y = np.meshgrid(x_rel, y_rel)
        pv_anom = -1e-5 * np.exp(-(X**2 + Y**2) / 200)
        pv_dx = rng.normal(0, 1e-11, (ny, nx))
        pv_dy = rng.normal(0, 1e-11, (ny, nx))
        return pv_anom, pv_dx, pv_dy, x_rel, y_rel

    def test_next_equivalent_to_manual_lerp(self):
        """_next API must produce identical basis to manual lerp_fields."""
        prev_anom, prev_dx, prev_dy, x_rel, y_rel = self._make_fields()
        rng = np.random.default_rng(99)
        next_anom = prev_anom + rng.normal(0, 1e-6, prev_anom.shape)
        next_dx = prev_dx + rng.normal(0, 1e-12, prev_dx.shape)
        next_dy = prev_dy + rng.normal(0, 1e-12, prev_dy.shape)

        # Manual lerp then basis
        alpha = 0.75
        lerped = lerp_fields(
            {"pv_anom": prev_anom, "pv_dx": prev_dx, "pv_dy": prev_dy},
            {"pv_anom": next_anom, "pv_dx": next_dx, "pv_dy": next_dy},
            alpha=alpha,
        )
        basis_manual = compute_orthogonal_basis(
            lerped["pv_anom"], lerped["pv_dx"], lerped["pv_dy"],
            x_rel, y_rel, apply_smoothing=False,
        )

        # _next API
        basis_auto = compute_orthogonal_basis(
            prev_anom, prev_dx, prev_dy, x_rel, y_rel,
            apply_smoothing=False,
            pv_anom_next=next_anom,
            pv_dx_next=next_dx,
            pv_dy_next=next_dy,
            interp_alpha=alpha,
        )

        np.testing.assert_allclose(basis_auto.phi_int, basis_manual.phi_int, atol=1e-15)
        np.testing.assert_allclose(basis_auto.phi_dx, basis_manual.phi_dx, atol=1e-15)
        np.testing.assert_allclose(basis_auto.phi_dy, basis_manual.phi_dy, atol=1e-15)
        np.testing.assert_allclose(basis_auto.phi_def, basis_manual.phi_def, atol=1e-15)

    def test_backward_compat_no_next(self):
        """Without _next fields the result is identical to the original API."""
        pv_anom, pv_dx, pv_dy, x_rel, y_rel = self._make_fields()
        basis = compute_orthogonal_basis(
            pv_anom, pv_dx, pv_dy, x_rel, y_rel, apply_smoothing=False,
        )
        assert basis.phi_int.shape == (29, 49)
        assert basis.norms["beta"] > 0

    def test_partial_next_raises(self):
        """Providing only 1 or 2 of the 3 _next fields must raise ValueError."""
        pv_anom, pv_dx, pv_dy, x_rel, y_rel = self._make_fields()
        with pytest.raises(ValueError, match="all three"):
            compute_orthogonal_basis(
                pv_anom, pv_dx, pv_dy, x_rel, y_rel,
                pv_anom_next=pv_anom,  # only 1 of 3
            )
        with pytest.raises(ValueError, match="all three"):
            compute_orthogonal_basis(
                pv_anom, pv_dx, pv_dy, x_rel, y_rel,
                pv_anom_next=pv_anom,
                pv_dx_next=pv_dx,     # 2 of 3
            )
