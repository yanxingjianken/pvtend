"""Tests for adiabatic/diabatic omega decomposition utilities."""

from __future__ import annotations

import numpy as np
import pytest

from pvtend.moist_dry import verify_div_additivity


class TestVerifyDivAdditivity:
    """Test the divergent-wind additivity diagnostic."""

    def test_exact_split(self):
        """If adiabatic + diabatic == total, max error is 0."""
        rng = np.random.default_rng(42)
        u_div = rng.standard_normal((5, 21, 41))
        frac = rng.uniform(0.3, 0.7, u_div.shape)
        u_div_adiabatic = u_div * frac
        u_div_diabatic = u_div * (1 - frac)
        err = verify_div_additivity(u_div, u_div_adiabatic, u_div_diabatic)
        assert err == pytest.approx(0.0, abs=1e-14)

    def test_nonzero_residual(self):
        """Perturbed split should yield a positive error."""
        rng = np.random.default_rng(99)
        u_div = rng.standard_normal((3, 10, 20))
        u_div_adiabatic = u_div * 0.6
        u_div_diabatic = u_div * 0.3  # deliberate imbalance
        err = verify_div_additivity(u_div, u_div_adiabatic, u_div_diabatic)
        assert err > 0.0

    def test_scalar_input(self):
        """Works for single scalar values."""
        err = verify_div_additivity(
            np.array(1.0), np.array(0.4), np.array(0.6)
        )
        assert err == pytest.approx(0.0, abs=1e-15)
