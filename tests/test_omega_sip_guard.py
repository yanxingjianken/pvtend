"""The divergence guard of the SIP omega solver.

Stone's procedure at alpha 0.93 diverges on some events and returns a
complete-looking field of 1e38 Pa/s with no error.  The guard judges the solve
by its final residual ratio and restarts it at a smaller alpha from the same
initial state.  A solve that converged is not touched, so wherever the
unguarded solver converged the output is byte-identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvtend import omega as omega_mod


def _inputs(nlev=9, nlat=13, nlon=16):
    lat = np.linspace(20.0, 80.0, nlat)
    lon = np.arange(nlon) * (360.0 / nlon)
    plevs_pa = np.array([1000, 850, 700, 500, 400, 300, 250, 200, 100], float)[:nlev] * 100.0
    lon2d, lat2d = np.meshgrid(lon, lat)
    t = np.stack([280.0 - 6.0 * k + 3.0 * np.cos(np.radians(lat2d)) for k in range(nlev)])
    u = np.stack([(5.0 + k) * np.cos(np.radians(lat2d)) + np.sin(np.radians(2 * lon2d))
                  for k in range(nlev)])
    v = np.stack([np.cos(np.radians(3 * lon2d)) * np.cos(np.radians(lat2d)) for _ in range(nlev)])
    return u, v, t, lat, lon, plevs_pa


class _FakeCore:
    """Stands in for ``_sip_core``: diverges at 0.93, converges at anything smaller."""

    def __init__(self):
        self.calls: list[tuple[float, int]] = []

    def __call__(self, AP, AE, AW, AN, AS, AT, AB, Q, T, Nk, Nj, Ni, alpha, maxit,
                 resmax, periodic_lon=1):
        self.calls.append((float(alpha), int(maxit)))
        if alpha > 0.9:
            T[1:-1, 1:-1, :] += 1.0e30          # the runaway mode
            return maxit, 7.1e44
        T[1:-1, 1:-1, :] += 0.5                 # a converged, finite answer
        return 42, 3.0e-15


def test_diverged_solve_is_retried_at_the_smaller_alpha(monkeypatch):
    fake = _FakeCore()
    monkeypatch.setattr(omega_mod, "_sip_core", fake)
    u, v, t, lat, lon, plevs = _inputs()
    out, info = omega_mod.solve_qg_omega_sip(u, v, t, lat, lon, plevs, center_lat=50.0,
                                             bc_top=0.0, bc_bot=0.0)
    assert fake.calls == [(0.93, 300), (0.5, 600)]
    assert info["retried"] is True
    assert info["alpha"] == 0.5
    assert info["diverged"] is False
    assert info["iters"] == 42 and info["final_residual"] == pytest.approx(3.0e-15)
    assert info["first_final_residual"] == pytest.approx(7.1e44)
    assert np.all(np.isfinite(out)) and np.nanmax(np.abs(out)) < 10.0


def test_converged_solve_is_left_alone(monkeypatch):
    fake = _FakeCore()
    monkeypatch.setattr(omega_mod, "_sip_core", fake)
    u, v, t, lat, lon, plevs = _inputs()
    out, info = omega_mod.solve_qg_omega_sip(u, v, t, lat, lon, plevs, center_lat=50.0,
                                             bc_top=0.0, bc_bot=0.0, alpha=0.8)
    assert fake.calls == [(0.8, 300)]
    assert info["retried"] is False and info["diverged"] is False
    assert info["alpha"] == 0.8


def test_no_retry_reports_the_divergence(monkeypatch):
    fake = _FakeCore()
    monkeypatch.setattr(omega_mod, "_sip_core", fake)
    u, v, t, lat, lon, plevs = _inputs()
    _, info = omega_mod.solve_qg_omega_sip(u, v, t, lat, lon, plevs, center_lat=50.0,
                                           bc_top=0.0, bc_bot=0.0, retry_alpha=None)
    assert fake.calls == [(0.93, 300)]
    assert info["retried"] is False and info["diverged"] is True


def test_real_core_converges_on_a_smooth_state():
    """The guard must not fire on a healthy solve of the real core."""
    u, v, t, lat, lon, plevs = _inputs()
    out, info = omega_mod.solve_qg_omega_sip(u, v, t, lat, lon, plevs, center_lat=50.0,
                                             bc_top=0.0, bc_bot=0.0)
    assert info["retried"] is False and info["diverged"] is False
    assert info["final_residual"] < 1.0
    assert np.all(np.isfinite(out))
