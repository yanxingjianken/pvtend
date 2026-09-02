"""Piecewise potential-vorticity inversion (PPVI) for pvtend.

The inversion is global: the balanced rotational-wind anomaly of an event is
attributed to its PV and boundary-θ sources on the closed sphere, so there is
no lateral boundary and no piece standing for one.

Modules
-------
- :mod:`pvtend.ppvi.spherical` — the spectral Newton–Krylov solver, vendored
  verbatim from ``pv_inversion_spherical`` (see ``spherical/VENDORED.md``);
  it is re-copied from there rather than edited here.
- :mod:`pvtend.ppvi.spherical_engine` — the adapter that puts an archive
  hemisphere on the solver's grid and brings the pieces back.
- :mod:`pvtend.ppvi.scale_split` — the planetary/eddy split of a PV anomaly
  (zonal k1–4 inside the tracked object), pure numpy and independent of any
  solver, used for the archived PV the NPZ records carry.
"""
from __future__ import annotations
