"""Wu piecewise potential-vorticity inversion (PPVI) for pvtend.

This subpackage ports the Wu/Davis serial Gauss–Seidel SOR inversion
solver (``pvpialln`` → ``qinvert21`` → ``qinvertp21``) into pure,
I/O-free Fortran cores compiled with f2py into the ``_wuppvi`` extension.

The extension is **built locally on the HPC host only** (it is not part of
the pure-Python PyPI wheel).  It is imported lazily via :func:`load_ext`,
so importing :mod:`pvtend` never requires the compiled module unless the
``ppvi`` functionality is actually used.

Public API
----------
- :func:`pvtend.ppvi.solver.invert_piecewise` — chain the three cores.
- :func:`pvtend.ppvi.winds.psi_to_winds` — ψ → (u_rot, v_rot).
"""
from __future__ import annotations

from .solver import (
    PR,
    PIECES,
    PassABParams,
    PassCParams,
    PassDParams,
    invert_piecewise,
)
from .winds import psi_to_winds

__all__ = [
    "PR",
    "PIECES",
    "PassABParams",
    "PassCParams",
    "PassDParams",
    "invert_piecewise",
    "psi_to_winds",
]
