"""Packing spectra into the real vectors a Krylov solver works on.

A real field has ``(lmax+1)**2`` independent spectral degrees of freedom: the
zonal-mean coefficients are real, and every other coefficient contributes a real
and an imaginary part.  Packing exactly those -- rather than handing the solver
complex arrays with redundant or constrained entries -- keeps the operator
linear over the reals, which is what GMRES assumes, and stops the solver from
wandering into components that describe no field.
"""
from __future__ import annotations

import numpy as np


class SpectralPacker:
    """Flatten and rebuild ``(nlev, lmax+1, lmax+1)`` spectra for two fields.

    The state is ``(phi, psi)``; both are stored on the same interior levels.
    """

    def __init__(self, nlev: int, lmax: int, nfields: int = 2):
        self.nlev = int(nlev)
        self.lmax = int(lmax)
        self.nfields = int(nfields)
        m_idx, n_idx = np.meshgrid(
            np.arange(lmax + 1), np.arange(lmax + 1), indexing="ij"
        )
        valid = m_idx <= n_idx
        self._re_mask = valid
        self._im_mask = valid & (m_idx > 0)
        self.n_re = int(self._re_mask.sum())
        self.n_im = int(self._im_mask.sum())
        #: Real degrees of freedom per level per field.
        self.per_level = self.n_re + self.n_im
        assert self.per_level == (lmax + 1) ** 2
        self.size = self.nfields * self.nlev * self.per_level

    def pack(self, *fields: np.ndarray) -> np.ndarray:
        """Spectra to a flat real vector."""
        if len(fields) != self.nfields:
            raise ValueError(f"expected {self.nfields} fields, got {len(fields)}")
        out = np.empty(self.size, dtype=np.float64)
        stride = self.nlev * self.per_level
        for f, spec in enumerate(fields):
            spec = np.asarray(spec)
            if spec.shape != (self.nlev, self.lmax + 1, self.lmax + 1):
                raise ValueError(
                    f"field {f} has shape {spec.shape}, expected "
                    f"{(self.nlev, self.lmax + 1, self.lmax + 1)}"
                )
            block = out[f * stride : (f + 1) * stride].reshape(self.nlev, -1)
            block[:, : self.n_re] = spec.real[:, self._re_mask]
            block[:, self.n_re :] = spec.imag[:, self._im_mask]
        return out

    def unpack(self, vec: np.ndarray) -> tuple[np.ndarray, ...]:
        """Flat real vector back to spectra."""
        vec = np.asarray(vec, dtype=np.float64).ravel()
        if vec.size != self.size:
            raise ValueError(f"vector has {vec.size} entries, expected {self.size}")
        stride = self.nlev * self.per_level
        fields = []
        for f in range(self.nfields):
            block = vec[f * stride : (f + 1) * stride].reshape(self.nlev, -1)
            spec = np.zeros(
                (self.nlev, self.lmax + 1, self.lmax + 1), dtype=np.complex128
            )
            real = np.zeros((self.nlev, self.lmax + 1, self.lmax + 1))
            imag = np.zeros_like(real)
            real[:, self._re_mask] = block[:, : self.n_re]
            imag[:, self._im_mask] = block[:, self.n_re :]
            spec.real = real
            spec.imag = imag
            fields.append(spec)
        return tuple(fields)

    def canonicalize(self, spec: np.ndarray) -> np.ndarray:
        """Drop the components that describe no real field.

        Zeroes ``m > n`` and the imaginary part of the zonal mean.  Applied to
        anything entering the solver, so a stray value in an unused slot cannot
        show up as a phantom degree of freedom.
        """
        out = np.where(self._re_mask, spec, 0.0)
        out[..., 0, :] = out[..., 0, :].real
        return out
