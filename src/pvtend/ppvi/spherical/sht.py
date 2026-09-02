"""Double-precision spherical-harmonic transforms.

The inversion is solved by a preconditioned Krylov method driven to residuals
around 1e-8, and a transform that rounds to single precision anywhere along the
way cannot deliver that: its own rounding then sits an order of magnitude above
the tolerance, and the iteration stalls there instead of converging.  So this is
a self-contained float64 core -- an FFT in longitude and stored normalised
associated-Legendre matrices in latitude -- with no compiled dependency, which
also keeps worker start-up cheap in a process pool.

Two grid roles, deliberately kept apart:

*Data grids* carry the input and output fields and may include the poles.  Only
scalar analysis and synthesis happen there, so no metric factor is ever divided
by.  Their latitudes are arbitrary, and analysis is a weighted least-squares fit
(:class:`SHT` builds the pseudo-inverse once), which recovers band-limited fields
to machine precision.

The *solver grid* is Gaussian.  It carries every product, coefficient and
derivative, and excludes the poles by construction, so the ``1/cos(lat)`` factors
in the horizontal derivatives are never evaluated at a singular point.  Moving
between the two grids is an exact spectral operation for band-limited fields.

Conventions: latitudes ascend (south to north), longitudes are ``[0, 360)``,
harmonics are 4-pi normalised without the Condon-Shortley phase, and spectra are
packed as ``spec[..., m, n]`` for ``0 <= m <= n <= lmax`` (upper-triangular,
unused entries held at zero).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from .levels import R_EARTH

GridKind = Literal["gaussian", "regular", "custom"]


# ---------------------------------------------------------------------------
# Normalised associated Legendre functions
# ---------------------------------------------------------------------------


def legendre_table(lmax: int, mu: np.ndarray) -> np.ndarray:
    """Normalised associated Legendre functions on the nodes ``mu = sin(lat)``.

    Returns ``P`` with ``P[n, m, j] = Pbar_n^m(mu_j)``, 4-pi normalised without
    the Condon-Shortley phase, so that ``(1/2) * int Pbar_n^m ** 2 dmu == 1``.
    Entries with ``m > n`` are zero.

    The sectoral recursion carries a factor ``cos(lat)**m``, which underflows to
    zero within a few rows of the pole for large ``m``.  That is the true value to
    double precision, not a loss of accuracy, and it stays finite -- but it does
    cap the useful degree, so ``lmax`` above 512 is refused rather than returning
    silently degraded tables.
    """
    if lmax < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    if lmax > 512:
        raise ValueError(
            f"lmax={lmax} exceeds the 512 supported by this recursion; the "
            f"sectoral terms underflow near the poles beyond it"
        )
    mu = np.asarray(mu, dtype=np.float64).ravel()
    if np.any(np.abs(mu) > 1.0 + 1e-12):
        raise ValueError("mu must lie in [-1, 1] (it is sin(latitude))")
    mu = np.clip(mu, -1.0, 1.0)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))  # = cos(latitude)

    nj = mu.size
    p = np.zeros((lmax + 1, lmax + 1, nj), dtype=np.float64)
    p[0, 0] = 1.0

    # Sectoral terms, then one step up in degree, then the three-term recursion.
    for m in range(1, lmax + 1):
        p[m, m] = np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * sin_theta * p[m - 1, m - 1]
    for m in range(0, lmax):
        p[m + 1, m] = np.sqrt(2.0 * m + 3.0) * mu * p[m, m]
    for m in range(0, lmax + 1):
        for n in range(m + 2, lmax + 1):
            a = np.sqrt(
                (2.0 * n - 1.0) * (2.0 * n + 1.0) / ((n - m) * (n + m))
            )
            b = np.sqrt(
                (2.0 * n + 1.0)
                * (n + m - 1.0)
                * (n - m - 1.0)
                / ((n - m) * (n + m) * (2.0 * n - 3.0))
            )
            p[n, m] = a * mu * p[n - 1, m] - b * p[n - 2, m]
    return p


def _epsilon(n: np.ndarray | float, m: np.ndarray | float) -> np.ndarray:
    """``eps_{n,m} = sqrt((n^2 - m^2) / (4 n^2 - 1))``, zero at ``n = 0``."""
    n = np.asarray(n, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    num = n * n - m * m
    den = 4.0 * n * n - 1.0
    out = np.zeros(np.broadcast(n, m).shape, dtype=np.float64)
    ok = (den > 0) & (num > 0)
    out[ok] = np.sqrt(num[ok] / den[ok])
    return out


def legendre_derivative_table(p_ext: np.ndarray, lmax: int) -> np.ndarray:
    """``H[n, m, j] = (1 - mu^2) dPbar_n^m/dmu``, i.e. ``cos(lat) dPbar/dlat``.

    Built from the recursion ``H_n^m = -n eps_{n+1,m} P_{n+1}^m
    + (n+1) eps_{n,m} P_{n-1}^m``, which needs the Legendre table one degree
    beyond ``lmax`` -- pass ``p_ext = legendre_table(lmax + 1, mu)``.  Because it
    is built by combination rather than by differentiation, it carries no
    ``1/cos(lat)`` and stays regular at the poles.
    """
    nj = p_ext.shape[-1]
    h = np.zeros((lmax + 1, lmax + 1, nj), dtype=np.float64)
    for m in range(0, lmax + 1):
        for n in range(m, lmax + 1):
            term = np.zeros(nj, dtype=np.float64)
            if n + 1 <= lmax + 1:
                term -= n * _epsilon(n + 1, m) * p_ext[n + 1, m]
            if n - 1 >= m and n - 1 >= 0:
                term += (n + 1) * _epsilon(n, m) * p_ext[n - 1, m]
            h[n, m] = term
    return h


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """A latitude-longitude grid, latitudes ascending and longitudes ``[0,360)``.

    ``weights`` holds exact quadrature weights in ``mu`` when the grid has them
    (Gaussian); it is ``None`` otherwise, and analysis then falls back to a
    weighted least-squares fit.
    """

    lat: np.ndarray
    lon: np.ndarray
    kind: GridKind
    weights: np.ndarray | None = None

    @property
    def nlat(self) -> int:
        return int(self.lat.size)

    @property
    def nlon(self) -> int:
        return int(self.lon.size)

    @cached_property
    def mu(self) -> np.ndarray:
        return np.sin(np.radians(self.lat))

    @cached_property
    def cos_lat(self) -> np.ndarray:
        return np.cos(np.radians(self.lat))

    @property
    def has_poles(self) -> bool:
        return bool(np.any(np.abs(np.abs(self.lat) - 90.0) < 1e-8))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Grid({self.kind}, nlat={self.nlat}, nlon={self.nlon}, "
            f"lat={self.lat[0]:.3f}..{self.lat[-1]:.3f}, poles={self.has_poles})"
        )


def gaussian_grid(nlat: int, nlon: int) -> Grid:
    """Gauss-Legendre grid: exact quadrature, and no row on either pole."""
    mu, w = np.polynomial.legendre.leggauss(nlat)
    lat = np.degrees(np.arcsin(mu))  # leggauss returns ascending mu
    lon = np.arange(nlon, dtype=np.float64) * (360.0 / nlon)
    return Grid(lat=lat, lon=lon, kind="gaussian", weights=w)


def grid_from_axes(lat: np.ndarray, lon: np.ndarray) -> Grid:
    """Wrap data axes as a :class:`Grid`.

    Equally spaced latitudes are labelled ``"regular"``, anything else
    ``"custom"``; neither gets quadrature weights, so both are analysed by least
    squares.  The label is descriptive only -- nothing downstream assumes a
    spacing.

    Two things are refused rather than accommodated, because accommodating them
    quietly produces a field that still looks like weather.

    A descending latitude axis: reversing the axis without reversing the data
    turns every field upside down, and least-squares analysis fits that just as
    happily, returning a clean equator-mirrored solution with the eastward wind
    sign-flipped.  Reverse both, together, before calling.

    A longitude axis that does not begin at zero: the transform is an FFT over the
    column index and takes column zero to be the prime meridian, so a -180..180
    axis rotates the solver's frame by half a turn.  The per-level path never uses
    an absolute longitude and the shift would cancel, but anything that does --
    seeding a scale split at the event centre, for one -- would then be working at
    the antipode.  Roll the axis and the data together instead.
    """
    lat = np.asarray(lat, dtype=np.float64).ravel()
    lon = np.asarray(lon, dtype=np.float64).ravel()
    if lat.size > 1 and lat[0] > lat[-1]:
        raise ValueError(
            f"latitudes must ascend, but this axis runs {lat[0]:.3f} to "
            f"{lat[-1]:.3f}; reverse the axis and the data together"
        )
    dlon = np.diff(lon)
    if lon.size > 1 and (
        abs(lon[0]) > 1e-6
        or not np.allclose(dlon, 360.0 / lon.size, rtol=1e-6, atol=1e-9)
    ):
        raise ValueError(
            f"longitudes must start at 0 and be equally spaced over the full "
            f"circle; this axis starts at {lon[0]:.3f} with spacing "
            f"{dlon[0] if dlon.size else float('nan'):.3f}"
        )
    dlat = np.diff(lat)
    kind: GridKind = (
        "regular"
        if lat.size > 2 and np.allclose(dlat, dlat[0], rtol=1e-6, atol=1e-9)
        else "custom"
    )
    return Grid(lat=lat, lon=lon, kind=kind, weights=None)


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------


class SHT:
    """Float64 spherical-harmonic transform bound to one :class:`Grid`.

    Fields are arrays shaped ``(..., nlat, nlon)`` with latitude ascending;
    spectra are complex arrays shaped ``(..., lmax+1, lmax+1)`` indexed
    ``[m, n]``.  Leading axes are free, so a whole level stack transforms in one
    call and the Legendre contractions become BLAS matrix products.
    """

    def __init__(self, grid: Grid, lmax: int | None = None, radius: float = R_EARTH):
        self.grid = grid
        self.radius = float(radius)
        max_lmax = min(grid.nlat - 1, (grid.nlon - 1) // 2)
        self.lmax = int(max_lmax if lmax is None else lmax)
        if self.lmax < 1:
            raise ValueError(f"lmax must be at least 1, got {self.lmax}")
        if self.lmax > max_lmax:
            raise ValueError(
                f"lmax={self.lmax} is not resolved by a {grid.nlat}x{grid.nlon} "
                f"grid (maximum {max_lmax})"
            )

        p_ext = legendre_table(self.lmax + 1, grid.mu)
        # [m, n, j] ordering puts the (n, j) block for one m in contiguous memory.
        self._p = np.ascontiguousarray(
            np.transpose(p_ext[: self.lmax + 1, : self.lmax + 1], (1, 0, 2))
        )
        self._h = np.ascontiguousarray(
            np.transpose(legendre_derivative_table(p_ext, self.lmax), (1, 0, 2))
        )
        # The quadrature carries the weights and the transpose that analysis
        # contracts in, applied once here rather than at every transform.
        if grid.weights is None:
            self._p_quad = self._h_quad = None
        else:
            self._p_quad = self._quadrature(self._p)
            self._h_quad = self._quadrature(self._h)
        self._analysis = self._build_analysis()

        n = np.arange(self.lmax + 1, dtype=np.float64)
        self.n_deg = n
        #: Eigenvalues of the horizontal Laplacian, ``-n(n+1)/a^2``.
        self.laplacian_eigen = -n * (n + 1.0) / (self.radius**2)
        self._triu = np.tril(
            np.ones((self.lmax + 1, self.lmax + 1), dtype=bool)
        )  # [m, n] valid where n >= m

    # -- construction -------------------------------------------------------

    def _quadrature(self, table: np.ndarray) -> np.ndarray:
        """``(1/2) table W`` in the ``[m, j, n]`` layout analysis contracts in."""
        return np.ascontiguousarray(
            (0.5 * table * self.grid.weights).transpose(0, 2, 1)
        )

    def _build_analysis(self) -> np.ndarray:
        """Operator taking Fourier coefficients to spectra, laid out ``[m, j, n]``.

        On a Gaussian grid this is the exact quadrature ``(1/2) P W``, the same
        array the scalar transforms already hold, so it is shared rather than
        built again (the derivative table has a quadrature of its own, with
        different values).  Elsewhere it is the weighted least-squares
        pseudo-inverse of the synthesis matrix, which reproduces any field the
        grid can resolve to machine precision and otherwise returns the best
        area-weighted fit.  Entries whose degree is below the order are zero, so
        the whole triangle passes through one contraction.
        """
        if self.grid.weights is not None:
            return self._p_quad
        w = np.maximum(self.grid.cos_lat, 1e-6)
        sqrt_w = np.sqrt(w)
        out = np.zeros((self.lmax + 1, self.grid.nlat, self.lmax + 1))
        for m in range(self.lmax + 1):
            a = self._p[m, m:].T * sqrt_w[:, None]  # (nlat, n_count)
            out[m, :, m:] = (np.linalg.pinv(a, rcond=1e-12) * sqrt_w[None, :]).T
        return out

    # -- core transforms ----------------------------------------------------

    def analyze(self, field: np.ndarray) -> np.ndarray:
        """Grid to spectrum."""
        return self._analyze_with(field, self._analysis)

    def synthesize(self, spec: np.ndarray) -> np.ndarray:
        """Spectrum to grid."""
        return self._synth_with(spec, self._p)

    def synthesize_dlat(self, spec: np.ndarray) -> np.ndarray:
        """``cos(lat) * d/dlat`` of the field, on the grid.

        Regular at the poles: it is a Legendre combination, not a divided
        difference, so nothing is divided by ``cos(lat)`` here.
        """
        return self._synth_with(spec, self._h)

    def _synth_with(self, spec: np.ndarray, table: np.ndarray) -> np.ndarray:
        """Spectrum to grid against an arbitrary Legendre-like table.

        The zonal order is the batch index and the degree the contracted one, so
        every order of every leading level meets its table in a single batched
        matrix product.  Two things make that the right shape.  The table is zero
        wherever the degree is below the order, so padding each order's triangle
        out to a full rectangle changes no result and lets the orders share one
        call -- writing the loop is the natural thing to do, each order having its
        own length ``lmax + 1 - m``, but then the cost is entirely in getting into
        and out of ``lmax + 1`` small contractions.  And the real and imaginary
        parts are contracted separately: a real table meeting a complex spectrum
        would otherwise be widened to a complex copy, twice the size of the table
        itself, on every transform.
        """
        spec = np.asarray(spec, dtype=np.complex128)
        if spec.shape[-2:] != (self.lmax + 1, self.lmax + 1):
            raise ValueError(
                f"spectrum trailing shape {spec.shape[-2:]} does not match "
                f"lmax={self.lmax}"
            )
        lead = spec.shape[:-2]
        nfreq = self.grid.nlon // 2 + 1
        keep = min(self.lmax + 1, nfreq)
        fourier = np.zeros(lead + (self.grid.nlat, nfreq), dtype=np.complex128)
        # (m, levels, n) against (m, n, j), then back to (levels, j, m).
        by_order = spec.reshape((-1,) + spec.shape[-2:])[:, :keep].transpose(1, 0, 2)
        shape = lead + (self.grid.nlat, keep)
        fourier.real[..., :keep] = (
            (by_order.real @ table[:keep]).transpose(1, 2, 0).reshape(shape)
        )
        fourier.imag[..., :keep] = (
            (by_order.imag @ table[:keep]).transpose(1, 2, 0).reshape(shape)
        )
        return np.fft.irfft(fourier * self.grid.nlon, n=self.grid.nlon, axis=-1)

    # -- calculus -----------------------------------------------------------

    def dlon_spec(self, spec: np.ndarray) -> np.ndarray:
        """``d/dlon`` in spectral space (multiplication by ``i m``)."""
        m = np.arange(self.lmax + 1, dtype=np.float64)
        return spec * (1j * m)[:, None]

    def laplacian_spec(self, spec: np.ndarray) -> np.ndarray:
        """Horizontal Laplacian in spectral space."""
        return spec * self.laplacian_eigen[None, :]

    def invert_laplacian_spec(self, spec: np.ndarray) -> np.ndarray:
        """Inverse Laplacian, with the constant mode set to zero (mean-zero gauge)."""
        eig = self.laplacian_eigen.copy()
        eig[0] = np.inf  # n = 0 -> divides to zero
        return spec / eig[None, :]

    def gradient(self, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Eastward and northward derivative components of a scalar, on the grid.

        Returns ``(df_dx, df_dy)`` in per-metre units.  Both divide by
        ``cos(lat)``, so this must be called on a pole-free grid -- use the
        Gaussian solver grid.
        """
        if self.grid.has_poles:
            raise ValueError(
                "gradient divides by cos(lat) and cannot be evaluated on a grid "
                "that includes a pole; synthesise onto the Gaussian solver grid "
                "first"
            )
        cos_lat = self.grid.cos_lat[:, None]
        dfdx = self.synthesize(self.dlon_spec(spec)) / (self.radius * cos_lat)
        dfdy = self.synthesize_dlat(spec) / (self.radius * cos_lat)
        return dfdx, dfdy

    def divergence(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """Spectrum of the divergence of the vector field ``(fx, fy)``.

        Evaluated by integrating the meridional term by parts, so the metric
        factors appear only as ``1/(1 - mu^2)`` inside an analysis on the
        pole-free solver grid, never as a derivative of ``cos(lat)``.
        """
        if self.grid.weights is None:
            raise ValueError(
                "divergence uses exact quadrature and requires a Gaussian grid"
            )
        cos_lat = self.grid.cos_lat[:, None]
        inv_c2 = 1.0 / (cos_lat**2)
        u_hat = self._analyze_with(np.asarray(fx) * cos_lat * inv_c2, self._p_quad)
        v_hat = self._analyze_with(np.asarray(fy) * cos_lat * inv_c2, self._h_quad)
        return (self.dlon_spec(u_hat) - v_hat) / self.radius

    def vorticity(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """Spectrum of the vertical vorticity of ``(fx, fy)``; see :meth:`divergence`."""
        if self.grid.weights is None:
            raise ValueError(
                "vorticity uses exact quadrature and requires a Gaussian grid"
            )
        cos_lat = self.grid.cos_lat[:, None]
        inv_c2 = 1.0 / (cos_lat**2)
        u_hat = self._analyze_with(np.asarray(fx) * cos_lat * inv_c2, self._h_quad)
        v_hat = self._analyze_with(np.asarray(fy) * cos_lat * inv_c2, self._p_quad)
        return (self.dlon_spec(v_hat) + u_hat) / self.radius

    def _analyze_with(self, field: np.ndarray, operator: np.ndarray) -> np.ndarray:
        """Grid to spectrum against an analysis operator laid out ``[m, j, n]``.

        The mirror image of :meth:`_synth_with`: one batched matrix product over
        the zonal orders, contracting latitude, with the real and imaginary parts
        of the Fourier coefficients taken separately against the real operator.
        The entries with degree below order come out as zero because the operator
        is zero there, so they need not be masked.
        """
        field = np.asarray(field, dtype=np.float64)
        if field.shape[-2:] != (self.grid.nlat, self.grid.nlon):
            raise ValueError(
                f"field trailing shape {field.shape[-2:]} does not match grid "
                f"({self.grid.nlat}, {self.grid.nlon})"
            )
        fourier = np.fft.rfft(field, axis=-1) / self.grid.nlon  # (..., nlat, nfreq)
        lead = field.shape[:-2]
        keep = min(self.lmax + 1, fourier.shape[-1])
        spec = np.zeros(lead + (self.lmax + 1, self.lmax + 1), dtype=np.complex128)
        # (m, levels, j) against (m, j, n), then back to (levels, m, n).
        by_order = fourier.reshape((-1,) + fourier.shape[-2:])[..., :keep].transpose(
            2, 0, 1
        )
        shape = lead + (keep, self.lmax + 1)
        spec.real[..., :keep, :] = (
            (by_order.real @ operator[:keep]).transpose(1, 0, 2).reshape(shape)
        )
        spec.imag[..., :keep, :] = (
            (by_order.imag @ operator[:keep]).transpose(1, 0, 2).reshape(shape)
        )
        return spec

    # -- helpers ------------------------------------------------------------

    def zero_unused(self, spec: np.ndarray) -> np.ndarray:
        """Zero the ``m > n`` entries that carry no information."""
        return np.where(self._triu, spec, 0.0)

    def regrid_to(self, other: "SHT", field: np.ndarray) -> np.ndarray:
        """Move a field to another grid through the spectrum (exact if resolved)."""
        if other.lmax != self.lmax:
            raise ValueError(
                f"regrid needs matching truncations, got {self.lmax} and {other.lmax}"
            )
        return other.synthesize(self.analyze(field))

    def area_mean(self, field: np.ndarray) -> np.ndarray:
        """Area-weighted global mean, from the ``n = m = 0`` coefficient."""
        return np.real(self.analyze(field)[..., 0, 0])
