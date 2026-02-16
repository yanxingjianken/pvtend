"""Grid utilities: cropping, interpolation, and event-centred patch extraction.

Supports arbitrary input resolution and domain — crops to NH and
bilinearly interpolates to a regular 1.5° grid matching ERA5-style layout.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field
from typing import Optional

from .constants import (
    TARGET_LAT, TARGET_LON, LAT_HALF, LON_HALF, R_EARTH,
)


@dataclass(frozen=True)
class NHGrid:
    """Northern Hemisphere regular lat-lon grid.

    Attributes:
        lat: 1-D latitude array, descending (90 → 0).
        lon: 1-D longitude array (-180 → 180).
        dlat: Grid spacing in latitude [deg].
        dlon: Grid spacing in longitude [deg].
    """
    lat: np.ndarray
    lon: np.ndarray

    @property
    def dlat(self) -> float:
        """Grid spacing in latitude [deg]."""
        return float(abs(np.diff(self.lat).mean()))

    @property
    def dlon(self) -> float:
        """Grid spacing in longitude [deg]."""
        return float(abs(np.diff(self.lon).mean()))

    @property
    def nlat(self) -> int:
        """Number of latitude points."""
        return len(self.lat)

    @property
    def nlon(self) -> int:
        """Number of longitude points."""
        return len(self.lon)

    @property
    def lat_descending(self) -> bool:
        """True if latitude array is in descending order."""
        return bool(np.all(np.diff(self.lat) < 0))

    @property
    def dy(self) -> float:
        """Meridional grid spacing in metres."""
        return np.deg2rad(self.dlat) * R_EARTH

    @property
    def dx_arr(self) -> np.ndarray:
        """Zonal grid spacing per latitude row [m], shape (nlat,)."""
        dx = np.deg2rad(self.dlon) * R_EARTH * np.cos(np.deg2rad(self.lat))
        return np.maximum(dx, self.dy * 0.01)


def default_nh_grid() -> NHGrid:
    """Return the standard 1.5° NH grid (90°N–0°, -180°–180°)."""
    return NHGrid(lat=TARGET_LAT.copy(), lon=TARGET_LON.copy())


def crop_to_nh(lat: np.ndarray, lon: np.ndarray,
               data: np.ndarray, lat_axis: int = -2
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop data to Northern Hemisphere (lat >= 0).

    Parameters:
        lat: Input latitude array.
        lon: Input longitude array.
        data: N-D array with latitude along `lat_axis`.
        lat_axis: Axis index for latitude.

    Returns:
        (nh_lat, lon, nh_data)
    """
    mask = lat >= 0
    nh_lat = lat[mask]
    slices = [slice(None)] * data.ndim
    slices[lat_axis] = mask
    return nh_lat, lon, data[tuple(slices)]


def bilinear_interpolate(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    data: np.ndarray,
    dst_lat: np.ndarray = TARGET_LAT,
    dst_lon: np.ndarray = TARGET_LON,
) -> np.ndarray:
    """Bilinearly interpolate 2-D or N-D data to target grid.

    The last two axes are assumed to be (lat, lon).

    Parameters:
        src_lat: Source latitude (ascending or descending).
        src_lon: Source longitude.
        data: Array with shape (..., nlat_src, nlon_src).
        dst_lat: Target latitude array.
        dst_lon: Target longitude array.

    Returns:
        Interpolated array with shape (..., nlat_dst, nlon_dst).
    """
    # Ensure ascending lat for interpolator
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        data = data[..., ::-1, :]

    # Build target mesh
    dst_lat_g, dst_lon_g = np.meshgrid(dst_lat, dst_lon, indexing="ij")
    points = np.stack([dst_lat_g.ravel(), dst_lon_g.ravel()], axis=-1)

    orig_shape = data.shape[:-2]
    flat = data.reshape(-1, data.shape[-2], data.shape[-1])
    out = np.empty(
        (flat.shape[0], len(dst_lat), len(dst_lon)), dtype=data.dtype
    )
    for i in range(flat.shape[0]):
        interp = RegularGridInterpolator(
            (src_lat, src_lon), flat[i],
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        out[i] = interp(points).reshape(len(dst_lat), len(dst_lon))

    return out.reshape(*orig_shape, len(dst_lat), len(dst_lon))


@dataclass
class EventPatch:
    """Event-centred patch extraction from a full NH grid.

    Attributes:
        grid: The underlying NHGrid.
        lat_half: Half-window in latitude [deg].
        lon_half: Half-window in longitude [deg].
    """
    grid: NHGrid
    lat_half: float = LAT_HALF
    lon_half: float = LON_HALF

    @property
    def lat_pad(self) -> int:
        """Number of grid points of padding in latitude."""
        return int(round(self.lat_half / self.grid.dlat))

    @property
    def lon_pad(self) -> int:
        """Number of grid points of padding in longitude."""
        return int(round(self.lon_half / self.grid.dlon))

    @property
    def patch_shape(self) -> tuple[int, int]:
        """Shape of the extracted patch (nlat_patch, nlon_patch)."""
        return (2 * self.lat_pad + 1, 2 * self.lon_pad + 1)

    def relative_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return relative coordinate arrays (Y_rel, X_rel) in degrees."""
        rlat = np.linspace(-self.lat_half, self.lat_half, 2 * self.lat_pad + 1)
        rlon = np.linspace(-self.lon_half, self.lon_half, 2 * self.lon_pad + 1)
        Y_rel, X_rel = np.meshgrid(rlat, rlon, indexing="ij")
        return Y_rel, X_rel

    def nearest_idx(self, lat0: float, lon0: float
                    ) -> tuple[int, int, bool]:
        """Find nearest grid index and check if patch fits.

        Parameters:
            lat0: Event centre latitude [deg].
            lon0: Event centre longitude [deg].

        Returns:
            (ilat, ilon, ok) where ok=True means the full patch fits
            within the latitude bounds.
        """
        ilat = int(np.abs(self.grid.lat - lat0).argmin())
        ilon = int(np.abs(self.grid.lon - lon0).argmin())
        ok = (ilat >= self.lat_pad and
              ilat + self.lat_pad < self.grid.nlat)
        return ilat, ilon, ok

    def wrapped_lon_index(self, ilon: int) -> np.ndarray:
        """Return longitude indices with periodic wrapping.

        Parameters:
            ilon: Centre longitude index.

        Returns:
            Array of longitude indices of length (2 * lon_pad + 1).
        """
        start = ilon - self.lon_pad
        return (np.arange(2 * self.lon_pad + 1) + start) % self.grid.nlon

    def extract(self, data: np.ndarray, ilat: int, ilon: int,
                eff_north: Optional[int] = None,
                eff_south: Optional[int] = None) -> np.ndarray:
        """Extract event-centred patch from (..., nlat, nlon) data.

        Handles zonal wrap and asymmetric polar padding.

        Parameters:
            data: Array with last two dims (nlat, nlon).
            ilat: Centre latitude index.
            ilon: Centre longitude index.
            eff_north: Effective northward padding (default: lat_pad).
            eff_south: Effective southward padding (default: lat_pad).

        Returns:
            Patch array of shape (..., 2*lat_pad+1, 2*lon_pad+1).
            NaN-filled where data doesn't reach.
        """
        if eff_north is None:
            eff_north = self.lat_pad
        if eff_south is None:
            eff_south = self.lat_pad

        lon_idx = self.wrapped_lon_index(ilon)
        full_h = 2 * self.lat_pad + 1

        out_shape = data.shape[:-2] + (full_h, len(lon_idx))
        out = np.full(out_shape, np.nan, dtype=data.dtype)

        if self.grid.lat_descending:
            i0 = max(0, ilat - eff_north)
            i1 = min(self.grid.nlat, ilat + eff_south + 1)
        else:
            i0 = max(0, ilat - eff_south)
            i1 = min(self.grid.nlat, ilat + eff_north + 1)

        lat_slice = data[..., i0:i1, :]
        lon_sub = lat_slice[..., lon_idx]

        if self.grid.lat_descending:
            lon_sub = lon_sub[..., ::-1, :]

        y_eff = lon_sub.shape[-2]
        y0 = self.lat_pad - eff_south
        out[..., y0:y0 + y_eff, :] = lon_sub

        return out
