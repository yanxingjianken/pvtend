"""PV tendency term computation for weather events.

Orchestrates the full computation pipeline for a single event:

1. Load ERA5 data for the time window
2. Subtract climatology to get anomalies
3. Compute all spatial/temporal derivatives
4. FFT Helmholtz decomposition on the full NH hemisphere
5. QG omega solver → omega_dry
6. Moist/dry decomposition → omega_moist, chi_moist
7. Extract event-centred patches
8. Compute PV cross-terms and vertical weighted averages
9. Write per-timestep NPZ files

The :class:`TendencyComputer` class is parameterized by event type
(blocking / PRP), eliminating the 95 % code duplication between the
original scripts.
"""

from __future__ import annotations

import gc
import os
import tempfile
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from .constants import (
    DEFAULT_LEVELS,
    G0,
    H_SCALE,
    KAPPA,
    OMEGA_E,
    R_DRY,
    R_EARTH,
    WAVG_LEVELS,
)

# Alias for brevity
LEVELS = DEFAULT_LEVELS
from .helmholtz import helmholtz_decomposition
from .moist_dry import solve_chi_moist
from .omega import compute_geostrophic_wind, solve_qg_omega

# ── Full list of variables stored in each NPZ ──────────────────────────
VARS_3D: list[str] = [
    "z",
    "pv",
    "u",
    "v",
    "w",
    "t",
    "pv_dt",
    "pv_total_dx",
    "pv_total_dy",
    "pv_total_dp",
    "u_bar",
    "v_bar",
    "w_bar",
    "pv_bar",
    "u_anom",
    "v_anom",
    "w_anom",
    "pv_anom",
    "pv_bar_dx",
    "pv_bar_dy",
    "pv_bar_dp",
    "pv_bar_dt",
    "pv_anom_dx",
    "pv_anom_dy",
    "pv_anom_dp",
    "pv_anom_dt",
    "theta",
    "theta_dt",
    "theta_dot",
    "Q",
    "u_anom_rot",
    "u_anom_div",
    "u_anom_har",
    "v_anom_rot",
    "v_anom_div",
    "v_anom_har",
    "u_anom_div_moist",
    "v_anom_div_moist",
    "u_anom_div_dry",
    "v_anom_div_dry",
    "omega_dry",
    "omega_moist",
]


@dataclass
class TendencyConfig:
    """Configuration for PV tendency computation.

    Attributes:
        event_type: ``'blocking'`` or ``'prp'``.
        data_dir: Path to ERA5 monthly NetCDF files.
        clim_path: Path to climatology file or directory.
        output_dir: Root output directory for NPZ files.
        csv_path: Path to TempestExtremes event CSV.
        levels: Pressure levels [hPa].
        wavg_levels: Subset of *levels* used for vertical weighted
            averaging.
        rel_hours: Relative hours around the event reference time.
        year_start: First event year to process.
        year_end: Last event year to process (exclusive upper bound
            for ``range``).
        lat_half: Half-width of the extraction patch in degrees
            latitude.
        lon_half: Half-width of the extraction patch in degrees
            longitude.
        partial_at_pole: If ``True``, allow truncated (asymmetric)
            patches near the poles instead of skipping those events.
        use_constant_sigma: Use a constant static stability parameter
            in the QG omega solver (``True``) or compute it from the
            temperature field (``False``).
        center_mode: ``'eulerian'`` (fixed centre) or ``'lagrangian'``
            (track-following centre).
        engine: NetCDF engine passed to :func:`xarray.open_dataset`.
    """

    event_type: str = "blocking"
    data_dir: Path = Path("/net/flood/data2/users/x_yan/era")
    clim_path: Path = Path(
        "/net/flood/data2/users/x_yan/era/era5_hourly_clim_1990-2019.nc"
    )
    output_dir: Path = Path(
        "/net/flood/data2/users/x_yan/composite_blocking_tempest"
    )
    csv_path: Path = Path("")
    levels: list[int] = field(default_factory=lambda: list(LEVELS))
    wavg_levels: list[int] = field(default_factory=lambda: list(WAVG_LEVELS))
    rel_hours: list[int] = field(default_factory=lambda: list(range(-49, 25)))
    year_start: int = 1990
    year_end: int = 2020
    lat_half: float = 21.0
    lon_half: float = 36.0
    partial_at_pole: bool = True
    use_constant_sigma: bool = True
    center_mode: str = "eulerian"
    engine: str = "netcdf4"


class TendencyComputer:
    """Computes PV tendency terms for weather events.

    Parameterized by :class:`TendencyConfig` — works for both blocking
    and PRP event types without code duplication.

    Example::

        cfg = TendencyConfig(
            event_type="blocking",
            csv_path=Path("events_blocking.csv"),
        )
        tc = TendencyComputer(cfg)
        n = tc.process_event("onset", track_id=42, lat0=55.0,
                             lon0=-30.0, base_ts=pd.Timestamp("2010-01-15"))
        print(f"Wrote {n} NPZ files.")
    """

    def __init__(self, config: TendencyConfig) -> None:
        self.cfg = config
        self._clim: xr.Dataset | None = None
        self._grid: dict[str, np.ndarray] | None = None

    # ── public API ─────────────────────────────────────────────────

    def process_event(
        self,
        evt_name: str,
        track_id: int,
        lat0: float,
        lon0: float,
        base_ts: pd.Timestamp,
    ) -> int:
        """Process a single event and write NPZ files.

        For every relative hour in ``self.cfg.rel_hours`` the method:

        1. Calls :meth:`_with_derivs_for_window` to obtain the
           full-hemisphere dataset with anomalies, derivatives,
           Helmholtz components, QG omega, and moist/dry decomposition.
        2. Extracts an event-centred lat/lon patch.
        3. Computes the PV advection cross-terms (mean-on-anomaly,
           anomaly-on-mean, anomaly-on-anomaly, etc.).
        4. Performs a pressure-weighted vertical average over
           ``cfg.wavg_levels``.
        5. Writes the result as a compressed ``.npz`` file via
           :meth:`_out_path`.

        Args:
            evt_name: Event stage name (e.g. ``'onset'``, ``'peak'``,
                ``'decay'``).
            track_id: Integer track identifier from the CSV catalogue.
            lat0: Event centre latitude [degrees N].
            lon0: Event centre longitude [degrees E].
            base_ts: Reference timestamp for the event stage.

        Returns:
            Number of NPZ files successfully written.
        """
        written = 0

        for dh in self.cfg.rel_hours:
            ts = base_ts + pd.Timedelta(hours=dh)

            # Skip if output already exists
            out_file = self._out_path(evt_name, dh, track_id, ts)
            if out_file.exists():
                written += 1
                continue

            try:
                ds = self._with_derivs_for_window(ts)
            except (FileNotFoundError, KeyError) as exc:
                # Data gap — skip this timestep
                continue

            # ── Extract event-centred patch ──
            patch = self._extract_patch(ds, lat0, lon0)
            if patch is None:
                continue

            # ── Compute PV cross-terms ──
            terms = self._compute_cross_terms(patch)

            # ── Vertical weighted average ──
            terms_wavg = self._vertical_weighted_average(terms)

            # ── Write NPZ ──
            out_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(out_file), **terms, **terms_wavg)
            written += 1

            # Free memory
            del ds, patch, terms, terms_wavg
            gc.collect()

        return written

    # ── internal methods ───────────────────────────────────────────

    def _with_derivs_for_window(
        self,
        centre_ts: pd.Timestamp,
    ) -> xr.Dataset:
        """Load data window, compute anomalies, derivatives, Helmholtz, QG omega.

        This is the heaviest function in the pipeline (~300 lines when
        fully ported).  It performs the following steps:

        1. **Load ERA5 fields** (u, v, z, t, w/omega, pv) for a ±6 h
           window around *centre_ts* from monthly NetCDF files.
        2. **Load / cache climatology** and subtract it to form
           anomaly fields (u', v', T', PV', etc.).
        3. **Spatial derivatives** (∂/∂x, ∂/∂y) on the sphere using
           centred finite differences with cos(lat) metric terms.
        4. **Pressure derivatives** (∂/∂p) using centred differences
           on the level axis.
        5. **Time derivatives** (∂/∂t) using centred differences over
           the ±6 h window.
        6. **Helmholtz decomposition** of anomalous wind into
           rotational, divergent, and harmonic parts on the full NH
           hemisphere via FFT.
        7. **QG omega** solver (``solve_qg_omega``) using geostrophic
           wind and temperature to obtain ω_dry.
        8. **Moist/dry decomposition** via ``solve_chi_moist`` to
           obtain ω_moist and the moist/dry divergent wind split.
        9. **Potential temperature** and diabatic heating residual.

        Args:
            centre_ts: Central timestamp of the data window.

        Returns:
            :class:`xr.Dataset` containing all fields, anomalies,
            derivatives, and decomposed winds on the ERA5 grid.

        Raises:
            FileNotFoundError: If a required ERA5 monthly file is
                missing.
            KeyError: If a required variable is absent from the file.

        .. todo::
            Port the full implementation from
            ``step2_compute_tendency_terms_blocking.py``.  The logic
            is ~300 lines of array manipulation and is deferred to
            avoid introducing bugs before unit tests are in place.
        """
        # TODO: Port full implementation from step2_compute_tendency_terms.
        # Skeleton:
        #   1. Determine file paths for centre_ts ± 6h
        #   2. Open datasets, select levels
        #   3. Load / cache climatology → self._clim
        #   4. Compute anomalies
        #   5. Spatial derivatives (dx, dy, dp)
        #   6. Time derivatives (dt)
        #   7. Helmholtz decomposition (helmholtz_decomposition)
        #   8. QG omega (compute_geostrophic_wind, solve_qg_omega)
        #   9. Moist/dry split (solve_chi_moist)
        #  10. Theta, diabatic heating
        #  11. Assemble and return xr.Dataset
        raise NotImplementedError(
            "_with_derivs_for_window not yet ported — see TODO above."
        )

    def _load_climatology(self) -> xr.Dataset:
        """Load and cache the ERA5 climatology dataset.

        Returns:
            Climatology dataset indexed by ``(dayofyear, hour, level,
            latitude, longitude)``.
        """
        if self._clim is None:
            self._clim = xr.open_dataset(
                self.cfg.clim_path, engine=self.cfg.engine
            )
        return self._clim

    def _extract_patch(
        self,
        ds: xr.Dataset,
        lat0: float,
        lon0: float,
    ) -> xr.Dataset | None:
        """Extract an event-centred lat/lon patch from the dataset.

        Args:
            ds: Full-hemisphere dataset.
            lat0: Event centre latitude [degrees N].
            lon0: Event centre longitude [degrees E].

        Returns:
            Sub-dataset covering ``[lat0 ± lat_half, lon0 ± lon_half]``,
            or ``None`` if the patch would fall outside the domain and
            ``partial_at_pole`` is ``False``.
        """
        lat = ds["latitude"].values
        lon = ds["longitude"].values

        lat_min = lat0 - self.cfg.lat_half
        lat_max = lat0 + self.cfg.lat_half

        # Handle pole truncation
        if not self.cfg.partial_at_pole:
            if lat_min < lat.min() or lat_max > lat.max():
                return None
        lat_min = max(lat_min, float(lat.min()))
        lat_max = min(lat_max, float(lat.max()))

        lon_min = lon0 - self.cfg.lon_half
        lon_max = lon0 + self.cfg.lon_half

        # Longitude wrapping
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        if lon_min < 0 or lon_max >= 360:
            # Wrap around the date line
            lon_min_w = lon_min % 360
            lon_max_w = lon_max % 360
            if lon_min_w > lon_max_w:
                lon_mask = (lon >= lon_min_w) | (lon <= lon_max_w)
            else:
                lon_mask = (lon >= lon_min_w) & (lon <= lon_max_w)
        else:
            lon_mask = (lon >= lon_min) & (lon <= lon_max)

        return ds.sel(
            latitude=lat[lat_mask],
            longitude=lon[lon_mask],
        )

    def _compute_cross_terms(
        self,
        patch: xr.Dataset,
    ) -> dict[str, np.ndarray]:
        """Compute PV advection cross-terms on the extracted patch.

        The cross-terms decompose the material PV tendency into
        contributions from mean and anomalous wind acting on mean
        and anomalous PV gradients:

        - ``u_bar * dpv_bar_dx`` — mean advection of mean PV
        - ``u_bar * dpv_anom_dx`` — mean advection of anomaly PV
        - ``u_anom * dpv_bar_dx`` — anomaly advection of mean PV
        - ``u_anom * dpv_anom_dx`` — anomaly advection of anomaly PV
        - (and the same for v/dy and w/dp components)
        - sub-decomposed by rotational/divergent/harmonic wind parts
        - sub-decomposed by moist/dry divergent wind parts

        Args:
            patch: Event-centred dataset with all fields.

        Returns:
            Dictionary mapping variable names to numpy arrays.
        """
        terms: dict[str, np.ndarray] = {}

        # Convenience accessors
        def _get(name: str) -> np.ndarray:
            return patch[name].values

        # Naming convention matches step2_compute_tendency_terms_blocking.py
        # and projection.py's ADVECTION_TERMS, e.g. "u_bar_pv_bar_dx",
        # "u_rot_pv_anom_dx", "omega_dry_pv_bar_dp", etc.

        # ── Mean / anomaly × mean / anomaly PV gradient ──
        for component, vel_name in [("x", "u"), ("y", "v"), ("p", "w")]:
            grad_key = f"d{component}"  # dx, dy, dp
            bar_vel = _get(f"{vel_name}_bar")
            anom_vel = _get(f"{vel_name}_anom")
            bar_grad = _get(f"pv_bar_{grad_key}")
            anom_grad = _get(f"pv_anom_{grad_key}")

            terms[f"{vel_name}_bar_pv_bar_{grad_key}"] = bar_vel * bar_grad
            terms[f"{vel_name}_bar_pv_anom_{grad_key}"] = bar_vel * anom_grad
            terms[f"{vel_name}_anom_pv_bar_{grad_key}"] = anom_vel * bar_grad
            terms[f"{vel_name}_anom_pv_anom_{grad_key}"] = anom_vel * anom_grad

        # ── Rotational / divergent / harmonic sub-decomposition ──
        for part in ["rot", "div", "har"]:
            for component in ["x", "y"]:
                vel_name = "u" if component == "x" else "v"
                grad_key = f"d{component}"
                anom_part = _get(f"{vel_name}_anom_{part}")
                bar_grad = _get(f"pv_bar_{grad_key}")
                anom_grad = _get(f"pv_anom_{grad_key}")
                terms[f"{vel_name}_{part}_pv_bar_{grad_key}"] = (
                    anom_part * bar_grad
                )
                terms[f"{vel_name}_{part}_pv_anom_{grad_key}"] = (
                    anom_part * anom_grad
                )

        # ── Moist / dry divergent sub-decomposition ──
        for md in ["moist", "dry"]:
            for component in ["x", "y"]:
                vel_name = "u" if component == "x" else "v"
                grad_key = f"d{component}"
                key = f"{vel_name}_anom_div_{md}"
                if key in patch:
                    vel_md = _get(key)
                    bar_grad = _get(f"pv_bar_{grad_key}")
                    anom_grad = _get(f"pv_anom_{grad_key}")
                    terms[f"{vel_name}_div_{md}_pv_bar_{grad_key}"] = (
                        vel_md * bar_grad
                    )
                    terms[f"{vel_name}_div_{md}_pv_anom_{grad_key}"] = (
                        vel_md * anom_grad
                    )

        # ── Moist / dry omega vertical sub-decomposition ──
        for md in ["moist", "dry"]:
            key = f"omega_{md}"
            if key in patch:
                vel_md = _get(key)
                bar_grad = _get("pv_bar_dp")
                anom_grad = _get("pv_anom_dp")
                terms[f"omega_{md}_pv_bar_dp"] = vel_md * bar_grad
                terms[f"omega_{md}_pv_anom_dp"] = vel_md * anom_grad

        return terms

    def _vertical_weighted_average(
        self,
        terms: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Pressure-weighted vertical average over ``cfg.wavg_levels``.

        The weight for each level is proportional to the pressure
        thickness of the layer (Δp).  The result keys are prefixed
        with ``"wavg_"``.

        Args:
            terms: Dictionary of 3-D arrays ``(nlev, nlat, nlon)``.

        Returns:
            Dictionary with ``"wavg_"``-prefixed keys containing 2-D
            arrays ``(nlat, nlon)``.
        """
        plevs_hpa = np.array(self.cfg.levels, dtype=float)
        wavg_mask = np.isin(plevs_hpa, self.cfg.wavg_levels)
        wavg_idx = np.where(wavg_mask)[0]

        if len(wavg_idx) == 0:
            return {}

        # Compute layer thickness weights (Δp)
        plevs_pa = plevs_hpa * 100.0
        dp = np.zeros_like(plevs_pa)
        for i, idx in enumerate(wavg_idx):
            if idx == 0:
                dp[idx] = plevs_pa[1] - plevs_pa[0]
            elif idx == len(plevs_pa) - 1:
                dp[idx] = plevs_pa[-1] - plevs_pa[-2]
            else:
                dp[idx] = (plevs_pa[idx + 1] - plevs_pa[idx - 1]) / 2.0

        weights = dp[wavg_idx]
        weights = weights / weights.sum()

        wavg_terms: dict[str, np.ndarray] = {}
        for key, arr in terms.items():
            if arr.ndim < 3:
                continue
            subset = arr[wavg_idx]  # (n_wavg, nlat, nlon)
            wavg_terms[f"wavg_{key}"] = np.einsum(
                "k,k...->...", weights, subset
            )

        return wavg_terms

    def _out_path(
        self,
        evt: str,
        dh: int,
        track_id: int,
        ts: pd.Timestamp,
    ) -> Path:
        """Compute the output NPZ file path.

        The directory structure is::

            {output_dir}/{evt}/dh={dh:+d}/track_{id}_{YYYYMMDDHH}_dh{dh:+d}.npz

        Args:
            evt: Event stage name (e.g. ``'onset'``).
            dh: Relative hour offset from the event reference time.
            track_id: Track identifier.
            ts: Absolute timestamp for this offset.

        Returns:
            :class:`Path` to the output ``.npz`` file.
        """
        return (
            self.cfg.output_dir
            / evt
            / f"dh={dh:+d}"
            / f"track_{track_id}_{ts.strftime('%Y%m%d%H')}_dh{dh:+d}.npz"
        )
