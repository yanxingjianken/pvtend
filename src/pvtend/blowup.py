"""Hard-threshold ω-blowup detection (±25 Pa/s at 300 hPa by default).

Background
----------
Earlier versions of :func:`pvtend.omega.solve_qg_omega_sip` silently
**clipped** ω at ±5 Pa/s when the 300 hPa magnitude exceeded that
bound.  The clip suppressed the very anomalies the downstream blowup
analysis needs to *flag*, while paradoxically *not* preventing
contamination of the derived ω_dry / ω_moist / ω_LHR / ω_qg-moist
fields by ill-conditioned spectral inversions.  The fix has two parts:

1. The QG-ω solver's default ``w_physical_max`` is now ``None`` — no
   in-solver clipping.
2. This module performs a **post-hoc scan**: any timestamp whose
   max absolute ω at 300 hPa exceeds the ±25 Pa/s hard threshold (or
   any user-supplied cutoff) is recorded in a CSV for exclusion by
   the event-NPZ builders, RWB classifier, and composite stage.

The threshold is a fixed physical cutoff (Pa s⁻¹), calibrated against
the empirical raw-ERA5 envelope at 300 hPa over 1990-2020 hourly
(max=22.4 Pa/s, 99.9th=19.9 Pa/s).  Default 25 Pa/s = raw_max + ~10 %
headroom, so values above it in the *derived* QG-solver fields
(ω_dry / ω_moist / ω_LHR / ω_qg-moist) are essentially always solver
pathology, never observable atmospheric vertical motion.

**Note (v2.10.8+):** The recommended path is now per-event NPZ scanning
via :mod:`scripts.aggregate_qg_blowup`, which reads the
``max_abs_w_*_300`` scalars embedded by :class:`pvtend.tendency.TendencyComputer`
in every NPZ.  This module is retained for back-compat scans of raw
ERA5 ω globs.

Usage
-----

Programmatic::

    from pvtend.blowup import scan_omega_blowup
    df = scan_omega_blowup(
        era5_w_glob="/net/flood/data2/users/x_yan/era/era5_w_*.nc",
        level_pa=30000.0,
        threshold=25.0,
        out_csv="outputs/blowup_scan/omega_300hPa_10pa.csv",
    )

CLI::

    pvtend blowup-scan \\
        --era5 '/net/flood/data2/users/x_yan/era/era5_w_*.nc' \\
        --level 300 --threshold 25.0 \\
        --out outputs/blowup_scan/omega_300hPa_10pa.csv
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _drop_expver(ds: xr.Dataset) -> xr.Dataset:
    """Drop ERA5/ERA5T ``expver`` coord which is inconsistent across files.

    Some ERA5 monthly files (the recent ERA5T overlap window) carry an
    ``expver`` scalar coordinate that older files lack.  Without
    dropping it, ``xr.open_mfdataset(..., combine="by_coords")`` raises
    ``ValueError: coordinate 'expver' not present in all datasets``.
    """
    drop = [c for c in ("expver", "number") if c in ds.coords]
    return ds.drop_vars(drop) if drop else ds


def _open_w_lazy(era5_w_glob: str) -> xr.Dataset:
    """Open ERA5 ω files lazily as a single dask-backed dataset."""
    files = sorted(glob.glob(era5_w_glob))
    if not files:
        raise FileNotFoundError(f"No files match: {era5_w_glob!r}")
    return xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks={"valid_time": 50},
        engine="netcdf4",
        preprocess=_drop_expver,
        compat="override",
        coords="minimal",
    )


def _select_level(da: xr.DataArray, level_pa: float) -> xr.DataArray:
    """Select the pressure level closest to *level_pa* [Pa]."""
    plev_name = next(
        n for n in ("pressure_level", "level", "plev", "lev")
        if n in da.dims or n in da.coords
    )
    plev_vals = da[plev_name].values.astype(float)
    # ERA5 stores in hPa; convert if needed
    factor = 1.0 if plev_vals.max() > 2000 else 100.0
    plev_pa = plev_vals * factor
    k = int(np.argmin(np.abs(plev_pa - level_pa)))
    return da.isel({plev_name: k})


def scan_omega_blowup(
    era5_w_glob: str,
    level_pa: float = 30000.0,
    threshold: float = 25.0,
    out_csv: str | os.PathLike | None = None,
) -> pd.DataFrame:
    """Flag ERA5 timestamps whose max absolute ω at *level_pa* exceeds *threshold*.

    The threshold is a hard physical cutoff in Pa s⁻¹ — the canonical
    ±5 Pa/s rule applied at 300 hPa for ERA5 ω blowup detection.

    Args:
        era5_w_glob: Glob for ERA5 ω NetCDF files.
        level_pa: Pressure level [Pa]; default ``30000`` (300 hPa).
        threshold: Hard cutoff abs(ω) [Pa s⁻¹]; default ``25.0`` (raw ERA5
            300 hPa ω rarely exceeds 2-3 Pa/s, so |ω|>10 Pa/s in derived
            ω_dry / ω_moist / ω_LHR is essentially always solver blowup).
        out_csv: Optional output CSV path.

    Returns:
        DataFrame with columns ``timestamp, level_pa, max_abs_omega,
        threshold, n_exceed, exceedance_ratio`` for every timestamp
        whose spatial-max abs(ω) > *threshold*.  ``n_exceed`` is the
        number of grid cells exceeding the threshold and
        ``exceedance_ratio`` is ``max_abs_omega / threshold`` (>1 →
        blowup).
    """
    ds = _open_w_lazy(era5_w_glob)
    var = "w" if "w" in ds.data_vars else next(iter(ds.data_vars))
    da = _select_level(ds[var], level_pa)
    abs_da = np.abs(da)

    spatial_dims = [d for d in abs_da.dims if d != "valid_time"]
    spatial_max = abs_da.max(dim=spatial_dims).compute().values
    n_exceed = (abs_da > threshold).sum(dim=spatial_dims).compute().values
    times = pd.to_datetime(abs_da["valid_time"].values)

    df = pd.DataFrame({
        "timestamp": times,
        "level_pa": float(level_pa),
        "max_abs_omega": spatial_max,
        "threshold": float(threshold),
        "n_exceed": n_exceed.astype(int),
        "exceedance_ratio": spatial_max / float(threshold),
    })

    flagged = df[df["max_abs_omega"] > threshold].copy()
    flagged = flagged.sort_values("timestamp").reset_index(drop=True)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        flagged.to_csv(out_csv, index=False)

    return flagged
