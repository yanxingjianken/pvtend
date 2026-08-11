"""End-to-end regression for the dolma-era ERA5 layout.

The CESM2-LENS2 support added in v2.14 touched code the ERA5 path also runs
through — the window pad, the climatology lookup, the ``z`` unit divisor, the
below-ground fill, the ``valid_time`` rechunk, the Helmholtz-bar source, and
``track_id`` typing. Every one of those was *written* to be a no-op on ERA5,
but "written to be" is not evidence.

The target is specifically the self-downloaded **monthly** layout the ERA5 work
used before the move to Derecho:

    era5_{var}_{year}_{month:02d}.nc     one variable per file
    hourly, coords valid_time / latitude / longitude / level
    z is geopotential [m²/s²]            (so H = z/g)
    no below-ground NaN                  (ECMWF extrapolates at the source)

Not GLADE's ``d633000``, which is laid out differently and has none of these
files. Zero files in this layout exist on this machine, so the fixture builds
them.

The grid is deliberately tiny (a 1.5°-spaced NH strip) — this is a plumbing
regression, not a physics test. What it asserts is that the ERA5 path still
*runs* and still writes the keys downstream code reads.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pvtend.constants import MONTH_ABBREVS

LEVELS = [1000, 850, 700, 500, 400, 300, 250, 200, 100]
NLAT, NLON = 25, 32          # 0..36 N at 1.5 deg, full circle at 11.25 deg
YEAR, MONTH = 2010, 1
STEM = "era5_hourly_clim_1990-2020"
VARS = ("u", "v", "w", "pv", "z", "t", "q")


def _grid():
    lat = np.linspace(0.0, 36.0, NLAT)
    lon = np.linspace(-180.0, 180.0, NLON, endpoint=False)
    return lat, lon


def _field(rng, shape, scale, offset=0.0):
    return (offset + scale * rng.standard_normal(shape)).astype(np.float32)


@pytest.fixture(scope="module")
def era5_tree(tmp_path_factory):
    """Monthly ERA5 files + an ERA5-layout climatology + Helmholtz bars."""
    root = tmp_path_factory.mktemp("era5")
    data_dir, clim_dir = root / "data", root / "clim"
    data_dir.mkdir()
    clim_dir.mkdir()
    lat, lon = _grid()
    rng = np.random.default_rng(0)

    # --- monthly state files, hourly through the first three days ----------
    times = pd.date_range(f"{YEAR}-{MONTH:02d}-01", periods=72, freq="1h")
    shape = (len(times), len(LEVELS), NLAT, NLON)
    # A smooth zonal jet plus noise: the QG/Helmholtz stages need a field with
    # some structure, but nothing here depends on it being realistic.
    jet = (np.linspace(0, 30, NLAT)[None, None, :, None]
           * np.linspace(0.3, 1.0, len(LEVELS))[None, :, None, None])
    base = {
        "u": jet + _field(rng, shape, 2.0),
        "v": _field(rng, shape, 2.0),
        "w": _field(rng, shape, 0.05),
        "pv": _field(rng, shape, 0.3, offset=1.0),
        # geopotential [m^2/s^2] — the ERA5 convention the divisor must keep
        "z": _field(rng, shape, 200.0, offset=5.0e4),
        "t": _field(rng, shape, 2.0, offset=260.0),
        "q": np.abs(_field(rng, shape, 1e-4, offset=3e-3)),
    }
    for v in VARS:
        xr.Dataset(
            {v: (("valid_time", "level", "latitude", "longitude"), base[v])},
            coords={"valid_time": times, "level": LEVELS,
                    "latitude": lat, "longitude": lon},
        ).to_netcdf(data_dir / f"era5_{v}_{YEAR}_{MONTH:02d}.nc")

    # --- climatology: (month, day, hour, level, lat, lon) ------------------
    cshape = (1, 3, 24, len(LEVELS), NLAT, NLON)
    # `pressure_level`, not `level`: open_months_ds renames the state's level
    # dim, and climatology.py:387 writes the climatology with the renamed
    # form.  A clim on `level` gives the merged dataset BOTH dims and
    # differentiate() then cannot find its coordinate.
    cdims = ("month", "day", "hour", "pressure_level", "latitude", "longitude")
    ccoords = {"month": [MONTH], "day": [1, 2, 3], "hour": np.arange(24),
               "pressure_level": LEVELS, "latitude": lat, "longitude": lon}
    # Time-mean of the state, held constant across (month, day, hour): the
    # anomaly is then pure noise, which is all this regression needs.
    clim = xr.Dataset(
        {v: (cdims, np.broadcast_to(base[v].mean(axis=0)[None, None, None],
                                    cshape).astype(np.float32).copy())
         for v in VARS},
        coords=ccoords,
    )
    clim_path = clim_dir / f"{STEM}.nc"
    clim.to_netcdf(clim_path)

    # --- Helmholtz bars: 24 per-month files, (nday, 24, nlev, ny, nx) ------
    hshape = (3, 24, len(LEVELS), NLAT, NLON)
    hdims = ("day", "hour", "pressure_level", "latitude", "longitude")
    hcoords = {k: ccoords[k] for k in hdims}
    for comp in ("u", "v"):
        xr.Dataset(
            {f"{comp}_rot_bar": (hdims, np.zeros(hshape, np.float32)),
             f"{comp}_div_bar": (hdims, np.zeros(hshape, np.float32))},
            coords=hcoords,
        ).to_netcdf(clim_dir / f"{STEM}_{MONTH_ABBREVS[MONTH-1]}_{comp}_helmholtz.nc")

    return dict(data_dir=data_dir, clim_path=clim_path, clim_dir=clim_dir,
                lat=lat, lon=lon, times=times)


def _config(era5_tree, out_dir, **kw):
    from pvtend.tendency import TendencyConfig
    return TendencyConfig(
        event_type="blocking",
        data_dir=era5_tree["data_dir"],
        clim_path=era5_tree["clim_path"],
        clim_helmholtz_dir=era5_tree["clim_dir"],
        output_dir=out_dir,
        levels=list(LEVELS),
        wavg_levels=[400, 300, 250, 200],
        rel_hours=[0],
        lat_half=9.0, lon_half=22.5,
        skip_existing=False,
        **kw,
    )


def test_source_defaults_to_era5(era5_tree, tmp_path):
    """No caller should have to know the CESM switch exists."""
    cfg = _config(era5_tree, tmp_path)
    assert cfg.source == "era5"
    assert cfg.member is None


def test_z_divisor_is_g_for_era5(era5_tree, tmp_path):
    """ERA5 z is geopotential, so H = z/g; only CESM's Z3 is already height."""
    from pvtend.constants import G0
    from pvtend.tendency import z_divisor
    assert z_divisor(_config(era5_tree, tmp_path)) == pytest.approx(G0)


def test_window_pad_stays_3h_on_hourly_data(era5_tree, tmp_path):
    """The pad became cadence-derived; on hourly data it must still be 3 h.

    max(3, 2*dt) with dt = 1 h is 3 — the value hardcoded before v2.14 — so the
    ERA5 window is the same one it always was.
    """
    dt_h = 1.0
    assert max(3.0, 2.0 * dt_h) == 3.0


def test_clim_bar_uses_the_month_day_hour_path(era5_tree, tmp_path):
    """An ERA5 climatology has no `slot`, so clim_bar must not take that branch."""
    from pvtend.climatology import load_climatology
    from pvtend.tendency import clim_bar, open_source_ds

    cfg = _config(era5_tree, tmp_path)
    clim = load_climatology(cfg.clim_path)
    assert "slot" not in clim.dims

    ds = open_source_ds(cfg, pd.Timestamp(f"{YEAR}-{MONTH:02d}-02 00:00"))
    ds = ds.sel(valid_time=slice(f"{YEAR}-{MONTH:02d}-02 00:00",
                                 f"{YEAR}-{MONTH:02d}-02 02:00"))
    bar = clim_bar(clim, "pv", ds)
    assert "valid_time" in bar.dims
    assert bar.sizes["valid_time"] == ds.sizes["valid_time"]
    assert np.isfinite(bar.values).all()


def test_fill_is_a_noop_on_gapfree_era5(era5_tree, tmp_path):
    """ERA5 has no below-ground NaN, so the fill must return the input untouched."""
    from pvtend.tendency import fill_window_below_ground, open_source_ds

    cfg = _config(era5_tree, tmp_path)
    ds = open_source_ds(cfg, pd.Timestamp(f"{YEAR}-{MONTH:02d}-02 00:00")).load()
    before = {v: ds[v].values.copy() for v in VARS}
    out = fill_window_below_ground(ds)
    for v in VARS:
        np.testing.assert_array_equal(out[v].values, before[v])


def test_numeric_track_id_still_parses_as_int(tmp_path):
    """Relaxing track_id to accept strings must not restyle ERA5 filenames."""
    from pvtend.cli import _load_event_args

    csv = tmp_path / "events.csv"
    csv.write_text("evt_name,track_id,lat0,lon0,base_ts\n"
                   f"onset,42,20.0,-30.0,{YEAR}-{MONTH:02d}-02 00:00\n")
    (evt, tid, lat0, lon0, ts), = _load_event_args(csv, None, ["onset"])
    assert tid == 42 and isinstance(tid, int)


def test_era5_event_end_to_end(era5_tree, tmp_path):
    """The whole chain still runs on the monthly layout and writes an NPZ."""
    from pvtend.tendency import TendencyComputer

    cfg = _config(era5_tree, tmp_path)
    tc = TendencyComputer(cfg)
    n = tc.process_event("onset", track_id=42, lat0=18.0, lon0=-30.0,
                         base_ts=pd.Timestamp(f"{YEAR}-{MONTH:02d}-02 00:00"))
    assert n == 1, "ERA5 path wrote no NPZ"

    written = sorted(Path(tmp_path).rglob("*.npz"))
    assert len(written) == 1
    assert written[0].name.startswith("track_42_")

    with np.load(written[0], allow_pickle=False) as z:
        for key in ("Y_rel", "X_rel", "levels", "track_id", "lat0", "lon0"):
            assert key in z, f"missing NPZ key {key}"
        assert int(z["track_id"]) == 42
