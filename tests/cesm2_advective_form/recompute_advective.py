#!/usr/bin/env python3
"""Recompute the CESM2 PV budget the ERA5 way, on a few dozen events, to find where RHS >> LHS.

The stored CESM2 budget has RMS(RHS) ~ 3-5x RMS(LHS) at the single-event level and a grid-scale
speckle the LHS does not have, while ERA5 on the same diagnostic sits at RHS/LHS ~ 1 with LHS and
RHS visually the same field. Four things differ between the two pipelines; this script turns each
one off in turn so the responsible one is identified rather than guessed.

    A  operator      CESM2 uses FLUX form -div(qV) for the rotational terms; ERA5 uses ADVECTIVE
                     -V.grad q for everything.
    B  wind          CESM2 advects with the PPVI BALANCED wind reconstructed from the UPPER PIECE
                     ONLY (build_pvbudget_15deg.py:300-325) -- measured 10% too strong and missing
                     the lower/surface pieces. ERA5 advects with the actual reanalysis wind.
    C  timestep      CESM2's dqdt is a 2-day centred difference of DAILY data; ERA5's is hourly.
                     A daily difference is a heavy low-pass; the RHS is instantaneous.
    D  event hour    CESM2 matches the archive by (year, month, day) ONLY (`:359`), but 71% of
                     events are tagged 06/12/18Z in the CSV. The tracked centre and the PV field
                     are then up to 18 h apart.

VARIANTS COMPUTED (all at 250 hPa, on the same events)
    stored      the budget as delivered, read back from outputs/cesm_blocking
    adv_ppvi    advective form, PPVI wind          -> isolates A
    adv_model   advective form, ACTUAL model wind  -> isolates A+B  (this is the ERA5 recipe)
    flux_model  flux form, actual model wind       -> isolates B alone

LHS is the same daily centred difference in all variants, so C is held fixed and shows up as a
floor common to all four. D is reported separately as the |hour| of each event.

Run:  micromamba run -n blocking python recompute_advective.py --n 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, "/net/flood/data2/users/x_yan/cesm-blocking/08_triad_resonance")
import build_pvbudget_15deg as Bg                                     # noqa: E402

HERE = Path(__file__).resolve().parent
LEVEL = 250
KLEV = Bg.WU.index(LEVEL)
R = 6371000.0


def _csv_hour(group):
    """(member, track, year, month, day) -> hour, so the phase error can be reported per event."""
    c = pd.read_csv(Bg.CSV[group])
    if "hour" not in c:
        return {}
    key = ["member", "track", "year", "month", "day"] if "member" in c else None
    if key is None:
        return {}
    return {tuple(int(v) for v in r[:5]): int(r[5])
            for r in c[key + ["hour"]].itertuples(index=False, name=None)}


def one_event(ds, dr, clim, rec, lat, lon, jb, band, coslat):
    """All four RHS variants + the LHS for one event, at 250 hPa, on the block-relative window."""
    didx = {(int(t.year), int(t.month), int(t.day)): i for i, t in enumerate(ds["time"].values)}
    ti = didx.get((int(rec["year"]), int(rec["month"]), int(rec["day"])))
    if ti is None or ti - 1 < 0 or ti + 1 >= ds.sizes["time"]:
        return None
    mo, di = int(rec["month"]), int(rec["day"]) - 1
    cm = clim[mo]
    clat, clon = float(rec["lat"]), float(rec["lon180"]) % 360.0
    ic = int(np.abs(((lon - clon + 180) % 360) - 180).argmin())
    iw = (np.arange(-Bg.WIN_PAD, Bg.WIN_PAD + 1) + ic) % lon.size
    jcen = int(np.abs(band - clat).argmin())
    W = lambda a: a[..., jb, :][..., iw]

    L = [KLEV]
    pv = W(ds["pv"].sel(lev=Bg.WU).isel(time=ti).load().values)
    u = W(ds["u"].sel(lev=Bg.WU).isel(time=ti).load().values)
    v = W(ds["v"].sel(lev=Bg.WU).isel(time=ti).load().values)
    om = W(ds["omega"].sel(lev=Bg.WU).isel(time=ti).load().values)
    pvp = W(ds["pv"].sel(lev=Bg.WU).isel(time=ti + 1).load().values)
    pvm = W(ds["pv"].sel(lev=Bg.WU).isel(time=ti - 1).load().values)
    pvb = W(cm["pv"][di])

    # actual archive time spacing, NOT the hard-coded 86400 — the noleap calendar leaves one
    # 48 h gap per leap year and the stored budget divides those by 2*86400 regardless.
    tt = ds["time"].values
    dt_s = float((tt[ti + 1] - tt[ti - 1]) / np.timedelta64(1, "s"))

    # ---- LHS: full tendency, centred, at the TRUE spacing ----
    lhs = (pvp[KLEV] - pvm[KLEV]) / dt_s
    lhs_hard = (pvp[KLEV] - pvm[KLEV]) / (2 * 86400.0)          # what the pipeline actually uses

    # ---- gradients (same operators as the builder) ----
    dx = (R * coslat * np.deg2rad(Bg.DLON))[:, None]
    DY = np.deg2rad(Bg.DLAT) * R
    gx = lambda f: np.gradient(f, axis=1) / dx
    gy = lambda f: -np.gradient(f, axis=0) / DY
    dqdp = np.gradient(pv, Bg.LEV_PA, axis=0)[KLEV]

    q = pv[KLEV]
    adv = lambda uu, vv: -(uu * gx(q) + vv * gy(q)) - om[KLEV] * dqdp
    cl3 = coslat[:, None]
    flux = lambda uu, vv: -(gx(uu * q) + gy(vv * cl3 * q) / cl3) - om[KLEV] * dqdp

    # ---- PPVI wind (block+eddy = the pipeline's total upper-piece balanced wind) ----
    ev = {k: Bg._win(ds[k].sel(lev=Bg.WU).isel(time=ti).load().values, jb, iw).astype(float)
          for k in ("z", "t", "u", "v")}
    mn = {k: Bg._win(cm[k][di], jb, iw).astype(float) for k in ("z", "t", "u", "v")}
    ev = dict(zip(("z", "t", "u", "v"), Bg.fill_below_ground(*[ev[k] for k in ("z", "t", "u", "v")]))) \
        if hasattr(Bg, "fill_below_ground") else ev
    try:
        iv = Bg.PB._prep_inversion(ev["z"], ev["t"], ev["u"], ev["v"],
                                   mn["z"], mn["t"], mn["u"], mn["v"], band, Bg.DLON)
        psi_t = Bg.PB._sum_pieces(iv, np.asfortranarray(iv.q_e, np.float32), iv.THTIN, "upper")
        up, vp = Bg.psi_to_winds(psi_t, band, Bg.DLON, Bg.DLAT)
        up, vp = up[KLEV], vp[KLEV]
    except Exception:
        up = vp = None

    out = {"lhs": lhs, "lhs_hard": lhs_hard, "dt_s": dt_s,
           "adv_model": adv(u[KLEV], v[KLEV]), "flux_model": flux(u[KLEV], v[KLEV])}
    if up is not None:
        out["adv_ppvi"] = adv(up, vp)
        out["flux_ppvi"] = flux(up, vp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--group", default="block")
    ap.add_argument("--stage", default="peak")
    args = ap.parse_args()

    Bg._load_clim15()
    rows, _ = Bg.build_tasks([args.group])
    d = rows[(rows["group"] == args.group) & (rows["stage"] == args.stage)]
    rng = np.random.default_rng(0)
    sel = d.iloc[rng.choice(len(d), size=min(args.n * 3, len(d)), replace=False)]
    hours = _csv_hour(args.group)

    res, nh = [], []
    for (mem, dec), g in sel.groupby(["member", "decade"]):
        fp = Bg.PVDIR / f"m{mem}" / f"cesm2_lens2_pv9_15deg_m{mem:02d}_d{dec}.nc"
        if not fp.exists():
            continue
        with xr.open_dataset(fp) as ds:
            lat, lon = ds["lat"].values, ds["lon"].values
            jb = np.where((lat >= Bg.BAND_S) & (lat <= Bg.BAND_N))[0][::-1]
            band = lat[jb]
            coslat = np.maximum(np.cos(np.deg2rad(band)), np.cos(np.deg2rad(89.5)))
            for r in g.to_dict("records"):
                o = one_event(ds, None, Bg._CLIM15, r, lat, lon, jb, band, coslat)
                if o is None:
                    continue
                res.append(o)
                nh.append(hours.get((int(r["member"]), int(r["track"]), int(r["year"]),
                                     int(r["month"]), int(r["day"])), -1))
                if len(res) >= args.n:
                    break
        if len(res) >= args.n:
            break

    print(f"\n{len(res)} events, {args.group} {args.stage}, {LEVEL} hPa, PVU/day\n")
    rms = lambda a: float(np.sqrt(np.nanmean(np.asarray(a) ** 2)))

    def score(key):
        cs, fr = [], []
        for o in res:
            if key not in o:
                continue
            l, r_ = o["lhs"] * 86400, o[key] * 86400
            m = np.isfinite(l) & np.isfinite(r_)
            if m.sum() < 50:
                continue
            cs.append(np.corrcoef(l[m], r_[m])[0, 1])
            fr.append(rms(r_) / rms(l))
        return (np.nanmedian(cs), np.nanmedian(fr), len(cs)) if cs else (np.nan, np.nan, 0)

    print(f"  {'variant':12s} {'corr(LHS,RHS)':>14s} {'RMS RHS / RMS LHS':>19s}   what it isolates")
    print("  " + "-" * 80)
    for k, note in (("adv_ppvi",   "advective + PPVI wind      -> operator only"),
                    ("flux_ppvi",  "flux + PPVI wind           -> the stored recipe, rebuilt"),
                    ("adv_model",  "advective + MODEL wind     -> the ERA5 recipe"),
                    ("flux_model", "flux + MODEL wind          -> wind only")):
        c, f, n = score(k)
        print(f"  {k:12s} {c:14.3f} {f:19.2f}   {note}  (n={n})")

    print()
    dts = np.array([o["dt_s"] for o in res]) / 86400
    print(f"  LHS spacing: {np.sum(dts == 2.0)}/{len(dts)} events at the nominal 2 days; "
          f"others {sorted(set(dts[dts != 2.0]))}")
    d_hard = [rms((o['lhs'] - o['lhs_hard']) * 86400) / rms(o['lhs'] * 86400) for o in res]
    print(f"  hard-coded 2*86400 vs true spacing: median relative error {np.median(d_hard):.4f}")
    nh = np.array(nh)
    if (nh >= 0).any():
        import collections
        print(f"  event hour (field is always 00Z): {dict(collections.Counter(nh[nh >= 0]))}")
        print(f"  -> median phase error {np.median(nh[nh >= 0]):.0f} h between centre and field")


if __name__ == "__main__":
    main()
