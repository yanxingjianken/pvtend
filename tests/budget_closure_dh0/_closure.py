"""Shared machinery for the four dh=0 PV-budget closure notebooks.

One module so `era5 x {blocking,prp}` and `cesm2 x {blocking,prp}` cannot drift apart on what "LHS",
"RHS" or "closure" mean. The notebooks only choose a dataset and lay out cells.

THE TWO SCHEMAS ARE NOT THE SAME SHAPE
--------------------------------------
    ERA5   stores the individual (wind component) x (PV gradient) PRODUCTS:
           u_bar_pv_bar_dx, u_anom_pv_anom_dx, w_bar_pv_anom_dp, ... (40 horizontal + 15 vertical)
           The RHS has to be ASSEMBLED here.
    CESM2  stores PRE-SUMMED groups: lin_adv, baroclinic, rot_nl, div, vertical.

`Q` MEANS DIFFERENT THINGS IN THE TWO — THIS IS THE TRAP
--------------------------------------------------------
    ERA5   Q = -g(f + zeta) dthetadot_LHR/dp   -- an INDEPENDENT latent-heating source
           (`pvtend/tendency.py:945`). Belongs on the RHS; the residual is then meaningful.
    CESM2  `Q` is `Q_resid` = dqdt - horiz_plus_vert, i.e. a RESIDUAL BY CONSTRUCTION
           (`build_pvbudget_15deg.py`, `v3["Q_resid"] = v3["dqdt"] - v3["horiz_plus_vert"]`).

Put the CESM2 `Q` on the RHS and `LHS - RHS` is identically zero: the closure panel would be a
field of numerical noise and would look like a perfect result. CESM2 therefore uses `Q_lhr`, its own
independently-computed latent-heating source, and the honest residual is `Q_resid_lhr`.

WHAT THE THIRD PANEL ACTUALLY MEASURES
--------------------------------------
    LHS - RHS  is NOT "numerical error". It is everything the budget does not represent: diabatic
    sources beyond the LHR parameterisation, sub-daily rectification the centred daily difference
    misses, interpolation error in the archive, and (for CESM2) the fact that the horizontal
    rotational terms are in FLUX form -div(qV) rather than -V.grad q, which differ discretely by
    -q(div V).

Run from the notebooks:  from _closure import *
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np

OUT = Path("/net/flood/data2/users/x_yan/pvtend/outputs")
LEVEL = 250                      # every panel in these notebooks is at 250 hPa
STAGES = ("onset", "peak", "decay")
DAY = 86400.0
#: UNITS DIFFER BY A FACTOR OF A MILLION between the two archives, and nothing in either npz says so.
#:   ERA5  stores SI potential vorticity per second  (1 PVU = 1e-6 SI)  -> x1e6 x86400
#:   CESM2 stores PVU per second                                        ->       x86400
#: Measured on a single peak event: ERA5 pv_dt RMS = 1.33e-10, CESM2 pv_dt RMS = 1.01e-05.
#: Using one factor for both would put the two datasets 1e6 apart while every plot still "worked".
TO_PVU_DAY = {"era5": 1.0e6 * DAY, "cesm2": DAY}

#: ERA5 RHS = -(sum of these 12 products) + Q. Uses the TOTAL wind (bar + anom) against the TOTAL
#: PV gradient (bar + anom), so u_rot_*/u_div_* must NOT also be included or the wind is counted
#: twice: `u_anom = u_rot_anom + u_div_anom`.
ERA5_PRODUCTS = [
    "u_bar_pv_bar_dx", "u_bar_pv_anom_dx", "u_anom_pv_bar_dx", "u_anom_pv_anom_dx",
    "v_bar_pv_bar_dy", "v_bar_pv_anom_dy", "v_anom_pv_bar_dy", "v_anom_pv_anom_dy",
    "w_bar_pv_bar_dp", "w_bar_pv_anom_dp", "w_anom_pv_bar_dp", "w_anom_pv_anom_dp",
]
#: CESM2 RHS = these + Q_lhr. Already negated and already summed by the builder.
#:
#: `mean_adv` IS INCLUDED, and that is not optional. The LHS is the FULL tendency `pv_dt` in both
#: archives, so the RHS must be the FULL advection. `horiz_plus_vert` deliberately EXCLUDES the
#: mean-on-mean term -Vbar.grad qbar - omegabar dqbar/dp, which the builder stores separately as
#: `mean_adv` (build_pvbudget_15deg.py:503). The ERA5 product list below DOES contain its
#: mean-on-mean members (`u_bar_pv_bar_dx`, ...), so leaving `mean_adv` out of the CESM2 sum would
#: compare two different quantities and make the RHS/LHS ratios meaningless across archives.
CESM_TERMS = ["lin_adv", "baroclinic", "rot_nl", "div", "vertical", "mean_adv"]

DATASETS = {
    "era5_blocking":  dict(dirname="blocking",      kind="era5"),
    "era5_prp":       dict(dirname="prp",           kind="era5"),
    "cesm2_blocking": dict(dirname="cesm_blocking", kind="cesm2"),
    "cesm2_prp":      dict(dirname="cesm_prp",      kind="cesm2"),
}


# ── file listing ────────────────────────────────────────────────────────────────────────────
def list_events(dataset, stage, dh="dh=+0"):
    """Event npz paths for one (dataset, stage). Layout is outputs/<dir>/<stage>/<dh>/track_*.npz."""
    #: `track_*.npz` NOT `*.npz`: the ERA5 trees carry leftover `tmp*.npz` from interrupted writes
    #: (38 of them under outputs/blocking). Globbing `*.npz` picks those up and they raise
    #: BadZipFile — they are not corrupt data, they are incomplete temp files.
    d = OUT / DATASETS[dataset]["dirname"] / stage / dh
    return sorted(glob.glob(str(d / "track_*.npz")))


def rwb_label(path, kind):
    """RWB variant for one event.

    CESM2 carries it in the npz (`label` = AWB / CWB / NEUTRAL / Omega). ERA5 does NOT — its npz
    holds only track_id / ts / dh / center_lat / center_lon, and the variant assignment lives
    outside (pvtend.classify writes a `rwb_variant_tracksets.pkl`, whose location was not resolved
    when these notebooks were written). Until that mapping is wired in, ERA5 events all report
    "ALL", so the ERA5 notebooks show one RWB column rather than five. That is a KNOWN GAP, not a
    statement that ERA5 events are unclassified.
    """
    if kind != "cesm2":
        return "ALL"
    try:
        with np.load(path, allow_pickle=True) as z:
            return str(z["label"]) if "label" in z.files else "ALL"
    except Exception:
        return "ALL"


def level_index(z, level=LEVEL):
    """Row of `level` in this file's own `levels` axis. Index by VALUE — ERA5 has 9 levels
    (1000..100) and CESM2 has 4 (400/300/250/200), so a hard-coded index would silently read a
    different pressure in the two datasets."""
    lv = [int(x) for x in np.asarray(z["levels"]).ravel()]
    if level not in lv:
        raise KeyError(f"{level} hPa not in levels={lv}")
    return lv.index(level)


# ── LHS / RHS for one event ─────────────────────────────────────────────────────────────────
_ERA5_SIGN = {}          # dataset -> +1/-1, detected once (see below)
BAD_FILES = []           # npz that failed to open — see `report_bad()`


def safe_load(path):
    """np.load that survives a corrupt archive.

    The ERA5 output trees contain npz that raise `BadZipFile`. A notebook that dies on the first
    one tells you nothing; one that silently skips them tells you the wrong N. So they are skipped
    AND recorded, and `report_bad()` prints the count so it appears in the notebook output.
    """
    try:
        return np.load(path, allow_pickle=True)
    except Exception as e:
        BAD_FILES.append((path, type(e).__name__))
        return None


def report_bad():
    if BAD_FILES:
        print(f"    [!] {len(BAD_FILES)} unreadable npz skipped, e.g. "
              f"{BAD_FILES[0][0].split('outputs/')[-1]} ({BAD_FILES[0][1]})")
    return len(BAD_FILES)


def first_readable(paths):
    """The first path that actually opens — for grabbing coordinate axes."""
    for p in paths:
        z = safe_load(p)
        if z is not None:
            return p, z
    raise RuntimeError("no readable npz in this selection")


def _era5_sign(dataset, paths, n=25):
    """Whether the stored ERA5 products are `u dq/dx` or already `-u dq/dx`.

    The convention is not documented in the npz, and guessing wrong flips the whole RHS. Rather
    than assume, correlate the LHS against both candidates over a sample and take the better. The
    detected sign is printed so it is on the record rather than buried.
    """
    if dataset in _ERA5_SIGN:
        return _ERA5_SIGN[dataset]
    a = b = 0.0
    for p in paths[:n]:
        try:
            z = safe_load(p)
            if z is None:
                continue
            with z:
                k = level_index(z)
                lhs = np.asarray(z["pv_dt_3d"][k], float)
                s = sum(np.asarray(z[f"{t}_3d"][k], float) for t in ERA5_PRODUCTS)
                q = np.asarray(z["Q_3d"][k], float)
                m = np.isfinite(lhs) & np.isfinite(s) & np.isfinite(q)
                if m.sum() < 50:
                    continue
                a += np.corrcoef(lhs[m], (-s + q)[m])[0, 1]
                b += np.corrcoef(lhs[m], (s + q)[m])[0, 1]
        except Exception:
            continue
    sign = -1.0 if a >= b else +1.0
    _ERA5_SIGN[dataset] = sign
    print(f"    [{dataset}] ERA5 product sign convention detected: RHS = {sign:+.0f}*sum(products) "
          f"+ Q   (mean corr {max(a, b) / max(n, 1):.3f} vs {min(a, b) / max(n, 1):.3f})")
    return sign


def lhs_rhs(path, kind, dataset=None, paths=None):
    """(LHS, RHS, LHS-RHS) at 250 hPa for one event, in PVU/day. NaN-preserving.

    Raises if the file is unreadable — callers that loop use `safe_load` first.
    """
    _z = safe_load(path)
    if _z is None:
        raise RuntimeError(f"unreadable: {path}")
    with _z as z:
        k = level_index(z)
        lhs = np.asarray(z["pv_dt_3d"][k], float)
        if kind == "cesm2":
            # Q_lhr, NOT Q. `Q` here is Q_resid and would close the budget by definition.
            rhs = sum(np.asarray(z[f"{t}_3d"][k], float) for t in CESM_TERMS)
            rhs = rhs + np.asarray(z["Q_lhr_3d"][k], float)
        else:
            s = sum(np.asarray(z[f"{t}_3d"][k], float) for t in ERA5_PRODUCTS)
            rhs = _era5_sign(dataset, paths) * s + np.asarray(z["Q_3d"][k], float)
    f = TO_PVU_DAY[kind]
    return lhs * f, rhs * f, (lhs - rhs) * f


def composite(paths, kind, dataset=None, limit=None):
    """nanmean of (LHS, RHS, LHS-RHS) over events. `limit` subsamples for interactive speed."""
    ps = paths if limit is None else paths[:: max(1, len(paths) // limit)][:limit]
    acc = None
    n = 0
    for p in ps:
        try:
            trio = lhs_rhs(p, kind, dataset, paths)
        except Exception:
            continue
        if acc is None:
            acc = [np.zeros_like(t) for t in trio] + [np.zeros_like(trio[0])]
        for i, t in enumerate(trio):
            good = np.isfinite(t)
            acc[i][good] += t[good]
        acc[3] += np.isfinite(trio[0])
        n += 1
    if acc is None:
        return None, None, None, 0
    c = np.where(acc[3] > 0, acc[3], np.nan)
    return acc[0] / c, acc[1] / c, acc[2] / c, n


# ── plotting ────────────────────────────────────────────────────────────────────────────────
def _axes(z_or_path):
    """(rel_lon, rel_lat) for the panel, from the file's own coordinate vectors."""
    z = np.load(z_or_path, allow_pickle=True) if isinstance(z_or_path, str) else z_or_path
    x = np.asarray(z["X_rel"])[0] if "X_rel" in z.files else np.arange(z["pv_dt"].shape[1])
    y = np.asarray(z["Y_rel"])[:, 0] if "Y_rel" in z.files else np.arange(z["pv_dt"].shape[0])
    return x, y


def panel_row(axrow, trio, x, y, title_prefix, vmax=None):
    """One row of three panels: LHS, RHS, LHS-RHS. All three share ONE symmetric colour scale so
    the residual is readable as a FRACTION of the terms, not renormalised into looking small."""
    import matplotlib.pyplot as plt
    lhs, rhs, res = trio
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(np.concatenate([lhs.ravel(), rhs.ravel()])), 99))
    lev = np.linspace(-vmax, vmax, 21)
    names = [r"LHS  $\partial q/\partial t$", "RHS  (advection + Q)", "LHS $-$ RHS"]
    for ax, f, nm in zip(axrow, (lhs, rhs, res), names):
        cf = ax.contourf(x, y, f, levels=lev, cmap="RdBu_r", extend="both")
        ax.set_aspect("equal")
        r = np.sqrt(np.nanmean(f ** 2))
        ax.set_title(f"{title_prefix} {nm}\nRMS={r:.3f}", fontsize=8.5)
        ax.axhline(0, color="0.4", lw=0.4); ax.axvline(0, color="0.4", lw=0.4)
    m = np.isfinite(lhs) & np.isfinite(rhs)
    corr = float(np.corrcoef(lhs[m], rhs[m])[0, 1]) if m.sum() > 8 else np.nan
    frac = float(np.sqrt(np.nanmean(res ** 2)) / np.sqrt(np.nanmean(lhs ** 2)))
    return cf, vmax, corr, frac


def cell(dataset, stage, rwb="ALL", event_index=0, limit=400, figsize=(11.5, 6.4)):
    """The full 2x3 cell: composite on top, one single event below. Returns the figure."""
    import matplotlib.pyplot as plt
    kind = DATASETS[dataset]["kind"]
    paths = list_events(dataset, stage)
    if rwb != "ALL":
        paths = [p for p in paths if rwb_label(p, kind) == rwb]
    if not paths:
        print(f"  {dataset} {stage} {rwb}: no events"); return None

    cl, cr, cres, n = composite(paths, kind, dataset, limit=limit)
    # a readable event for the single-event row, not just paths[0]
    ev = None
    for j in range(len(paths)):
        try:
            ev = lhs_rhs(paths[(event_index + j) % len(paths)], kind, dataset, paths)
            break
        except Exception:
            continue
    if ev is None:
        print(f"  {dataset} {stage} {rwb}: no readable event"); return None
    _p0, z0 = first_readable(paths)
    with z0:
        x, y = _axes(z0)

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
    cf1, v1, c1, f1 = panel_row(axes[0], (cl, cr, cres), x, y, "composite")
    cf2, v2, c2, f2 = panel_row(axes[1], ev, x, y, "single event")
    for ax in axes[1]:
        ax.set_xlabel("rel. lon [deg]")
    for ax in axes[:, 0]:
        ax.set_ylabel("rel. lat [deg]")
    # Colourbars in explicitly positioned axes. `fig.colorbar(ax=...)` steals space from the axes
    # it is handed and the result depends on the order relative to subplots_adjust, which is how
    # the first version ended up overlapping the panels.
    fig.subplots_adjust(top=0.80, bottom=0.09, left=0.06, right=0.88, hspace=0.16, wspace=0.06)
    for row, cf in ((0, cf1), (1, cf2)):
        bb = [ax.get_position() for ax in axes[row]]
        x1 = max(b.x1 for b in bb); y0 = min(b.y0 for b in bb); y1 = max(b.y1 for b in bb)
        cax = fig.add_axes([x1 + 0.018, y0, 0.014, y1 - y0])
        cb = fig.colorbar(cf, cax=cax); cb.set_label("PVU day$^{-1}$", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{dataset} · {stage} · RWB={rwb} · dh=+0 · {LEVEL} hPa\n"
        f"composite (N={n}): corr(LHS,RHS)={c1:.3f}, RMS(LHS−RHS)/RMS(LHS)={f1:.2f}   |   "
        f"single event: corr={c2:.3f}, {f2:.2f}\n"
        f"RHS = "
        + ("lin_adv+baroclinic+rot_nl+div+vertical+mean_adv + Q_lhr   (NOT Q — that is Q_resid)"
           if kind == "cesm2" else
           "−Σ(12 wind×∇q products) + Q  (Q is the independent LHR source)"),
        fontsize=9)
    report_bad()
    return fig


def rwb_variants(dataset, stage):
    """RWB variants present, ALL first. ERA5 returns just ("ALL",) — see `rwb_label`."""
    kind = DATASETS[dataset]["kind"]
    if kind != "cesm2":
        return ("ALL",)
    seen = set()
    for p in list_events(dataset, stage)[::37]:
        seen.add(rwb_label(p, kind))
    return ("ALL",) + tuple(sorted(seen - {"ALL"}))
