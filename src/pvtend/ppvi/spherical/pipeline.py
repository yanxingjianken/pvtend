"""One event, end to end: hemisphere of data in, per-level induced winds out.

The output key names are the ones downstream consumers expect, so a composite
built from these files reads the same names and means the same thing by them.
The residual key holds the unbalanced part alone -- what the balance equations
cannot represent, and nothing else.  An optional rotated-frame track carries the
patch for events whose box would otherwise run past the pole.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import InversionConfig
from .levels import LevelSet, build_levels
from .passab import ertel_pv_si
from .passd import PiecewiseResult, default_pieces, invert_pieces
from .mirror import mirrored_latitudes
from .prepare import prepare_state
from .sht import SHT, gaussian_grid, grid_from_axes
from .sphere import SphereOps
from .winds import (
    geographic_patch,
    rotated_patch,
    rotational_wind_stack,
    to_cartesian,
)

#: Levels averaged for the two-dimensional summary of each piece, and the scale
#: height weighting them.  Upper-tropospheric levels weighted by ``exp(-z/H)`` is
#: the convention these summaries are read against downstream, so it is fixed here
#: rather than exposed as a knob a batch could set two ways.
WAVG_LEVELS = (400, 300, 250, 200)
H_SCALE = 7000.0


def winds_on_regular_grid(
    ops: SphereOps, out_sht: SHT, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Move a wind pair from the Gaussian solver grid to a regular one.

    Carried as ``u cos(lat)`` and ``v cos(lat)``, which are band limited, so the
    transfer is exact; the components themselves are not, and transferring them
    directly loses several digits.

    The output grid's pole rows come back as NaN.  There the components depend on
    which meridian you approach along, so no single pair of numbers is right --
    the wind is perfectly well defined, its *components* are not.  Cropping is
    done on a regular grid rather than on the solver's Gaussian one because both
    croppings index latitude linearly, and an uneven axis would misplace every row
    but the centre while still looking plausible.
    """
    cos_solver = ops.grid.cos_lat[:, None]
    cos_out = out_sht.grid.cos_lat[:, None]
    u_out = ops.sht.regrid_to(out_sht, u * cos_solver)
    v_out = ops.sht.regrid_to(out_sht, v * cos_solver)
    usable = np.abs(cos_out) > 1e-8
    u_final = np.full_like(u_out, np.nan)
    v_final = np.full_like(v_out, np.nan)
    np.divide(u_out, cos_out, out=u_final, where=usable)
    np.divide(v_out, cos_out, out=v_final, where=usable)
    return u_final, v_final


def dealiased_truncation(nlat: int) -> int:
    """Largest truncation a grid carries without aliasing quadratic terms.

    The classical three-halves rule.  Taking instead the largest truncation the
    grid can *represent* aliases every product in the balance and
    potential-vorticity equations, and costs several times the work per solve for
    the privilege -- at 96 latitudes that is truncation 95 rather than 63, and
    about four times the time per event.
    """
    return max(1, (2 * nlat) // 3 - 1)


@dataclass
class EventOutput:
    """Everything one event contributes to a batch."""

    arrays: dict[str, np.ndarray]
    meta: dict
    result: PiecewiseResult = field(repr=False, default=None)


def _key(component: str, piece: str, suffix: str) -> str:
    """Output key for one component of one piece.

    The residual sits outside the piece namespace by convention:
    ``u_rot_anom_residual_ppvi`` rather than ``u_rot_anom_ppvi_residual``.  Loaders
    enumerate the pieces by the ``u_rot_anom_ppvi_`` prefix, so a residual named
    inside that namespace would be picked up as a piece and the piece sum would
    then count the unbalanced part twice over.

    The observed anomaly the pieces are meant to account for keeps its plain name,
    ``u_rot_anom``, for the same reason.
    """
    if piece == "residual":
        return f"{component}_rot_anom_residual_ppvi{suffix}"
    if piece == "observed":
        return f"{component}_rot_anom{suffix}"
    return f"{component}_rot_anom_ppvi_{piece}{suffix}"


def _weighted_column_mean(
    values: np.ndarray, height: np.ndarray, levels: LevelSet
) -> np.ndarray:
    """Average over the upper-tropospheric levels, weighted by ``exp(-z/H)``.

    Divided by the weight of the levels that were actually valid at each point,
    never by the full count: filling a gap with zero and dividing by everything
    silently pulls the answer towards zero while keeping it finite, which is far
    harder to notice than a NaN.
    """
    idx = [int(np.where(levels.p_hpa == float(p))[0][0]) for p in WAVG_LEVELS]
    weights = np.exp(-height[idx] / H_SCALE)
    valid = np.isfinite(values[idx]) & np.isfinite(weights)
    weights = np.where(valid, weights, 0.0)
    numerator = np.nansum(np.where(valid, values[idx], 0.0) * weights, axis=0)
    denominator = np.nansum(weights, axis=0)
    out = np.full(numerator.shape, np.nan)
    good = denominator > 0
    out[good] = numerator[good] / denominator[good]
    return out


def _piece_spec(ops, levels, mean, event, cfg, mode, lat0, lon0):
    """Piece definitions and, for the scale mode, the sources that separate them.

    The two upper scale pieces share a level list and differ only in the source
    they are handed, which is why the split travels as an override.  The top
    boundary temperature is split along with the interior potential vorticity: a
    piece defined by scale has to divide every source it carries, or its parts
    stop summing to it.
    """
    if mode == "per_level":
        return default_pieces(levels), None, None, None
    if mode != "scale":
        raise ValueError(f"unknown pieces_mode {mode!r}; expected per_level or scale")

    from .qmin import floor_pv
    from .scale_split import scale_pieces

    weights = ops.grid.weights
    q_event, _ = floor_pv(
        event.q_hat, levels, weights, cfg.pv_floor.qmin_pieces, cfg.clamps.mode
    )
    q_mean, _ = floor_pv(
        mean.q_hat, levels, weights, cfg.pv_floor.qmin_pieces, cfg.clamps.mode
    )
    q_anom = q_event - q_mean
    theta_top_anom = event.theta_top - mean.theta_top

    interior = list(levels.interior)
    upper_positions = [i for i, k in enumerate(interior) if levels.p_hpa[k] <= 400.0]
    sources, split = scale_pieces(
        ops, q_anom, theta_top_anom, upper_positions, lat0, lon0
    )

    upper_levels = [k + 1 for k in interior if levels.p_hpa[k] <= 400.0]
    pieces = {
        "surface": [1],
        "lower": [k + 1 for k in interior if levels.p_hpa[k] > 400.0],
        "upper_p": upper_levels + [levels.nlev],
        "upper_e": upper_levels + [levels.nlev],
    }
    qp_overrides = {
        "lower": sources["lower"],
        "upper_p": sources["upper_p"],
        "upper_e": sources["upper_e"],
    }
    th_overrides = {
        "lower": (None, None),
        "upper_p": (None, split["theta_p"]),
        "upper_e": (None, split["theta_e"]),
    }
    return pieces, qp_overrides, th_overrides, split


def invert_event(
    mean_fields: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    event_fields: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lat_nh: np.ndarray,
    lon: np.ndarray,
    centre: tuple[float, float],
    cfg: InversionConfig | None = None,
    lat_half: float = 30.0,
    lon_half: float = 60.0,
    solver_nlat: int = 128,
    solver_nlon: int = 256,
    lmax: int | None = None,
    rotated_track: bool = False,
    keep_result: bool = False,
    pieces_mode: str = "per_level",
) -> EventOutput:
    """Invert one event and crop the pieces around its centre.

    Args:
        mean_fields: ``(height, temperature, u, v)`` for the climatology, each
            ``(nlev, nlat_nh, nlon)`` and bottom-up.
        event_fields: The same for the event.
        lat_nh, lon: Axes of those fields; latitudes ascend from the equator.
        centre: ``(lat, lon)`` of the event in degrees.
        cfg: Solver configuration.
        lat_half, lon_half: Half-widths of the output patch in degrees.
        solver_nlat, solver_nlon: Gaussian solver grid.
        lmax: Truncation; defaults to what the solver grid resolves.
        rotated_track: Also write the rotated-frame patch, which stays complete
            for a centre near or past the pole.
        keep_result: Attach the full inversion to the output, for diagnostics.
        pieces_mode: ``"per_level"`` for one piece per source, or ``"scale"`` for
            the four-way surface / lower / planetary-upper / eddy-upper split.  The
            two write disjoint key sets and must not be mixed in one output
            directory.

    Returns:
        An :class:`EventOutput` whose ``arrays`` are ready to save.
    """
    cfg = cfg or InversionConfig()
    levels = build_levels(cfg.levels)
    grid = gaussian_grid(solver_nlat, solver_nlon)
    if lmax is None:
        lmax = cfg.lmax if cfg.lmax is not None else dealiased_truncation(solver_nlat)
    ops = SphereOps(SHT(grid, lmax=lmax))
    # Output on the data's own grid, which is regular and is what the patch
    # coordinates are quoted against.
    out_grid = grid_from_axes(mirrored_latitudes(lat_nh), lon)
    out_sht = SHT(out_grid, lmax=lmax)

    mean = prepare_state(
        *mean_fields, lat_nh, lon, levels, ops, cfg.mirror.f_floor_deg
    )
    event = prepare_state(
        *event_fields, lat_nh, lon, levels, ops, cfg.mirror.f_floor_deg
    )
    pieces, qp_overrides, th_overrides, split = _piece_spec(
        ops, levels, mean, event, cfg, pieces_mode, lat0=centre[0], lon0=centre[1]
    )
    result = invert_pieces(
        ops,
        levels,
        mean,
        event,
        cfg=cfg,
        pieces=pieces,
        qp_overrides=qp_overrides,
        th_overrides=th_overrides,
    )

    # The observed anomaly the pieces are meant to account for, and the height
    # field the column average is weighted by.
    u_obs_solver, v_obs_solver = rotational_wind_stack(
        ops, event.psi_spec - mean.psi_spec
    )
    u_obs, v_obs = winds_on_regular_grid(ops, out_sht, u_obs_solver, v_obs_solver)
    height = (
        np.stack([out_sht.synthesize(event.phi_spec[k]) for k in range(levels.nlev)])
        / 9.81
    )

    lat0, lon0 = float(centre[0]), float(centre[1])
    scaffold = out_grid.lat < float(np.min(lat_nh)) - 1e-9
    arrays: dict[str, np.ndarray] = {}
    summed_u = np.zeros_like(u_obs)
    summed_v = np.zeros_like(v_obs)

    def cartesian_on_output(u_solver: np.ndarray, v_solver: np.ndarray):
        """Cartesian wind components on the output grid, finite at the poles.

        For a band-limited streamfunction these are band-limited scalars, so the
        transfer is a spectral one and exact -- and unlike the eastward and
        northward components they are defined on the pole rows, which the rotated
        cropping needs.
        """
        parts = to_cartesian(u_solver, v_solver, ops.grid.lat, ops.grid.lon)
        return tuple(ops.sht.regrid_to(out_sht, part) for part in parts)

    cartesian_observed = (
        cartesian_on_output(u_obs_solver, v_obs_solver) if rotated_track else None
    )

    def blank_scaffold(field: np.ndarray) -> np.ndarray:
        """Erase the rows the mirror invented.

        The output grid spans both hemispheres, and south of the data's own first
        row every value is a reflection of the north: smooth, correctly scaled and
        entirely fabricated.  A patch centred below about 42 degrees reaches into
        it, and those rows would otherwise enter a composite as if they were
        observations.
        """
        out = np.array(field, dtype=float, copy=True)
        out[..., scaffold, :] = np.nan
        return out

    def store(
        name: str,
        u_field: np.ndarray,
        v_field: np.ndarray,
        cartesian=None,
    ) -> None:
        u_field = blank_scaffold(u_field)
        v_field = blank_scaffold(v_field)
        for label, values in (("u", u_field), ("v", v_field)):
            patch3d = geographic_patch(
                values, out_grid.lat, out_grid.lon, lat0, lon0, lat_half, lon_half
            )
            arrays[_key(label, name, "_3d")] = patch3d.values.astype(np.float32)
            column = _weighted_column_mean(values, height, levels)
            patch2d = geographic_patch(
                column, out_grid.lat, out_grid.lon, lat0, lon0, lat_half, lon_half
            )
            arrays[_key(label, name, "")] = patch2d.values.astype(np.float32)
            arrays.setdefault("lat_rel", patch2d.lat_rel)
            arrays.setdefault("lon_rel", patch2d.lon_rel)
            arrays.setdefault("lat_vec", patch2d.lat)
            arrays.setdefault("lon_vec", patch2d.lon)
        if rotated_track:
            u_r, v_r = rotated_patch(
                u_field,
                v_field,
                out_grid.lat,
                out_grid.lon,
                lat0,
                lon0,
                lat_half,
                lon_half,
                cartesian=cartesian,
            )
            arrays[_key("u", name, "_3d_rot")] = u_r.values.astype(np.float32)
            arrays[_key("v", name, "_3d_rot")] = v_r.values.astype(np.float32)
            arrays.setdefault("lat_rel_rot", u_r.lat_rel)
            arrays.setdefault("lon_rel_rot", u_r.lon_rel)

    summed_cartesian = None
    for name, piece in result.pieces.items():
        u_solver, v_solver = rotational_wind_stack(ops, piece.psi_spec)
        u_piece, v_piece = winds_on_regular_grid(ops, out_sht, u_solver, v_solver)
        summed_u += u_piece
        summed_v += v_piece
        cart = cartesian_on_output(u_solver, v_solver) if rotated_track else None
        if rotated_track:
            summed_cartesian = (
                cart
                if summed_cartesian is None
                else tuple(a + b for a, b in zip(summed_cartesian, cart))
            )
        store(name, u_piece, v_piece, cart)

    # What the balance equations could not represent: the divergent and
    # unbalanced part, and nothing else.  Named outside the piece namespace so a
    # loader enumerating pieces by prefix does not pick it up as one; see `_key`.
    residual_cartesian = (
        tuple(a - b for a, b in zip(cartesian_observed, summed_cartesian))
        if rotated_track
        else None
    )
    store("residual", u_obs - summed_u, v_obs - summed_v, residual_cartesian)

    # The anomaly the pieces are accounting for.  Stored so that the closure --
    # every piece plus the residual against this field -- can be checked from the
    # file alone rather than trusted, and so that a file states which anomaly it
    # was aiming at instead of leaving that to be reconstructed.
    store("observed", u_obs, v_obs, cartesian_observed)

    # Converted to SI before it is written, never left as the solver's raw
    # right-hand side.  The two differ by `pv_rhs_scale`, which is 31.6 at 850 hPa
    # falling to 11.3 at 200: smooth in the vertical, so writing the unconverted
    # field would show up as a plausible physical profile rather than as the unit
    # error it is.
    pv_anom = np.full((levels.nlev, out_grid.nlat, out_grid.nlon), np.nan)
    anomaly_si = ertel_pv_si(event.q_hat - mean.q_hat, levels)
    for position, k in enumerate(levels.interior):
        pv_anom[k] = out_sht.synthesize(ops.analyze(anomaly_si[position]))
    pv_anom[..., scaffold, :] = np.nan
    pv_patch = geographic_patch(
        pv_anom, out_grid.lat, out_grid.lon, lat0, lon0, lat_half, lon_half
    )
    arrays["pv_anom_wu_3d"] = pv_patch.values.astype(np.float32)

    meta = {
        "pieces_mode": pieces_mode,
        **(
            {
                "split_q_min": split["q_min"],
                "split_contour": split["contour"],
                "split_object_fraction": split["object_fraction"],
                "split_top_fraction": split["top_fraction"],
            }
            if split
            else {}
        ),
        "data_lat_min": float(np.min(lat_nh)),
        "centre_lat": lat0,
        "centre_lon": lon0,
        "levels": cfg.levels,
        "level_pressures": levels.p_hpa,
        "piece_names": list(result.pieces),
        "solver_grid": (solver_nlat, solver_nlon),
        "lmax": ops.sht.lmax,
        "newton_steps": result.newton_steps,
        "newton_converged": result.diagnostics["newton_converged"],
        "newton_final_increment_m": result.diagnostics["newton_final_increment_m"],
        "linear_iterations": result.diagnostics["linear_iterations"],
        "clamp_worst_fraction": result.clamp_worst,
        "pv_floor_fraction_event": result.floor_event.fraction,
        "pv_floor_fraction_mean": result.floor_mean.fraction,
        "all_pieces_converged": all(
            p.report.converged for p in result.pieces.values()
        ),
    }
    return EventOutput(
        arrays=arrays, meta=meta, result=result if keep_result else None
    )
