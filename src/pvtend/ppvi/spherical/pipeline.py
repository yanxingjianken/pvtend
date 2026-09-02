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
from .mirror import mirrored_latitudes, restrict_to_nh
from .prepare import prepare_state
from .sht import SHT, Grid, gaussian_grid, grid_from_axes
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


def rotational_wind_on_regular_grid(
    out_sht: SHT, psi_spec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(u, v)`` on the output grid, evaluated from the streamfunction there.

    Evaluated, not transferred.  ``synthesize_dlat`` returns ``cos(lat) d/dlat``
    as a combination of Legendre functions and divides by nothing, so
    ``u cos(lat)`` and ``v cos(lat)`` come out of the same spectrum the solver
    holds, at the output latitudes, with no intermediate grid.

    The predecessor formed the wind on the solver grid, multiplied by
    ``cos(lat)``, projected that onto degrees up to ``lmax``, and synthesised the
    projection here.  The projection is where it went wrong.  ``u cos(lat)`` is a
    combination of the functions ``cos(lat) dP(n,m)/dlat``, and each of those is
    a combination of ``P(n-1,m)`` and ``P(n+1,m)``: the field carries degree
    ``lmax + 1``, and the projection discards it.  What is discarded does not
    vanish at the pole -- ``P(L+1,0)`` is one there -- while the division by
    ``cos(lat)`` that follows does, so a roughly constant error of order a tenth
    of a metre per second in ``u cos(lat)`` became a relative error growing like
    one over the cosine.  Measured against this function on a Greenland event:
    0.8 per cent at 55 N, 5.8 per cent at 85 N, and **39 per cent at 89 N**,
    worst on the two upper pieces, which carry the most small-scale structure and
    so the most of the discarded degree.

    The pole rows still come back as NaN, for a different and unavoidable reason:
    the eastward direction is defined by a meridian, and every meridian meets
    there, so no pair of components is the right one.  The wind is defined; its
    components are not.  Use the rotated-frame output, which carries the
    Cartesian components, if the pole itself is needed.
    """
    cos_out = out_sht.grid.cos_lat[:, None]
    u_cos = -out_sht.synthesize_dlat(psi_spec) / out_sht.radius
    v_cos = out_sht.synthesize(out_sht.dlon_spec(psi_spec)) / out_sht.radius
    usable = np.abs(cos_out) > 1e-8
    u_final = np.full_like(u_cos, np.nan)
    v_final = np.full_like(v_cos, np.nan)
    np.divide(u_cos, cos_out, out=u_final, where=usable)
    np.divide(v_cos, cos_out, out=v_final, where=usable)
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


@dataclass
class SphereEngine:
    """The transforms one data grid needs, built once and reused across events.

    The spectral tables are a fixed cost per grid, not per event: a batch over one
    archive builds them once per worker and inverts every state through the same
    object.  ``out_sht`` is on the data's own regular grid mirrored onto the
    sphere -- the grid the pieces are delivered on and the one the state
    preparation regrids from, so it serves both.
    """

    levels: LevelSet
    ops: SphereOps
    out_grid: Grid
    out_sht: SHT
    lat_nh: np.ndarray
    lon: np.ndarray
    cfg: InversionConfig
    solver_nlat: int
    solver_nlon: int

    @classmethod
    def build(
        cls,
        lat_nh: np.ndarray,
        lon: np.ndarray,
        cfg: InversionConfig | None = None,
        solver_nlat: int = 128,
        solver_nlon: int = 256,
        lmax: int | None = None,
    ) -> "SphereEngine":
        """Build the engine for one data grid.

        Args:
            lat_nh: Northern latitudes of the data, ascending from the equator.
            lon: Longitudes in ``[0, 360)``.
            cfg: Solver configuration; the default otherwise.
            solver_nlat, solver_nlon: Gaussian solver grid.
            lmax: Truncation; defaults to what the solver grid resolves without
                aliasing the quadratic terms.
        """
        cfg = cfg or InversionConfig()
        levels = build_levels(cfg.levels)
        grid = gaussian_grid(solver_nlat, solver_nlon)
        if lmax is None:
            lmax = cfg.lmax if cfg.lmax is not None else dealiased_truncation(solver_nlat)
        ops = SphereOps(SHT(grid, lmax=lmax))
        lat_nh = np.asarray(lat_nh, dtype=float)
        lon = np.asarray(lon, dtype=float)
        out_grid = grid_from_axes(mirrored_latitudes(lat_nh), lon)
        out_sht = SHT(out_grid, lmax=lmax, radius=ops.sht.radius)
        return cls(
            levels=levels,
            ops=ops,
            out_grid=out_grid,
            out_sht=out_sht,
            lat_nh=lat_nh,
            lon=lon,
            cfg=cfg,
            solver_nlat=int(solver_nlat),
            solver_nlon=int(solver_nlon),
        )

    def fits(self, lat_nh: np.ndarray, lon: np.ndarray) -> bool:
        """Whether this engine was built for these axes."""
        lat_nh = np.asarray(lat_nh, dtype=float)
        lon = np.asarray(lon, dtype=float)
        return (
            lat_nh.shape == self.lat_nh.shape
            and lon.shape == self.lon.shape
            and bool(np.allclose(lat_nh, self.lat_nh, atol=1e-6))
            and bool(np.allclose(lon, self.lon, atol=1e-6))
        )


@dataclass
class HemisphereInversion:
    """One event's pieces on the data's own grid, before any cropping.

    Every field is on the mirrored output grid, both hemispheres, ``(nlev, nlat,
    nlon)``; the rows south of the data's first latitude are the mirror's
    invention and :meth:`northern` drops them.  Winds are the eastward and
    northward components, NaN on a pole row where those are undefined.  The
    pieces are stored unblanked, as they are summed: the residual is the observed
    anomaly minus the pieces in the order they were solved.
    """

    piece_u: dict[str, np.ndarray]
    piece_v: dict[str, np.ndarray]
    u_obs: np.ndarray
    v_obs: np.ndarray
    u_residual: np.ndarray
    v_residual: np.ndarray
    height: np.ndarray
    pv_anom: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    lat_nh: np.ndarray
    meta: dict
    split: dict | None = None
    result: PiecewiseResult | None = field(repr=False, default=None)
    piece_cartesian: dict | None = field(repr=False, default=None)
    observed_cartesian: tuple | None = field(repr=False, default=None)
    residual_cartesian: tuple | None = field(repr=False, default=None)

    def northern(self, values: np.ndarray) -> np.ndarray:
        """The rows the data supplied: latitude ascending from the equator."""
        return restrict_to_nh(values, self.lat_nh)

    def piece_names(self) -> list[str]:
        return list(self.piece_u)


def invert_hemisphere(
    engine: SphereEngine,
    mean_fields: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    event_fields: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    centre: tuple[float, float],
    pieces_mode: str = "per_level",
    rotated_track: bool = False,
    keep_result: bool = False,
) -> HemisphereInversion:
    """Invert one event and deliver the pieces on the data's own grid.

    Args:
        engine: Transforms for the data grid, from :meth:`SphereEngine.build`.
        mean_fields: ``(height, temperature, u, v)`` for the climatology, each
            ``(nlev, nlat_nh, nlon)`` and bottom-up, on the engine's axes.
        event_fields: The same for the event.
        centre: ``(lat, lon)`` of the event in degrees; the scale split seeds its
            planetary object there.
        pieces_mode: ``"per_level"`` for one piece per source, or ``"scale"`` for
            the four-way surface / lower / planetary-upper / eddy-upper split.
        rotated_track: Also form the Cartesian components every piece needs for a
            rotated-frame patch.
        keep_result: Attach the full inversion, for diagnostics.

    Returns:
        A :class:`HemisphereInversion`.
    """
    cfg = engine.cfg
    levels = engine.levels
    ops = engine.ops
    out_grid = engine.out_grid
    out_sht = engine.out_sht
    lat_nh = engine.lat_nh
    lon = engine.lon

    mean = prepare_state(
        *mean_fields,
        lat_nh,
        lon,
        levels,
        ops,
        cfg.mirror.f_floor_deg,
        cfg.pv_source,
        data_sht=out_sht,
    )
    event = prepare_state(
        *event_fields,
        lat_nh,
        lon,
        levels,
        ops,
        cfg.mirror.f_floor_deg,
        cfg.pv_source,
        data_sht=out_sht,
    )
    # Real data are hydrostatically consistent to a few percent; a potential
    # temperature handed in as a temperature, or a height as a geopotential,
    # is not, and would otherwise survive every other check.
    for label, state_ in (("climatology", mean), ("event", event)):
        ratio = state_.boundary_theta_ratio[1]
        if not 0.7 < ratio < 1.4:
            raise ValueError(
                f"the {label}'s geopotential and temperature are not "
                f"hydrostatically consistent: the top boundary temperature the "
                f"geopotential implies is {ratio:.2f} times the one from the "
                f"temperature (a potential temperature passed as temperature, or "
                f"a height as geopotential, does this)"
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
    u_obs_solver, v_obs_solver = (
        rotational_wind_stack(ops, event.psi_spec - mean.psi_spec)
        if rotated_track
        else (None, None)
    )
    u_obs, v_obs = rotational_wind_on_regular_grid(
        out_sht, event.psi_spec - mean.psi_spec
    )
    height = out_sht.synthesize(event.phi_spec) / 9.81

    def cartesian_on_output(u_solver: np.ndarray, v_solver: np.ndarray):
        """Cartesian wind components on the output grid, finite at the poles.

        For a band-limited streamfunction these are band-limited scalars, so the
        transfer is a spectral one and exact -- and unlike the eastward and
        northward components they are defined on the pole rows, which the rotated
        cropping needs.
        """
        parts = to_cartesian(u_solver, v_solver, ops.grid.lat, ops.grid.lon)
        return tuple(ops.sht.regrid_to(out_sht, part) for part in parts)

    observed_cartesian = (
        cartesian_on_output(u_obs_solver, v_obs_solver) if rotated_track else None
    )

    piece_u: dict[str, np.ndarray] = {}
    piece_v: dict[str, np.ndarray] = {}
    piece_cartesian: dict[str, tuple | None] = {}
    summed_u = np.zeros_like(u_obs)
    summed_v = np.zeros_like(v_obs)
    summed_cartesian = None
    for name, piece in result.pieces.items():
        u_piece, v_piece = rotational_wind_on_regular_grid(out_sht, piece.psi_spec)
        summed_u += u_piece
        summed_v += v_piece
        # The solver-grid pair is needed only by the rotated track, which goes
        # through the Cartesian components; forming it otherwise is two
        # transforms per level for nothing.
        cart = (
            cartesian_on_output(*rotational_wind_stack(ops, piece.psi_spec))
            if rotated_track
            else None
        )
        if rotated_track:
            summed_cartesian = (
                cart
                if summed_cartesian is None
                else tuple(a + b for a, b in zip(summed_cartesian, cart))
            )
        piece_u[name] = u_piece
        piece_v[name] = v_piece
        piece_cartesian[name] = cart

    # What the balance equations could not represent: the divergent and
    # unbalanced part, and nothing else.
    residual_cartesian = (
        tuple(a - b for a, b in zip(observed_cartesian, summed_cartesian))
        if rotated_track
        else None
    )

    # Converted to SI before it is delivered, never left as the solver's raw
    # right-hand side.  The two differ by `pv_rhs_scale`, which is 31.6 at 850 hPa
    # falling to 11.3 at 200: smooth in the vertical, so an unconverted field
    # would show up as a plausible physical profile rather than as the unit
    # error it is.  The boundary levels carry no interior potential vorticity
    # and stay NaN; the rows the mirror invented are erased.
    scaffold = out_grid.lat < float(np.min(lat_nh)) - 1e-9
    pv_anom = np.full((levels.nlev, out_grid.nlat, out_grid.nlon), np.nan)
    anomaly_si = ertel_pv_si(event.q_hat - mean.q_hat, levels)
    pv_anom[levels.interior] = out_sht.synthesize(ops.analyze(anomaly_si))
    pv_anom[..., scaffold, :] = np.nan

    lat0, lon0 = float(centre[0]), float(centre[1])
    meta = {
        "pieces_mode": pieces_mode,
        "pv_source": cfg.pv_source,
        "deformation_limiter": cfg.newton.deformation_limiter,
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
        "solver_grid": (engine.solver_nlat, engine.solver_nlon),
        "lmax": ops.sht.lmax,
        "newton_steps": result.newton_steps,
        "newton_converged": result.diagnostics["newton_converged"],
        "newton_final_increment_m": result.diagnostics["newton_final_increment_m"],
        "linear_iterations": result.diagnostics["linear_iterations"],
        "inner_solves_converged": result.diagnostics["inner_solves_converged"],
        "inner_solves_unconverged": result.diagnostics["inner_solves_unconverged"],
        # Residuals of the equations as posed at the balanced state, in metres
        # of height and in PVU, globally and poleward of the taper band: what
        # "converged" means physically, which the increment test alone does not
        # say.  And where the deformation limiter acted, per level.
        **{
            f"newton_final_{k}": float(v)
            for k, v in result.diagnostics["newton_final_norms"].items()
        },
        "newton_deformation_fraction": result.diagnostics[
            "newton_deformation_fraction"
        ],
        "newton_limiter_refreshes": result.diagnostics["newton_limiter_refreshes"],
        "newton_final_nonelliptic_fraction": result.diagnostics[
            "newton_final_nonelliptic_fraction"
        ],
        "piece_deformation_fraction": result.diagnostics["piece_deformation_fraction"],
        "clamp_worst_fraction": result.clamp_worst,
        "pv_floor_fraction_event": result.floor_event.fraction,
        "pv_floor_fraction_mean": result.floor_mean.fraction,
        "all_pieces_converged": all(
            p.report.converged for p in result.pieces.values()
        ),
    }
    return HemisphereInversion(
        piece_u=piece_u,
        piece_v=piece_v,
        u_obs=u_obs,
        v_obs=v_obs,
        u_residual=u_obs - summed_u,
        v_residual=v_obs - summed_v,
        height=height,
        pv_anom=pv_anom,
        lat=out_grid.lat,
        lon=out_grid.lon,
        lat_nh=lat_nh,
        meta=meta,
        split=split,
        result=result if keep_result else None,
        piece_cartesian=piece_cartesian if rotated_track else None,
        observed_cartesian=observed_cartesian,
        residual_cartesian=residual_cartesian,
    )


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
    engine: SphereEngine | None = None,
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
        engine: Transforms built once by :meth:`SphereEngine.build` for these
            axes, when many events share a grid.  Its configuration and solver
            grid are the ones used; ``cfg``, ``solver_nlat``, ``solver_nlon`` and
            ``lmax`` then have to agree with it or be left at their defaults.

    Returns:
        An :class:`EventOutput` whose ``arrays`` are ready to save.
    """
    if engine is None:
        engine = SphereEngine.build(
            lat_nh, lon, cfg=cfg, solver_nlat=solver_nlat, solver_nlon=solver_nlon, lmax=lmax
        )
    else:
        if not engine.fits(lat_nh, lon):
            raise ValueError("the engine was built for other latitude or longitude axes")
        if cfg is not None and cfg is not engine.cfg:
            raise ValueError("cfg was given along with an engine built from another")
        if (solver_nlat, solver_nlon) != (engine.solver_nlat, engine.solver_nlon) and (
            solver_nlat,
            solver_nlon,
        ) != (128, 256):
            raise ValueError("solver_nlat/solver_nlon disagree with the engine's grid")
        if lmax is not None and lmax != engine.ops.sht.lmax:
            raise ValueError("lmax disagrees with the engine's truncation")
    levels = engine.levels
    out_grid = engine.out_grid
    lat_nh = engine.lat_nh

    hi = invert_hemisphere(
        engine,
        mean_fields,
        event_fields,
        centre,
        pieces_mode=pieces_mode,
        rotated_track=rotated_track,
        keep_result=keep_result,
    )

    lat0, lon0 = float(centre[0]), float(centre[1])
    scaffold = out_grid.lat < float(np.min(lat_nh)) - 1e-9
    arrays: dict[str, np.ndarray] = {}

    def blank_scaffold(field_: np.ndarray) -> np.ndarray:
        """Erase the rows the mirror invented.

        The output grid spans both hemispheres, and south of the data's own first
        row every value is a reflection of the north: smooth, correctly scaled and
        entirely fabricated.  A patch centred below about 42 degrees reaches into
        it, and those rows would otherwise enter a composite as if they were
        observations.
        """
        out = np.array(field_, dtype=float, copy=True)
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
            column = _weighted_column_mean(values, hi.height, levels)
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
            # The Cartesian components are handed over unblanked -- they have to
            # be, since a NaN spreads through the sampler's prefilter and empties
            # the patch -- so the mirror's fabricated southern rows are erased
            # here instead, by where each sampled point actually landed.  A patch
            # centred in the north never reaches them; one centred low does, and
            # would otherwise deliver a reflection as an observation.
            invented = u_r.lat < float(np.min(lat_nh)) - 1e-9
            u_r.values[..., invented] = np.nan
            v_r.values[..., invented] = np.nan
            arrays[_key("u", name, "_3d_rot")] = u_r.values.astype(np.float32)
            arrays[_key("v", name, "_3d_rot")] = v_r.values.astype(np.float32)
            arrays.setdefault("lat_rel_rot", u_r.lat_rel)
            arrays.setdefault("lon_rel_rot", u_r.lon_rel)

    for name in hi.piece_names():
        store(
            name,
            hi.piece_u[name],
            hi.piece_v[name],
            hi.piece_cartesian[name] if rotated_track else None,
        )

    # Named outside the piece namespace so a loader enumerating pieces by prefix
    # does not pick it up as one; see `_key`.
    store("residual", hi.u_residual, hi.v_residual, hi.residual_cartesian)

    # The anomaly the pieces are accounting for.  Stored so that the closure --
    # every piece plus the residual against this field -- can be checked from the
    # file alone rather than trusted, and so that a file states which anomaly it
    # was aiming at instead of leaving that to be reconstructed.
    store("observed", hi.u_obs, hi.v_obs, hi.observed_cartesian)

    pv_patch = geographic_patch(
        hi.pv_anom, out_grid.lat, out_grid.lon, lat0, lon0, lat_half, lon_half
    )
    arrays["pv_anom_wu_3d"] = pv_patch.values.astype(np.float32)

    return EventOutput(arrays=arrays, meta=hi.meta, result=hi.result)
