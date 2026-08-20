"""Tests for the CLI entry point."""

from __future__ import annotations

import pytest

from pvtend.cli import _build_parser, _parse_dh_range, main


# ── dh range parser ──────────────────────────────────────────────────

class TestParseDhRange:
    """Test _parse_dh_range helper."""

    def test_two_part(self):
        result = _parse_dh_range("-5:5")
        assert result == list(range(-5, 5))

    def test_three_part(self):
        result = _parse_dh_range("0:10:2")
        assert result == [0, 2, 4, 6, 8]

    def test_negative_range(self):
        result = _parse_dh_range("-49:25:1")
        assert len(result) == 74
        assert result[0] == -49
        assert result[-1] == 24

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="start:stop"):
            _parse_dh_range("5")

    def test_four_parts_raises(self):
        with pytest.raises(ValueError, match="start:stop"):
            _parse_dh_range("1:2:3:4")


# ── Argument parser ──────────────────────────────────────────────────

class TestBuildParser:
    """Test that _build_parser creates valid parser with all subcommands."""

    def test_parser_creation(self):
        parser = _build_parser()
        assert parser is not None

    def test_compute_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args([
            "compute",
            "--event-type", "blocking",
            "--events-csv", "events.csv",
            "--era5-dir", "/data/era5",
            "--clim-path", "/data/clim.nc",
            "--out-dir", "/data/output",
        ])
        assert args.command == "compute"
        assert args.event_type == "blocking"

    def test_classify_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args([
            "classify",
            "--npz-dir", "/data/npz",
            "--output", "rwb.pkl",
        ])
        assert args.command == "classify"
        assert args.threshold == 3  # default

    def test_composite_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args([
            "composite",
            "--npz-dir", "/data/npz",
            "--pkl-out", "composite.pkl",
        ])
        assert args.command == "composite"
        assert args.rwb_pkl is None  # optional

    def test_decompose_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args([
            "decompose",
            "--pkl-in", "composite.pkl",
            "--out-dir", "decomp/",
        ])
        assert args.command == "decompose"

    def test_default_qg_method_none(self):
        parser = _build_parser()
        args = parser.parse_args([
            "compute",
            "--event-type", "prp",
            "--events-csv", "e.csv",
            "--era5-dir", "/d",
            "--clim-path", "/c",
            "--out-dir", "/o",
        ])
        assert args.qg_method == "log20"  # default solver

    def test_compute_options(self):
        parser = _build_parser()
        args = parser.parse_args([
            "compute",
            "--event-type", "blocking",
            "--events-csv", "e.csv",
            "--era5-dir", "/d",
            "--clim-path", "/c",
            "--out-dir", "/o",
            "--dh-range=-25:25:1",
            "--qg-method", "sp19",
            "--center-mode", "lagrangian",
            "--n-workers", "4",
            "--skip-existing",
        ])
        assert args.dh_range == "-25:25:1"
        assert args.qg_method == "sp19"
        assert args.center_mode == "lagrangian"
        assert args.n_workers == 4
        assert args.skip_existing is True

    def test_classify_custom_levels(self):
        parser = _build_parser()
        args = parser.parse_args([
            "classify",
            "--npz-dir", "/d",
            "--output", "r.pkl",
            "--levels", "500", "300", "200",
            "--threshold", "2",
        ])
        assert args.levels == [500, 300, 200]
        assert args.threshold == 2


# ── Main entry point ────────────────────────────────────────────────

class TestMain:
    """Test CLI main() exit codes."""

    def test_no_command_returns_1(self):
        ret = main([])
        assert ret == 1

    def test_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_version(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0


class TestPPVIPiecesFlag:
    """``--ppvi-pieces`` must exist on BOTH subcommands that write PPVI keys:
    `compute` starts a catalogue and `ppvi` appends to one, and the two
    decompositions write different key sets.
    """

    BASE = ["--event-type", "blocking", "--events-csv", "e.csv",
            "--era5-dir", "/d/era5", "--clim-path", "/d/clim.nc",
            "--out-dir", "/d/out"]

    @pytest.mark.parametrize("cmd", ["compute", "ppvi"])
    def test_defaults_to_per_level(self, cmd):
        args = _build_parser().parse_args([cmd, *self.BASE])
        assert args.ppvi_pieces == "per_level"

    @pytest.mark.parametrize("cmd", ["compute", "ppvi"])
    def test_scale_is_selectable(self, cmd):
        args = _build_parser().parse_args(
            [cmd, *self.BASE, "--ppvi-pieces", "scale"])
        assert args.ppvi_pieces == "scale"

    @pytest.mark.parametrize("cmd", ["compute", "ppvi"])
    def test_unknown_mode_rejected(self, cmd):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(
                [cmd, *self.BASE, "--ppvi-pieces", "lower_middle_upper"])


class TestDispatchReachable:
    """Every dispatch key must name a real subcommand, and vice versa.

    ``clim-helmholtz`` was deleted from the parser in f421e06 as collateral of
    removing the adjacent one-time ``qsplit`` retrofit, while its handler and
    its dispatch entry both survived — so the subcommand looked present in the
    source and in the README, and only failed at argparse's invalid-choice at
    the moment someone tried to rebuild a Helmholtz climatology. Nothing in the
    suite noticed, because every test named its subcommand explicitly. This
    couples the two lists so the next accidental deletion fails here.
    """

    def _choices(self):
        import argparse
        sub = next(a for a in _build_parser()._subparsers._group_actions
                   if isinstance(a, argparse._SubParsersAction))
        return set(sub.choices)

    def _dispatch_keys(self):
        import inspect
        import re
        from pvtend.cli import main as _main
        src = inspect.getsource(_main)
        body = src.split("dispatch = {", 1)[1].split("}", 1)[0]
        return set(re.findall(r'"([^"]+)":', body))

    def test_every_dispatch_key_is_a_subcommand(self):
        missing = self._dispatch_keys() - self._choices()
        assert not missing, (
            f"handler(s) unreachable from the CLI: {sorted(missing)} — "
            f"the dispatch entry exists but argparse rejects the name")

    def test_every_subcommand_has_a_handler(self):
        orphan = self._choices() - self._dispatch_keys()
        assert not orphan, (
            f"subcommand(s) with no handler: {sorted(orphan)}")

    def test_clim_helmholtz_parses(self):
        args = _build_parser().parse_args(
            ["clim-helmholtz", "--clim-dir", "/d/clim", "--output-dir", "/d/out"])
        assert args.command == "clim-helmholtz"
        assert args.clim_stem == "era5_hourly_clim_1990-2020"
