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
        assert args.qg_method is None  # auto-selected in _cmd_compute

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
            "--qg-method", "fft",
            "--center-mode", "lagrangian",
            "--n-workers", "4",
            "--skip-existing",
        ])
        assert args.dh_range == "-25:25:1"
        assert args.qg_method == "fft"
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
