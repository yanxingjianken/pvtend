"""Lazy loader / builder for the f2py ``_wuppvi`` extension.

The Wu inversion cores live in ``fortran/wuppvi.f`` and are compiled with
f2py (meson backend, ``-frecursive`` for process-fork safety).  Because the
extension is **not** shipped in the pure-Python PyPI wheel, it is built on
demand on the local HPC host and cached next to the source.

Use :func:`load_ext` to obtain the imported module; it raises a helpful
:class:`RuntimeError` with the exact build command if the build fails.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_FORT_DIR = Path(__file__).resolve().parent / "fortran"
_SRC = _FORT_DIR / "wuppvi.f"
_MODNAME = "_wuppvi"
# f2py flags that produced the validated, byte-exact build.
_F77FLAGS = "-std=legacy -O2 -frecursive"

_cached_mod = None


def _find_so(search_dirs: list[Path]) -> Path | None:
    for d in search_dirs:
        if not d.is_dir():
            continue
        for so in d.glob(f"{_MODNAME}*.so"):
            return so
    return None


def build_ext(force: bool = False) -> Path:
    """Compile ``wuppvi.f`` into the ``_wuppvi`` extension via f2py.

    Args:
        force: Rebuild even if an up-to-date ``.so`` already exists.

    Returns:
        Path to the compiled shared object.

    Raises:
        RuntimeError: If f2py is unavailable or the build fails.
    """
    build_dir = _FORT_DIR / "_build"
    build_dir.mkdir(exist_ok=True)
    existing = _find_so([build_dir, _FORT_DIR / "_build_test"])
    if existing is not None and not force:
        if existing.stat().st_mtime >= _SRC.stat().st_mtime:
            return existing

    cmd = [
        sys.executable, "-m", "numpy.f2py",
        "-c", "-m", _MODNAME, str(_SRC),
        "--backend", "meson", f"--f77flags={_F77FLAGS}",
    ]
    proc = subprocess.run(cmd, cwd=str(build_dir),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to build the _wuppvi f2py extension.\n"
            f"Command: {' '.join(cmd)}\n(cwd={build_dir})\n"
            f"--- stdout (tail) ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-2000:]}"
        )
    so = _find_so([build_dir])
    if so is None:
        raise RuntimeError(
            f"_wuppvi build reported success but no .so found in {build_dir}."
        )
    return so


def load_ext(build_if_missing: bool = True):
    """Import and return the compiled ``_wuppvi`` module (cached).

    Args:
        build_if_missing: If True, build the extension when no ``.so`` is
            found on disk.

    Returns:
        The imported ``_wuppvi`` module.

    Raises:
        RuntimeError: If the extension cannot be found or built.
    """
    global _cached_mod
    if _cached_mod is not None:
        return _cached_mod

    so = _find_so([_FORT_DIR / "_build", _FORT_DIR / "_build_test"])
    if so is None:
        if not build_if_missing:
            raise RuntimeError(
                "_wuppvi extension not built. Call pvtend.ppvi._ext.build_ext()."
            )
        so = build_ext()

    spec = importlib.util.spec_from_file_location(_MODNAME, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _cached_mod = mod
    return mod
