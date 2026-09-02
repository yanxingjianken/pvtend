#!/usr/bin/env python
"""Add the PPVI blowup scalars to NPZs written before pvtend embedded them.

The omega and divergent-wind blowup families have always been recorded as
scalars at write time; the PPVI piece and residual magnitudes were added
later. This backfills them from the cubes already in each NPZ, so one store
can be scanned uniformly by ``aggregate_qg_blowup.py --fields ppvi|all``
without falling back to full-cube reads.

Every value written here is a reduction of data already in the file:

    max_abs_{u,v}_rot_anom_ppvi_{piece} = nanmax|piece cube|
    max_abs_{u,v}_rot_anom_residual_ppvi = nanmax|residual cube|

Nothing else is touched. Files that already carry the scalars are skipped, so
the script is idempotent and safe to re-run over a store that is still being
written. Rewrites are atomic (temp file in the destination directory, then
os.replace), matching the pipeline's own write convention.

Usage:
    python backfill_ppvi_blowup_scalars.py --npz-dir outputs/cesm6hourly_blocking
    python backfill_ppvi_blowup_scalars.py --npz-dir ... --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PARTS = ("surface", "lower", "upper_p", "upper_e")
SENTINEL = "max_abs_u_rot_anom_ppvi_surface"


def _targets() -> dict[str, str]:
    """scalar name -> source cube name."""
    out = {f"max_abs_{c}_rot_anom_ppvi_{p}": f"{c}_rot_anom_ppvi_{p}_3d"
           for c in "uv" for p in PARTS}
    out.update({f"max_abs_{c}_rot_anom_residual_ppvi":
                f"{c}_rot_anom_residual_ppvi_3d" for c in "uv"})
    return out


def process(args: tuple[str, bool]) -> str:
    """Return a status: added | skipped_present | skipped_no_ppvi | error."""
    fp_s, dry = args
    fp = Path(fp_s)
    tgt = _targets()
    try:
        with np.load(fp, allow_pickle=False) as z:
            names = set(z.files)
            if SENTINEL in names:
                return "skipped_present"
            if any(cube not in names for cube in tgt.values()):
                # inline PPVI failed for this event: base NPZ, no piece keys
                return "skipped_no_ppvi"
            store = {k: z[k] for k in z.files}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for scalar, cube in tgt.items():
                    store[scalar] = np.float32(
                        np.nanmax(np.abs(store[cube])))
        if dry:
            return "added"
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".npz.tmp", dir=str(fp.parent),
                    delete=False) as fh:
                tmp = fh.name
                np.savez_compressed(fh, **store)
            os.replace(tmp, str(fp))
            tmp = None
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)
        return "added"
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as e:
        print(f"  [warn] {fp}: {type(e).__name__}: {e}", file=sys.stderr)
        return "error"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", required=True, type=Path,
                    help="Store root; walks {stage}/dh=*/track_*.npz.")
    ap.add_argument("--stages", nargs="+",
                    default=["onset", "peak", "decay"])
    ap.add_argument("--workers", type=int,
                    default=min(32, (os.cpu_count() or 8)))
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without rewriting.")
    args = ap.parse_args()

    files: list[str] = []
    for stage in args.stages:
        for dh in sorted((args.npz_dir / stage).glob("dh=*")):
            files += [str(p) for p in sorted(dh.glob("track_*.npz"))]
    if not files:
        print(f"no NPZ under {args.npz_dir}/{{{','.join(args.stages)}}}")
        return

    print(f"[backfill] {len(files)} NPZ, {args.workers} workers"
          f"{' (dry run)' if args.dry_run else ''}", flush=True)
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, st in enumerate(ex.map(process,
                                      ((f, args.dry_run) for f in files),
                                      chunksize=16), 1):
            counts[st] = counts.get(st, 0) + 1
            if i % max(1, len(files) // 20) == 0:
                print(f"  [{i:>6d}/{len(files)}] {counts}", flush=True)
    print(f"[backfill] done: {counts}")


if __name__ == "__main__":
    main()
