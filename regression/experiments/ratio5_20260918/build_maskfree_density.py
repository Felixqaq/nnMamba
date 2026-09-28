"""Segmentation-free emphysema channel: threshold the whole volume, then downsample.

The %LAA-950 channel earns its +0.027 (fixed-70) / +0.069 (GLI) AUC because the
threshold is applied at native resolution *before* the volume is shrunk to the
model grid -- resizing first blurs away 81% of the %LAA-950 signal. The lung
mask only restricts that threshold to lung voxels, and the network already sees
the CT, so it can tell lung from outside air on its own.

Dropping the mask removes the one step that needs a segmentation network at
inference. What remains -- compare every voxel with -950 HU and average over
each block -- is a few lines of numpy that run in about a second on a CPU,
which matters because the hospital terminals have no GPU and no internet.

Stored as (2, D, H, W) uint8 so the existing loader reads it unchanged:
  channel 0  fraction of each block below -950 HU, whole volume, no mask
  channel 1  fraction below -910 HU, kept for a possible later arm; the
             2-channel run reads channel 0 only
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np

REG = Path('/home/felix/Research/nnMamba/regression')
BASE = REG / 'outputs/ratio5_expanded_20260922'
OUT = REG / 'outputs/ratio5_expanded_20260922_maskfree/density_maskfree'
IMAGE_SIZE = (112, 136, 112)
THRESHOLDS = (-950.0, -910.0)


def one(job):
    pid, path = job
    target = OUT / f'{pid}.npy'
    if target.exists():
        return pid, None
    try:
        import nibabel as nib
        from skimage import transform
        ct = np.asanyarray(nib.load(path).dataobj).astype(np.float32)
        if ct.ndim > 3:
            ct = ct[..., 0]
        chans = []
        for hu in THRESHOLDS:
            # Same geometry as the masked channel: whole volume resized to the
            # model grid, indicator first, so a voxel holds the fraction of its
            # block below the threshold.
            frac = transform.resize((ct < hu).astype(np.float32), IMAGE_SIZE, order=1,
                                    preserve_range=True, anti_aliasing=True)
            chans.append(np.clip(np.rint(frac * 255.0), 0, 255).astype(np.uint8))
        tmp = target.with_suffix('.tmp.npy')
        np.save(tmp, np.stack(chans, axis=0))
        tmp.rename(target)
        return pid, None
    except Exception as exc:  # reported, never swallowed
        return pid, repr(exc)


def sanity(pids) -> None:
    """Inside the lung the two channels must agree; outside they need not."""
    diffs, outside = [], []
    for pid in pids:
        masked = np.load(BASE / 'density' / f'{pid}.npy').astype(np.float32) / 255.0
        free = np.load(OUT / f'{pid}.npy')[0].astype(np.float32) / 255.0
        lung = masked[1] > 0.95
        diffs.append(float(np.abs(free[lung] - masked[0][lung]).mean()))
        outside.append(float((free[masked[1] < 0.05] > 0.5).mean()))
    print('sanity on %d patients:' % len(pids))
    print('  inside lung (occupancy > 0.95): mean |maskfree - masked| = %.4f  (max %.4f)'
          % (np.mean(diffs), np.max(diffs)))
    print('  outside lung: %.1f%% of blocks are mostly air (outside body, trachea)'
          % (100 * np.mean(outside)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads((BASE / 'ct/build_summary.json').read_text(
        encoding='utf-8-sig'))['records'] if r['ok']]
    jobs = [(r['patient_id'], r['path']) for r in rows]
    failed = []
    with Pool(4) as pool:
        for i, (pid, err) in enumerate(pool.imap_unordered(one, jobs), 1):
            if err:
                failed.append((pid, err))
            if i % 100 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} built, failed {len(failed)}', flush=True)
    if failed:
        raise SystemExit(f'{len(failed)} failed: {failed[:5]}')
    sanity([r['patient_id'] for r in rows[:40]])
    print('MASKFREE_BUILD_EXIT=0')


if __name__ == '__main__':
    sys.exit(main())
