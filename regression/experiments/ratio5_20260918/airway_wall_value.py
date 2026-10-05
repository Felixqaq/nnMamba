"""Is there small/large-airway information the ratio model is missing? Analysis only.

Inspiratory CT carries no voxel-threshold marker of small-airway disease (the
-856 HU band is an expiratory measure and was shown empty on 2026-10-01). The
remaining inspiratory route is the wall of the airways that are visible. This
measures it on 200 training patients and asks two questions:

  1. Does airway-wall burden relate to FEV1/FVC beyond emphysema and lung size?
  2. Does it explain what the current model gets WRONG? The model's errors come
     from out-of-fold predictions of the 1149-cohort mask-free CV run: each
     patient was predicted by a fold model that never trained on them, so the
     residual is honest and the validation set is not touched.

If (2) is near zero the model already captures it, or it is not there; either
way an airway training target would not help.

Airways come from TotalSegmentator lung_vessels (lung_airways, lung_airways_wall),
~36 s per patient on the GPU. On 3 mm slices only large and medium airways are
resolved, so these are volumetric proxies for wall thickness, not Pi10.
"""

from __future__ import annotations

import csv
import glob
import json
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

REG = Path('/home/felix/Research/nnMamba/regression')
BASE = REG / 'outputs/ratio5_expanded_20260922'                 # 1149 cohort snapshot
OOF = REG / 'outputs/ratio5_expanded_20260922_cv_maskfree'
AIRWAYS = REG / 'masks/totalseg_lung_vessels'
LUNG_DIRS = [REG / 'masks/totalseg/lung'] + [REG / f'outputs/{d}/masks/lung' for d in (
    'ratio5_expanded_20260918', 'ratio5_expanded_20260922', 'ratio5_expanded_20260929_masked',
    'ratio5_expanded_20260930_maskfree_guided', 'ratio5_expanded_20260915')]
OUT = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260930_maskfree_guided/'
           'airway_wall_value.txt')
N = 200


def log(msg):
    print(msg, flush=True)


def main():
    oof = {}
    for f in sorted(glob.glob(str(OOF / 'oof_fold*.json'))):
        oof.update(json.load(open(f))['patients'])
    paths = {r['patient_id']: r['path'] for r in json.loads((BASE / 'ct/build_summary.json').read_text(
        encoding='utf-8-sig'))['records'] if r['ok']}
    pft = {}
    for r in csv.DictReader(open(BASE / 'pft_original.csv', encoding='utf-8-sig')):
        r = {(k or '').strip(): (v or '').strip() for k, v in r.items()}
        if r.get('PatientID'):
            pft[r['PatientID']] = r

    rng = np.random.default_rng(20261001)
    pool = sorted(p for p in oof if p in paths)
    sample = sorted(rng.choice(pool, size=N, replace=False))

    AIRWAYS.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, pid in enumerate(sample):
        out = AIRWAYS / pid
        if not (out / 'lung_airways_wall.nii.gz').exists():
            tmp = AIRWAYS / (pid + '.partial')
            shutil.rmtree(tmp, ignore_errors=True)
            code = subprocess.run(['TotalSegmentator', '-i', paths[pid], '-o', str(tmp), '-ta',
                                   'lung_vessels'], stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL).returncode
            if code != 0 or not (tmp / 'lung_airways_wall.nii.gz').exists():
                log(f'{pid}: airway segmentation failed, skipped')
                continue
            shutil.rmtree(out, ignore_errors=True)
            tmp.rename(out)
        lung_path = next((d / f'{pid}.nii.gz' for d in LUNG_DIRS if (d / f'{pid}.nii.gz').exists()), None)
        if lung_path is None:
            continue
        img = nib.load(paths[pid])
        hu = np.asanyarray(img.dataobj).astype(np.float32)
        hu = hu[..., 0] if hu.ndim > 3 else hu
        lung = np.asanyarray(nib.load(str(lung_path)).dataobj) > 0
        wall = np.asanyarray(nib.load(str(out / 'lung_airways_wall.nii.gz')).dataobj) > 0
        lumen = np.asanyarray(nib.load(str(out / 'lung_airways.nii.gz')).dataobj) > 0
        if not (lung.shape == wall.shape == lumen.shape == hu.shape):
            log(f'{pid}: grid mismatch, skipped')
            continue
        voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
        lung_ml = lung.sum() * voxel_ml
        wall_ml, lumen_ml = wall.sum() * voxel_ml, lumen.sum() * voxel_ml
        rec = {'pid': pid, 'ratio': oof[pid]['true_ratio'], 'pred': oof[pid]['mean_predicted_ratio'],
               'laa950': 100 * float((hu[lung] < -950).mean()), 'lung_l': lung_ml / 1000,
               'wall_ml': wall_ml, 'lumen_ml': lumen_ml,
               'wall_frac': wall_ml / max(wall_ml + lumen_ml, 1e-6),      # WA%-like, size-free
               'wall_per_l': wall_ml / max(lung_ml / 1000, 1e-6),
               'lumen_per_l': lumen_ml / max(lung_ml / 1000, 1e-6),
               'wall_hu': float(hu[wall].mean()) if wall.any() else np.nan}
        try:
            rec['height'] = float(pft[pid]['Height_cm'])
            rec['male'] = 1.0 if pft[pid]['Sex'].upper().startswith('M') else 0.0
        except (KeyError, ValueError):
            rec['height'], rec['male'] = np.nan, np.nan
        rows.append(rec)
        if (i + 1) % 20 == 0:
            log(f'{i + 1}/{N} processed, {len(rows)} usable')

    rows = [r for r in rows if np.isfinite(r['wall_hu'])]
    y = np.array([r['ratio'] for r in rows])
    resid = np.array([r['ratio'] - r['pred'] for r in rows])
    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    kf = KFold(5, shuffle=True, random_state=0)

    def cv_r2(target, keys):
        A = np.column_stack([col(k) for k in keys])
        p = cross_val_predict(LinearRegression(), A, target, cv=kf)
        return 1 - ((target - p) ** 2).sum() / ((target - target.mean()) ** 2).sum()

    def partial(k, controls):
        C = np.column_stack([col(c) for c in controls])
        ry = y - LinearRegression().fit(C, y).predict(C)
        rx = col(k) - LinearRegression().fit(C, col(k)).predict(C)
        return float(np.corrcoef(rx, ry)[0, 1])

    airway = ('wall_frac', 'wall_per_l', 'lumen_per_l', 'wall_hu')
    lines = ['AIRWAY WALL VALUE (n=%d training patients, out-of-fold model predictions)' % len(rows),
             'proxies from TotalSegmentator on 3 mm inspiratory CT; not Pi10', '',
             '1. relation with FEV1/FVC',
             '  %-12s %8s %22s %20s' % ('metric', 'r', 'partial | LAA, lung L', 'r with model error')]
    for k in ('laa950',) + airway:
        pr = partial(k, ['laa950', 'lung_l']) if k != 'laa950' else float('nan')
        lines.append('  %-12s %+8.3f %22s %+20.3f' % (
            k, np.corrcoef(col(k), y)[0, 1], ('%+.3f' % pr) if k != 'laa950' else '-',
            np.corrcoef(col(k), resid)[0, 1]))
    lines += ['', '2. 5-fold CV R^2 for FEV1/FVC (linear)']
    for keys in (['laa950', 'lung_l'], ['laa950', 'lung_l', *airway]):
        lines.append('  %-50s %.3f' % ('+'.join(keys), cv_r2(y, keys)))
    lines += ['', '3. can airway metrics predict what the model gets wrong? (5-fold CV R^2 on residual)']
    lines.append('  model residual: mean %+.2f, sd %.2f, MAE %.2f' % (resid.mean(), resid.std(), np.abs(resid).mean()))
    for keys in (list(airway), ['laa950', 'lung_l', *airway]):
        lines.append('  %-50s %.3f' % ('+'.join(keys), cv_r2(resid, keys)))
    lines += ['', 'R^2 near 0 or negative in section 3 = the airway proxies hold nothing the model',
              'is missing, so an airway training target would not help.']
    text = '\n'.join(lines)
    print(text, flush=True)
    OUT.write_text(text, encoding='utf-8')
    (OUT.with_suffix('.json')).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
