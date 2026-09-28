"""Airway channel: CT + LAA density + airway-wall fraction, paired against 2 channels.

COPD has two imaging faces. Emphysema is already an input -- the %LAA-950
density channel, worth +0.027 AUC fixed-70 and +0.069 under GLI. Airway disease
is not: nothing tells the network where the bronchial walls are or how thick.
The occupancy channel tried on 2026-09-19 added nothing because it repeats the
lung mask; the airway wall is a different signal.

TotalSegmentator's lung_vessels task segments lung_airways and lung_airways_wall
directly (measured 2026-09-22: ~36 s per patient, 4.3 GB peak VRAM, a connected
tree holding ~95% of the airway voxels even on 3 mm slices). On 3 mm slices the
smallest airways are invisible, so this is the coarse, large-airway version of
the Pi10 idea, not the small-airway one.

No loader change is needed. The loader already accepts a (2, D, H, W) uint8
array per patient in density_and_occupancy mode; here channel 1 carries the
airway-wall fraction instead of the lung occupancy, in a separate directory.
The config written below says so, because the mode name is now a misnomer.

Stages, each resumable:
  1. segment airways for every cohort patient
  2. build the (density, airway wall) arrays and a per-patient airway table
  3. a cheap univariate check of the airway signal before six hours of training
  4. train five seeds, score, and compare against the 2-channel run seed by seed
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import time

import numpy as np

REG = Path('/home/felix/Research/nnMamba/regression')
BASE = REG / 'outputs/ratio5_expanded_20260922'
ROOT = REG / 'outputs/ratio5_expanded_20260922_airway'
PUBLIC = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260922_airway')
MASKS = REG / 'masks/totalseg_lung_vessels'
CHANNEL = ROOT / 'density_airway'
SEEDS = [72, 73, 74, 75, 76]
EPOCHS = 80
IMAGE_SIZE = (112, 136, 112)
CUTOFF, BORDER = 70.0, 7.0


def save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')


def log(message: str) -> None:
    print('[%s] %s' % (time.strftime('%H:%M:%S'), message), flush=True)


def execute(name: str, command: list[str]) -> None:
    log(name)
    with (PUBLIC / f'{name}.log').open('a') as fh:
        subprocess.run(command, cwd=REG, stdout=fh, stderr=subprocess.STDOUT, check=True)


# ---------------------------------------------------------- 1. segmentation --
def segment(rows) -> None:
    MASKS.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows
            if not (MASKS / r['patient_id'] / 'lung_airways_wall.nii.gz').exists()]
    log(f'segment airways: {len(rows) - len(todo)} done, {len(todo)} to go')
    failed = []
    for i, row in enumerate(todo):
        out = MASKS / row['patient_id']
        tmp = MASKS / (row['patient_id'] + '.partial')
        shutil.rmtree(tmp, ignore_errors=True)
        with (PUBLIC / 'segment_airways.log').open('a') as fh:
            code = subprocess.run(['TotalSegmentator', '-i', row['path'], '-o', str(tmp),
                                   '-ta', 'lung_vessels'],
                                  stdout=fh, stderr=subprocess.STDOUT).returncode
        # Written to a partial directory and renamed only on success, so an
        # interrupted run never leaves a half-written mask that looks finished.
        if code == 0 and (tmp / 'lung_airways_wall.nii.gz').exists():
            shutil.rmtree(out, ignore_errors=True)
            tmp.rename(out)
        else:
            failed.append(row['patient_id'])
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            log(f'segment airways: {i + 1}/{len(todo)}, failed so far {len(failed)}')
    if failed:
        raise SystemExit(f'airway segmentation failed for {len(failed)}: {failed[:20]}')


# ------------------------------------------------------------- 2. channel ----
def build_channel(rows) -> list[dict]:
    import nibabel as nib
    from skimage import transform

    CHANNEL.mkdir(parents=True, exist_ok=True)
    table_path = ROOT / 'airway_table.csv'
    done = {}
    if table_path.exists():
        with table_path.open(encoding='utf-8') as fh:
            done = {r['patient_id']: r for r in csv.DictReader(fh)}

    table = []
    for i, row in enumerate(rows):
        pid = row['patient_id']
        target = CHANNEL / f'{pid}.npy'
        if target.exists() and pid in done:
            table.append(done[pid])
            continue
        density = np.load(BASE / 'density' / f'{pid}.npy')
        if density.shape != (2, *IMAGE_SIZE) or density.dtype != np.uint8:
            raise SystemExit(f'{pid}: unexpected density array {density.shape} {density.dtype}')
        ct_img = nib.load(row['path'])
        wall_img = nib.load(str(MASKS / pid / 'lung_airways_wall.nii.gz'))
        lumen_img = nib.load(str(MASKS / pid / 'lung_airways.nii.gz'))
        shape = ct_img.shape[:3]
        if wall_img.shape[:3] != shape or lumen_img.shape[:3] != shape:
            raise SystemExit(f'{pid}: airway mask grid {wall_img.shape} differs from CT {shape}')
        wall = np.asanyarray(wall_img.dataobj) > 0
        lumen = np.asanyarray(lumen_img.dataobj) > 0

        # Same geometry as the density channel: the whole volume resized to the
        # model grid, thresholded mask first, so each voxel holds the fraction of
        # its block that is airway wall.
        frac = transform.resize(wall.astype(np.float32), IMAGE_SIZE, order=1,
                                preserve_range=True, anti_aliasing=True)
        airway = np.clip(np.rint(frac * 255.0), 0, 255).astype(np.uint8)
        np.save(target, np.stack([density[0], airway], axis=0))

        voxel_ml = float(np.prod(ct_img.header.get_zooms()[:3])) / 1000.0
        wall_ml, lumen_ml = float(wall.sum()) * voxel_ml, float(lumen.sum()) * voxel_ml
        table.append({'patient_id': pid, 'wall_ml': round(wall_ml, 2),
                      'lumen_ml': round(lumen_ml, 2),
                      'wall_to_lumen': round(wall_ml / lumen_ml, 4) if lumen_ml else ''})
        if (i + 1) % 100 == 0:
            log(f'airway channel: {i + 1}/{len(rows)}')

    with table_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['patient_id', 'wall_ml', 'lumen_ml',
                                                'wall_to_lumen'])
        writer.writeheader()
        writer.writerows(table)
    shutil.copy2(table_path, PUBLIC / 'airway_table.csv')
    return table


# ------------------------------------------------------- 3. signal check -----
def signal_check(table, train_ids) -> dict:
    """Univariate association of airway measures with FEV1/FVC, training patients only."""
    from scipy.stats import spearmanr

    ratio = {}
    with (BASE / 'clinical.csv').open(encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            ratio[r['PatientID']] = float(r['FEV1FVC_pct'])
    rows = {r['patient_id']: r for r in table}
    out = {}
    lines = ['AIRWAY SIGNAL CHECK (training patients only; Spearman vs FEV1/FVC)']
    for key in ('wall_ml', 'lumen_ml', 'wall_to_lumen'):
        pairs = [(float(rows[p][key]), ratio[p]) for p in train_ids
                 if p in rows and rows[p][key] not in ('', None)]
        x, y = zip(*pairs)
        rho, p = spearmanr(x, y)
        border = [(a, b) for a, b in pairs if abs(b - CUTOFF) < BORDER]
        bx, by = zip(*border)
        brho, bp = spearmanr(bx, by)
        out[key] = {'rho': float(rho), 'p': float(p), 'n': len(pairs),
                    'border_rho': float(brho), 'border_p': float(bp), 'border_n': len(border)}
        lines.append('  %-14s rho %+.3f (p=%.2g, n=%d) | borderline rho %+.3f (p=%.2g, n=%d)'
                     % (key, rho, p, len(pairs), brho, bp, len(border)))
    # For scale: the emphysema channel's own univariate strength on the same patients.
    laa = []
    for pid in train_ids:
        d = np.load(BASE / 'density' / f'{pid}.npy')
        occ = d[1].astype(np.float64).sum()
        laa.append((100.0 * d[0].astype(np.float64).sum() / occ if occ else 0.0, ratio[pid]))
    x, y = zip(*laa)
    rho, p = spearmanr(x, y)
    out['laa_reference'] = {'rho': float(rho), 'p': float(p)}
    lines.append('  %-14s rho %+.3f (p=%.2g)   <- the density channel, for scale'
                 % ('%LAA (approx)', rho, p))
    text = '\n'.join(lines)
    print(text, flush=True)
    (PUBLIC / 'airway_signal_check.txt').write_text(text, encoding='utf-8')
    return out


# ---------------------------------------------------------------- 4. train ---
def train_and_compare() -> None:
    import yaml
    config = yaml.safe_load((BASE / 'config.yaml').read_text())
    if int(config['model']['in_channels']) != 2:
        raise SystemExit('the base run is not 2-channel; refusing to guess')
    config['model']['in_channels'] = 3
    config['data']['laa_density_mode'] = 'density_and_occupancy'
    config['data']['laa_density_dir'] = str(CHANNEL)
    (ROOT / 'config.yaml').write_text(
        '# NOTE: laa_density_mode says "occupancy", but channel 1 of every array in\n'
        '# laa_density_dir is the AIRWAY-WALL fraction, not lung occupancy. The loader\n'
        '# reads whatever is stored there; see run_airway.py.\n' + yaml.safe_dump(config))
    for name in ('train.py', 'score.py'):
        shutil.copy2(BASE / name, ROOT / name)

    common = ['--source-dir', str(BASE / 'ct'), '--manifest', str(ROOT / 'manifest.json'),
              '--pft-csv', str(BASE / 'clinical.csv')]
    for seed in SEEDS:
        out = ROOT / f'seed{seed}'
        if (out / f'regressor_seed{seed}.pth').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        execute(f'train_seed{seed}', [sys.executable, '-u', str(ROOT / 'train.py'),
                                      '--config', str(ROOT / 'config.yaml'),
                                      '--split-json', str(BASE / 'split.json'), *common,
                                      '--out', str(out), '--epochs', str(EPOCHS),
                                      '--seed', str(seed), '--skip-holdout'])
    checkpoints = [str(ROOT / f'seed{s}/regressor_seed{s}.pth') for s in SEEDS]
    output = ROOT / 'ensemble5.json'
    if not output.exists():
        execute('evaluate_5', [sys.executable, '-u', str(ROOT / 'score.py'),
                               '--config', str(ROOT / 'config.yaml'),
                               '--split-json', str(BASE / 'split_scoring.json'), *common,
                               '--checkpoints', *checkpoints, '--out', str(output)])
    shutil.copy2(output, PUBLIC / output.name)
    compare(json.loads((BASE / 'ensemble5.json').read_text()), json.loads(output.read_text()))


def compare(two: dict, air: dict) -> None:
    from scipy.stats import wilcoxon
    from sklearn.metrics import roc_auc_score

    def members(blob):
        return {v['seed']: (v['fixed70']['auc'], v['ratio_mae']) for v in blob['per_member'].values()}

    m2, ma = members(two), members(air)
    lines = ['AIRWAY vs 2-CHANNEL, paired per seed (same cohort, split, epochs)',
             '%5s | %-24s | %s' % ('seed', 'AUC  2ch -> airway', 'MAE  2ch -> airway')]
    da, dm = [], []
    for s in SEEDS:
        (a0, e0), (a1, e1) = m2[s], ma[s]
        da.append(a1 - a0)
        dm.append(e1 - e0)
        lines.append('%5d | %.4f -> %.4f  %+.4f | %6.3f -> %6.3f  %+.3f'
                     % (s, a0, a1, a1 - a0, e0, e1, e1 - e0))
    lines += ['', 'AUC delta mean %+.4f  (improved in %d/5)' % (st.mean(da), sum(d > 0 for d in da)),
              'MAE delta mean %+.3f  (improved in %d/5)' % (st.mean(dm), sum(d < 0 for d in dm)),
              'member AUC sd: 2ch %.4f  airway %.4f'
              % (st.stdev(v[0] for v in m2.values()), st.stdev(v[0] for v in ma.values())), '']

    hospital66 = {str(r['patient_id']).strip() for r in json.loads(
        (REG / 'datasets/generated/rq1_nva66_manifest.image.json').read_text())['records']}
    for name, keep in (('reviewed 180', lambda p: p not in hospital66), ('all validation', lambda p: True)):
        ids = sorted(p for p in two['patients'] if keep(p))
        t = np.array([two['patients'][p]['true_ratio'] for p in ids])
        p2 = np.array([two['patients'][p]['mean_predicted_ratio'] for p in ids])
        pa = np.array([air['patients'][p]['mean_predicted_ratio'] for p in ids])
        y = (t < CUTOFF).astype(int)
        b = np.abs(t - CUTOFF) < BORDER
        def bal(pred):
            yh = (pred < CUTOFF).astype(int)
            return ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean()) / 2
        lines.append('%s (n=%d)' % (name.upper(), len(ids)))
        lines.append('  ensemble AUC %.4f -> %.4f | border AUC %.4f -> %.4f | balacc %.4f -> %.4f'
                     % (roc_auc_score(y, -p2), roc_auc_score(y, -pa),
                        roc_auc_score(y[b], -p2[b]), roc_auc_score(y[b], -pa[b]), bal(p2), bal(pa)))
        e2, ea = np.abs(p2 - t), np.abs(pa - t)
        lines.append('  ensemble MAE %.3f -> %.3f  (Wilcoxon p=%.3f) | border MAE %.3f -> %.3f'
                     % (e2.mean(), ea.mean(), wilcoxon(ea, e2).pvalue, e2[b].mean(), ea[b].mean()))
        lines.append('')
    text = '\n'.join(lines)
    print('\n' + text, flush=True)
    (PUBLIC / 'comparison_2ch_vs_airway.txt').write_text(text, encoding='utf-8')


def main() -> None:
    if not (BASE / 'ensemble5.json').exists():
        raise SystemExit(f'{BASE}/ensemble5.json is missing; the 2-channel run must finish first')
    ROOT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads((BASE / 'ct/build_summary.json').read_text(
        encoding='utf-8-sig'))['records'] if r['ok']]
    train_ids = json.loads((BASE / 'split_scoring.json').read_text())['training_patient_ids']

    segment(rows)
    table = build_channel(rows)
    save(PUBLIC / 'airway_signal_check.json', signal_check(table, train_ids))
    train_and_compare()
    print('RUNAIRWAY_EXIT=0', flush=True)


if __name__ == '__main__':
    main()
