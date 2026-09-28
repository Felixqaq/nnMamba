"""Segmentation-free 2-channel arm, paired against the masked 2-channel baseline.

The only difference from the baseline run is channel 1: the fraction of each
block below -950 HU over the whole volume, instead of within a TotalSegmentator
lung mask. Same cohort, split, seeds and epochs. If it holds up, the deployed
model needs no segmentation network at inference -- the channel costs ~0.5 s on
one CPU core (measured 2026-09-22).

Inside the lung the two channels agree to within 0.0001 on average (checked on
40 patients by build_maskfree_density.py); they differ only outside it, where
~28% of non-lung blocks are air the network now has to learn to ignore. That is
what the five seeds are measuring.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import time

import numpy as np

REG = Path('/home/felix/Research/nnMamba/regression')
BASE = REG / 'outputs/ratio5_expanded_20260922'            # masked 2-channel run
ROOT = REG / 'outputs/ratio5_expanded_20260922_maskfree'
PUBLIC = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260922_maskfree')
CHANNEL = ROOT / 'density_maskfree'
SEEDS = [72, 73, 74, 75, 76]
EPOCHS = 80
CUTOFF, BORDER = 70.0, 7.0


def execute(name: str, command: list[str]) -> None:
    print('[%s] %s' % (time.strftime('%H:%M:%S'), name), flush=True)
    with (PUBLIC / f'{name}.log').open('a') as fh:
        subprocess.run(command, cwd=REG, stdout=fh, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    if not (BASE / 'ensemble5.json').exists():
        raise SystemExit(f'{BASE}/ensemble5.json is missing; the masked baseline must finish first')
    rows = [r for r in json.loads((BASE / 'ct/build_summary.json').read_text(
        encoding='utf-8-sig'))['records'] if r['ok']]
    absent = [r['patient_id'] for r in rows if not (CHANNEL / f"{r['patient_id']}.npy").exists()]
    if absent:
        raise SystemExit(f'{len(absent)} patients have no mask-free channel; run '
                         f'build_maskfree_density.py first: {absent[:5]}')
    ROOT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)

    import yaml
    config = yaml.safe_load((BASE / 'config.yaml').read_text())
    if int(config['model']['in_channels']) != 2 or config['data'].get('laa_density_mode') != 'density':
        raise SystemExit('the base run is not the 2-channel density run; refusing to guess')
    config['data']['laa_density_dir'] = str(CHANNEL)
    (ROOT / 'config.yaml').write_text(
        '# Channel 1 is the fraction below -950 HU over the WHOLE volume, with no lung\n'
        '# mask; see build_maskfree_density.py. Everything else matches the masked run.\n'
        + yaml.safe_dump(config))
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
    print('RUNMASKFREE_EXIT=0', flush=True)


def compare(masked: dict, free: dict) -> None:
    from scipy.stats import wilcoxon
    from sklearn.metrics import roc_auc_score

    def members(blob):
        return {v['seed']: (v['fixed70']['auc'], v['ratio_mae']) for v in blob['per_member'].values()}

    m0, m1 = members(masked), members(free)
    lines = ['MASK-FREE vs MASKED 2-channel, paired per seed (same cohort, split, epochs)',
             '%5s | %-26s | %s' % ('seed', 'AUC  masked -> free', 'MAE  masked -> free')]
    da, dm = [], []
    for s in SEEDS:
        (a0, e0), (a1, e1) = m0[s], m1[s]
        da.append(a1 - a0)
        dm.append(e1 - e0)
        lines.append('%5d | %.4f -> %.4f  %+.4f | %6.3f -> %6.3f  %+.3f'
                     % (s, a0, a1, a1 - a0, e0, e1, e1 - e0))
    lines += ['', 'AUC delta mean %+.4f  (better in %d/5)' % (st.mean(da), sum(d > 0 for d in da)),
              'MAE delta mean %+.3f  (better in %d/5)' % (st.mean(dm), sum(d < 0 for d in dm)),
              'member AUC sd: masked %.4f  free %.4f'
              % (st.stdev(v[0] for v in m0.values()), st.stdev(v[0] for v in m1.values())), '']

    hospital66 = {str(r['patient_id']).strip() for r in json.loads(
        (REG / 'datasets/generated/rq1_nva66_manifest.image.json').read_text())['records']}
    rng = np.random.default_rng(0)
    for name, keep in (('reviewed 180', lambda p: p not in hospital66),
                       ('all validation', lambda p: True)):
        ids = sorted(p for p in masked['patients'] if keep(p))
        t = np.array([masked['patients'][p]['true_ratio'] for p in ids])
        p0 = np.array([masked['patients'][p]['mean_predicted_ratio'] for p in ids])
        p1 = np.array([free['patients'][p]['mean_predicted_ratio'] for p in ids])
        y = (t < CUTOFF).astype(int)
        b = np.abs(t - CUTOFF) < BORDER

        def bal(pred, idx=slice(None)):
            yy, yh = y[idx], (pred[idx] < CUTOFF).astype(int)
            return ((yh[yy == 1] == 1).mean() + (yh[yy == 0] == 0).mean()) / 2

        diffs = []
        for _ in range(5000):
            i = rng.integers(0, len(ids), len(ids))
            if len(set(y[i])) > 1:
                diffs.append(roc_auc_score(y[i], -p1[i]) - roc_auc_score(y[i], -p0[i]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        e0, e1 = np.abs(p0 - t), np.abs(p1 - t)
        lines.append('%s (n=%d)' % (name.upper(), len(ids)))
        lines.append('  ensemble AUC    %.4f -> %.4f  (diff 95%% CI [%+.4f, %+.4f])'
                     % (roc_auc_score(y, -p0), roc_auc_score(y, -p1), lo, hi))
        lines.append('  border AUC      %.4f -> %.4f   (n=%d)'
                     % (roc_auc_score(y[b], -p0[b]), roc_auc_score(y[b], -p1[b]), b.sum()))
        lines.append('  balacc at 70    %.4f -> %.4f' % (bal(p0), bal(p1)))
        lines.append('  ensemble MAE    %.3f -> %.3f  (Wilcoxon p=%.3f)'
                     % (e0.mean(), e1.mean(), wilcoxon(e1, e0).pvalue))
        lines.append('')
    text = '\n'.join(lines)
    print('\n' + text, flush=True)
    (PUBLIC / 'comparison_masked_vs_maskfree.txt').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
