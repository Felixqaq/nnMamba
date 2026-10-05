"""Replicate guided attention vs average pooling with fresh seeds 77-81.

The 2026-10-01 result (guided beat average pooling in AUC on 5/5 seeds 72-76,
MAE 4/5) came from one seed set. Two earlier single-set signals of this size
vanished when repeated, so this reruns both arms on the same 1301 cohort, same
split and recipe, with seeds the first run never used.

Decision rule, fixed before any result exists:
  PASS  guided beats average pooling in AUC on at least 4 of the 5 new seeds,
        AND the mean per-seed MAE difference (guided - avg) is below 0.
  else  FAIL: the deployment model stays average pooling.

Pairs are trained interleaved (avg 77, guided 77, avg 78, ...) so that an
interruption still leaves complete pairs. Resumable.
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
BASE = REG / 'outputs/ratio5_expanded_20260930_maskfree'          # avg, seeds 72-76
GUIDED = REG / 'outputs/ratio5_expanded_20260930_maskfree_guided'  # guided, seeds 72-76
AVG2 = REG / 'outputs/ratio5_expanded_20260930_maskfree_s77'
GUIDED2 = REG / 'outputs/ratio5_expanded_20260930_maskfree_guided_s77'
PUBLIC = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260930_replicate_s77')
OCC_DIRS = [REG / 'outputs/ratio5_expanded_20260929_masked/density', GUIDED / 'occupancy_new']
SEEDS = [77, 78, 79, 80, 81]
OLD_SEEDS = [72, 73, 74, 75, 76]
EPOCHS = 80
LUNG_WEIGHT, ENTROPY_WEIGHT, ENTROPY_TARGET = 0.1, 0.1, 0.8   # as in run_guided_1301.py
CUTOFF, BORDER = 70.0, 7.0


def execute(name: str, command: list[str]) -> None:
    print('[%s] %s' % (time.strftime('%H:%M:%S'), name), flush=True)
    with (PUBLIC / f'{name}.log').open('a') as fh:
        subprocess.run(command, cwd=REG, stdout=fh, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    for need in (BASE / 'train.py', BASE / 'score.py', BASE / 'config.yaml',
                 GUIDED / 'train_guided.py', GUIDED / 'config.yaml'):
        if not need.exists():
            raise SystemExit(f'{need} is missing')
    PUBLIC.mkdir(parents=True, exist_ok=True)
    common = ['--split-json', str(BASE / 'split.json'), '--source-dir', str(BASE / 'ct'),
              '--pft-csv', str(BASE / 'clinical.csv')]
    guide = []
    for d in OCC_DIRS:
        guide += ['--lung-occupancy-dir', str(d)]
    guide += ['--lung-weight', str(LUNG_WEIGHT), '--entropy-weight', str(ENTROPY_WEIGHT),
              '--entropy-target', str(ENTROPY_TARGET)]
    arms = (('avg', AVG2, BASE / 'train.py', BASE / 'config.yaml', []),
            ('guided', GUIDED2, GUIDED / 'train_guided.py', GUIDED / 'config.yaml', guide))
    for seed in SEEDS:
        for name, root, trainer, config, extra in arms:
            out = root / f'seed{seed}'
            if (out / f'regressor_seed{seed}.pth').exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            execute(f'train_{name}_seed{seed}', [
                sys.executable, '-u', str(trainer), '--config', str(config), *common,
                '--manifest', str(root / 'manifest.json'), '--out', str(out),
                '--epochs', str(EPOCHS), '--seed', str(seed), '--skip-holdout', *extra])

    for name, root, _, config, _ in arms:
        output = root / 'ensemble5.json'
        if not output.exists():
            execute(f'evaluate_{name}', [
                sys.executable, '-u', str(BASE / 'score.py'), '--config', str(config),
                '--split-json', str(BASE / 'split_scoring.json'), '--source-dir', str(BASE / 'ct'),
                '--manifest', str(root / 'manifest.json'), '--pft-csv', str(BASE / 'clinical.csv'),
                '--checkpoints', *[str(root / f'seed{s}/regressor_seed{s}.pth') for s in SEEDS],
                '--out', str(output)])
        shutil.copy2(output, PUBLIC / f'ensemble5_{name}_s77.json')
    report()
    print('RUNREPLICATE_EXIT=0', flush=True)


def members(path):
    blob = json.loads(Path(path).read_text())
    return {v['seed']: (v['fixed70']['auc'], v['ratio_mae'], v['fixed70']['balanced_accuracy'])
            for v in blob['per_member'].values()}, blob['patients']


def report() -> None:
    from scipy.stats import wilcoxon
    from sklearn.metrics import roc_auc_score

    new_avg, pa = members(AVG2 / 'ensemble5.json')
    new_gui, pg = members(GUIDED2 / 'ensemble5.json')
    old_avg, _ = members(BASE / 'ensemble5.json')
    old_gui, _ = members(GUIDED / 'ensemble5.json')

    def diffs(avg, gui, seeds):
        return ([gui[s][0] - avg[s][0] for s in seeds], [gui[s][1] - avg[s][1] for s in seeds],
                [gui[s][2] - avg[s][2] for s in seeds])

    lines = ['REPLICATION: guided vs average pooling, seeds 77-81 (new), 1035 training', '',
             'PER SEED (246 validation): AUC / MAE / balacc',
             '%5s | %-24s | %-24s | %s' % ('seed', 'avg', 'guided', 'guided - avg (AUC, MAE)')]
    for s in SEEDS:
        lines.append('%5d | %.4f %6.3f %.4f | %.4f %6.3f %.4f | %+.4f  %+.3f' % (
            s, *new_avg[s], *new_gui[s], new_gui[s][0] - new_avg[s][0], new_gui[s][1] - new_avg[s][1]))
    da, dm, db = diffs(new_avg, new_gui, SEEDS)
    auc_wins = sum(d > 0 for d in da)
    passed = auc_wins >= 4 and st.mean(dm) < 0
    lines += ['  new seeds : AUC %+.4f (better %d/5) | MAE %+.3f (better %d/5) | balacc %+.4f (better %d/5)'
              % (st.mean(da), auc_wins, st.mean(dm), sum(d < 0 for d in dm), st.mean(db), sum(d > 0 for d in db))]
    oa, om, ob = diffs(old_avg, old_gui, OLD_SEEDS)
    lines.append('  old seeds : AUC %+.4f (better %d/5) | MAE %+.3f (better %d/5) | balacc %+.4f (better %d/5)'
                 % (st.mean(oa), sum(d > 0 for d in oa), st.mean(om), sum(d < 0 for d in om),
                    st.mean(ob), sum(d > 0 for d in ob)))
    ta, tm, tb = da + oa, dm + om, db + ob
    lines.append('  all 10    : AUC %+.4f (better %d/10) | MAE %+.3f (better %d/10) | balacc %+.4f (better %d/10)'
                 % (st.mean(ta), sum(d > 0 for d in ta), st.mean(tm), sum(d < 0 for d in tm),
                    st.mean(tb), sum(d > 0 for d in tb)))
    lines += ['', 'PRE-REGISTERED RULE (AUC better on >=4/5 new seeds AND mean MAE difference < 0): %s'
              % ('PASS -> switch the deployment model to guided attention' if passed
                 else 'FAIL -> keep average pooling'), '']

    h66 = {str(r['patient_id']).strip() for r in json.loads(
        (REG / 'datasets/generated/rq1_nva66_manifest.image.json').read_text())['records']}
    for title, keep in (('REVIEWED 180', lambda p: p not in h66), ('ALL 246', lambda p: True)):
        ids = sorted(p for p in pa if keep(p))
        t = np.array([pa[p]['true_ratio'] for p in ids])
        y = (t < CUTOFF).astype(int)
        b = np.abs(t - CUTOFF) < BORDER
        lines.append('%s (n=%d), new-seed ensembles' % (title, len(ids)))
        preds = {}
        for name, pts in (('avg', pa), ('guided', pg)):
            p = np.array([pts[q]['mean_predicted_ratio'] for q in ids])
            preds[name] = p
            yh = (p < CUTOFF).astype(int)
            bal = ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean()) / 2
            lines.append('  %-7s AUC %.4f  border %.4f  balacc %.4f  MAE %.3f' % (
                name, roc_auc_score(y, -p), roc_auc_score(y[b], -p[b]), bal, np.abs(p - t).mean()))
        lines.append('  MAE Wilcoxon p = %.3f' % wilcoxon(np.abs(preds['guided'] - t),
                                                          np.abs(preds['avg'] - t)).pvalue)
        lines.append('')
    text = '\n'.join(lines)
    print('\n' + text, flush=True)
    (PUBLIC / 'replication_report.txt').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
