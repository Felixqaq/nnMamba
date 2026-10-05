"""Attention pooling vs average pooling, paired by seed, on the deployment recipe.

Base: the 1257-patient mask-free run (991 training, 246 validation, seeds
72-76, 80 epochs). The only change is the model: hybrid_mamba_attnpool, a
subclass of hybrid_mamba_attention that replaces average pooling with attention
pooling and leaves the parent class untouched. Because the attention scorer
starts at zero, each attention model begins exactly as the average-pooling model
and the head keeps its size, so any difference comes from the pooling alone.

Two questions:
  1. Is the regression better? Judged per seed (five pairs), because one
     ensemble against another moves ~0.02 AUC / ~0.3 MAE on the 180 by chance.
  2. Does attention land on disease? attention_localization.py, measured
     against lung and emphysema maps the model never sees.
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
EXP = REG / 'experiments/ratio5_20260918'
BASE = REG / 'outputs/ratio5_expanded_20260929_maskfree'
ROOT = REG / 'outputs/ratio5_expanded_20260929_maskfree_attnpool'
PUBLIC = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260929_maskfree_attnpool')
SEEDS = [72, 73, 74, 75, 76]
EPOCHS = 80
CUTOFF, BORDER = 70.0, 7.0


def execute(name: str, command: list[str]) -> None:
    print('[%s] %s' % (time.strftime('%H:%M:%S'), name), flush=True)
    with (PUBLIC / f'{name}.log').open('a') as fh:
        subprocess.run(command, cwd=REG, stdout=fh, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    if not (BASE / 'ensemble5.json').exists():
        raise SystemExit(f'{BASE}/ensemble5.json is missing; the average-pooling base must exist')
    ROOT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)

    import yaml
    config = yaml.safe_load((BASE / 'config.yaml').read_text())
    if config['model']['name'] != 'hybrid_mamba_attention':
        raise SystemExit('the base run is not hybrid_mamba_attention; refusing to guess')
    # The loader picks the batch size by model name: the parent reads
    # swin_batch_size, the new name reads batch_size. They must agree or the
    # two arms would train at different batch sizes.
    if config['training']['batch_size'] != config['training']['swin_batch_size']:
        raise SystemExit('batch_size and swin_batch_size differ; the arms would not match')
    config['model']['name'] = 'hybrid_mamba_attnpool'
    (ROOT / 'config.yaml').write_text(
        '# Identical to the mask-free base run except model.name = hybrid_mamba_attnpool\n'
        '# (HybridMambaAttnPoolRegressor: the same network with attention pooling).\n'
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

    if not (PUBLIC / 'attention_localization.txt').exists():
        execute('localization', [sys.executable, '-u', str(EXP / 'attention_localization.py'),
                                 '--config', str(ROOT / 'config.yaml'),
                                 '--checkpoints', *checkpoints,
                                 '--source-dir', str(BASE / 'ct'),
                                 '--manifest', str(ROOT / 'manifest.json'),
                                 '--split-json', str(BASE / 'split_scoring.json'),
                                 '--maskfree-dir', str(BASE / 'density_maskfree'),
                                 '--clinical', str(BASE / 'clinical.csv'),
                                 '--out', str(PUBLIC)])
    print((PUBLIC / 'attention_localization.txt').read_text(encoding='utf-8'), flush=True)
    print('RUNATTNPOOL_EXIT=0', flush=True)


def compare(avg: dict, att: dict) -> None:
    from scipy.stats import wilcoxon
    from sklearn.metrics import roc_auc_score

    def members(blob):
        return {v['seed']: (v['fixed70']['auc'], v['ratio_mae'], v['fixed70']['balanced_accuracy'])
                for v in blob['per_member'].values()}
    m0, m1 = members(avg), members(att)
    lines = ['ATTENTION vs AVERAGE pooling, mask-free, 991 training, paired by seed', '',
             'PER SEED (246 validation)',
             '%5s | %-24s | %-24s | %s' % ('seed', 'AUC avg -> attn', 'MAE avg -> attn', 'balacc')]
    da, dm, db = [], [], []
    for s in SEEDS:
        (a0, e0, b0), (a1, e1, b1) = m0[s], m1[s]
        da.append(a1 - a0); dm.append(e1 - e0); db.append(b1 - b0)
        lines.append('%5d | %.4f -> %.4f %+.4f | %6.3f -> %6.3f %+.3f | %+.4f'
                     % (s, a0, a1, a1 - a0, e0, e1, e1 - e0, b1 - b0))
    lines += ['  AUC %+.4f (better %d/5) | MAE %+.3f (better %d/5) | balacc %+.4f (better %d/5)'
              % (st.mean(da), sum(d > 0 for d in da), st.mean(dm), sum(d < 0 for d in dm),
                 st.mean(db), sum(d > 0 for d in db)), '']

    h66 = {str(r['patient_id']).strip() for r in json.loads(
        (REG / 'datasets/generated/rq1_nva66_manifest.image.json').read_text())['records']}
    for name, keep in (('REVIEWED 180', lambda p: p not in h66), ('ALL 246', lambda p: True)):
        ids = sorted(p for p in avg['patients'] if keep(p))
        t = np.array([avg['patients'][p]['true_ratio'] for p in ids])
        p0 = np.array([avg['patients'][p]['mean_predicted_ratio'] for p in ids])
        p1 = np.array([att['patients'][p]['mean_predicted_ratio'] for p in ids])
        y = (t < CUTOFF).astype(int)
        b = np.abs(t - CUTOFF) < BORDER

        def bal(p):
            yh = (p < CUTOFF).astype(int)
            return ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean()) / 2
        e0, e1 = np.abs(p0 - t), np.abs(p1 - t)
        lines += ['%s (n=%d)' % (name, len(ids)),
                  '  AUC %.4f -> %.4f | border %.4f -> %.4f | balacc %.4f -> %.4f'
                  % (roc_auc_score(y, -p0), roc_auc_score(y, -p1), roc_auc_score(y[b], -p0[b]),
                     roc_auc_score(y[b], -p1[b]), bal(p0), bal(p1)),
                  '  MAE %.3f -> %.3f (Wilcoxon p=%.3f)' % (e0.mean(), e1.mean(),
                                                         wilcoxon(e1, e0).pvalue), '']
    lines.append('Single ensembles move ~0.02 AUC / ~0.3 MAE on the 180 by chance; the per-seed')
    lines.append('pairs above are the part of this comparison that controls for that.')
    text = '\n'.join(lines)
    print('\n' + text, flush=True)
    (PUBLIC / 'comparison_avg_vs_attention.txt').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
