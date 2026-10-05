"""Guided attention pooling on the 1301-patient cohort, against both references.

Three arms on identical data (1035 training, 246 validation, seeds 72-76, 80
epochs), compared seed by seed:
  avg      ratio5_expanded_20260930_maskfree            average pooling
  plain    ratio5_expanded_20260930_maskfree_attnpool   attention, no guidance
  guided   this run                                     attention + lung prior
                                                        + entropy hinge + random
                                                        scorer init

Lung masks feed the prior during training only. The 44 patients new in this
cohort have none yet, so the first stage segments them; the inputs to the model
are the usual two channels and inference needs no segmentation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import time

import numpy as np

REG = Path('/home/felix/Research/nnMamba/regression')
EXP = REG / 'experiments/ratio5_20260918'
BASE = REG / 'outputs/ratio5_expanded_20260930_maskfree'
PLAIN = REG / 'outputs/ratio5_expanded_20260930_maskfree_attnpool'
ROOT = REG / 'outputs/ratio5_expanded_20260930_maskfree_guided'
PUBLIC = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260930_maskfree_guided')
OCC_DIRS = [REG / 'outputs/ratio5_expanded_20260929_masked/density', ROOT / 'occupancy_new']
SEEDS = [72, 73, 74, 75, 76]
EPOCHS = 80
LUNG_WEIGHT, ENTROPY_WEIGHT, ENTROPY_TARGET = 0.1, 0.1, 0.8
CUTOFF, BORDER = 70.0, 7.0


def save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')


def execute(name: str, command: list[str]) -> None:
    print('[%s] %s' % (time.strftime('%H:%M:%S'), name), flush=True)
    with (PUBLIC / f'{name}.log').open('a') as fh:
        subprocess.run(command, cwd=REG, stdout=fh, stderr=subprocess.STDOUT, check=True)


def build_missing_occupancy(rows) -> None:
    """Lung masks and occupancy for patients the masked runs never covered."""
    missing = [r for r in rows
               if not any((d / f"{r['patient_id']}.npy").exists() for d in OCC_DIRS)]
    print('occupancy missing for %d patients' % len(missing), flush=True)
    if not missing:
        return
    masks = ROOT / 'masks'
    (masks / 'lung').mkdir(parents=True, exist_ok=True)
    save(ROOT / 'missing_occupancy_manifest.json', {'records': missing})
    execute('lung_masks', [sys.executable, str(REG / 'scripts/segment_lungs_totalseg.py'),
                           '--manifest', str(ROOT / 'missing_occupancy_manifest.json'),
                           '--out', str(masks), '--fast'])
    sys.path.insert(0, str(REG / 'scripts'))
    import precompute_laa_density as laa
    out = ROOT / 'occupancy_new'
    out.mkdir(exist_ok=True)
    for row in missing:
        pid = row['patient_id']
        result = laa.one((pid, row['path'], str(masks / 'lung' / f'{pid}.nii.gz'),
                          (112, 136, 112), -950., out, None))
        if not (out / f'{pid}.npy').exists():
            raise SystemExit(f'occupancy failed for {pid}: {result}')


def main() -> None:
    for need in (BASE / 'ensemble5.json', PLAIN / 'ensemble5.json'):
        if not need.exists():
            raise SystemExit(f'{need} is missing; both reference arms must finish first')
    ROOT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads((BASE / 'ct/build_summary.json').read_text(
        encoding='utf-8-sig'))['records'] if r['ok']]
    build_missing_occupancy(rows)
    absent = [r['patient_id'] for r in rows
              if not any((d / f"{r['patient_id']}.npy").exists() for d in OCC_DIRS)]
    if absent:
        raise SystemExit(f'{len(absent)} patients still have no occupancy: {absent[:5]}')

    import yaml
    config = yaml.safe_load((BASE / 'config.yaml').read_text())
    if config['model']['name'] != 'hybrid_mamba_attention':
        raise SystemExit('the base run is not hybrid_mamba_attention; refusing to guess')
    if config['training']['batch_size'] != config['training']['swin_batch_size']:
        raise SystemExit('batch_size and swin_batch_size differ; the arms would not match')
    config['model']['name'] = 'hybrid_mamba_attnpool_guided'
    (ROOT / 'config.yaml').write_text(
        '# Identical to the mask-free base run except model.name =\n'
        '# hybrid_mamba_attnpool_guided, trained with train_ratio_regression_guided.py.\n'
        + yaml.safe_dump(config))
    # Snapshot the trainer and the localisation analysis with their repository
    # root pinned, as the other runs snapshot theirs.
    for src, dst, fix in ((REG / 'scripts/train_ratio_regression_guided.py', 'train_guided.py', True),
                          (EXP / 'attention_localization.py', 'attention_localization.py', False)):
        code = src.read_text(encoding='utf-8')
        old, new = (('ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(REG)!r})')
                    if fix else
                    ("!= 'hybrid_mamba_attnpool':",
                     "not in ('hybrid_mamba_attnpool', 'hybrid_mamba_attnpool_guided'):"))
        if code.count(old) != 1:
            raise SystemExit(f'{src.name}: expected exactly one {old!r} to rewrite')
        code = code.replace(old, new)
        (ROOT / dst).write_text(code, encoding='utf-8')
    shutil.copy2(BASE / 'score.py', ROOT / 'score.py')

    common = ['--source-dir', str(BASE / 'ct'), '--manifest', str(ROOT / 'manifest.json'),
              '--pft-csv', str(BASE / 'clinical.csv')]
    guide = []
    for d in OCC_DIRS:
        guide += ['--lung-occupancy-dir', str(d)]
    guide += ['--lung-weight', str(LUNG_WEIGHT), '--entropy-weight', str(ENTROPY_WEIGHT),
              '--entropy-target', str(ENTROPY_TARGET)]
    for seed in SEEDS:
        out = ROOT / f'seed{seed}'
        if (out / f'regressor_seed{seed}.pth').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        execute(f'train_seed{seed}', [sys.executable, '-u', str(ROOT / 'train_guided.py'),
                                      '--config', str(ROOT / 'config.yaml'),
                                      '--split-json', str(BASE / 'split.json'), *common,
                                      '--out', str(out), '--epochs', str(EPOCHS),
                                      '--seed', str(seed), '--skip-holdout', *guide])
    checkpoints = [str(ROOT / f'seed{s}/regressor_seed{s}.pth') for s in SEEDS]
    output = ROOT / 'ensemble5.json'
    if not output.exists():
        execute('evaluate_5', [sys.executable, '-u', str(ROOT / 'score.py'),
                               '--config', str(ROOT / 'config.yaml'),
                               '--split-json', str(BASE / 'split_scoring.json'), *common,
                               '--checkpoints', *checkpoints, '--out', str(output)])
    shutil.copy2(output, PUBLIC / output.name)
    compare()
    if not (PUBLIC / 'attention_localization.txt').exists():
        execute('localization', [sys.executable, '-u', str(ROOT / 'attention_localization.py'),
                                 '--config', str(ROOT / 'config.yaml'), '--checkpoints', *checkpoints,
                                 '--source-dir', str(BASE / 'ct'),
                                 '--manifest', str(ROOT / 'manifest.json'),
                                 '--split-json', str(BASE / 'split_scoring.json'),
                                 '--maskfree-dir', str(BASE / 'density_maskfree'),
                                 '--clinical', str(BASE / 'clinical.csv'), '--out', str(PUBLIC)])
    print((PUBLIC / 'attention_localization.txt').read_text(encoding='utf-8'), flush=True)
    print('RUNGUIDED1301_EXIT=0', flush=True)


def compare() -> None:
    from scipy.stats import wilcoxon
    from sklearn.metrics import roc_auc_score

    arms = {'avg': BASE, 'plain': PLAIN, 'guided': ROOT}
    blobs = {k: json.loads((d / 'ensemble5.json').read_text()) for k, d in arms.items()}
    members = {k: {v['seed']: (v['fixed70']['auc'], v['ratio_mae'], v['fixed70']['balanced_accuracy'])
                   for v in b['per_member'].values()} for k, b in blobs.items()}
    lines = ['GUIDED vs PLAIN attention vs AVERAGE pooling, 1035 training, paired by seed', '',
             'PER SEED (246 validation): AUC / MAE / balacc',
             '%5s | %-24s | %-24s | %-24s' % ('seed', 'avg', 'plain', 'guided')]
    for s in SEEDS:
        lines.append('%5d | %s | %s | %s' % (s, *('%.4f %6.3f %.4f' % members[k][s] for k in arms)))
    for ref in ('avg', 'plain'):
        da = [members['guided'][s][0] - members[ref][s][0] for s in SEEDS]
        dm = [members['guided'][s][1] - members[ref][s][1] for s in SEEDS]
        db = [members['guided'][s][2] - members[ref][s][2] for s in SEEDS]
        lines.append('  guided - %-5s AUC %+.4f (better %d/5) | MAE %+.3f (better %d/5) | '
                     'balacc %+.4f (better %d/5)' % (ref, st.mean(da), sum(d > 0 for d in da),
                                                    st.mean(dm), sum(d < 0 for d in dm),
                                                    st.mean(db), sum(d > 0 for d in db)))
    lines.append('')
    h66 = {str(r['patient_id']).strip() for r in json.loads(
        (REG / 'datasets/generated/rq1_nva66_manifest.image.json').read_text())['records']}
    for name, keep in (('REVIEWED 180', lambda p: p not in h66), ('ALL 246', lambda p: True)):
        ids = sorted(p for p in blobs['avg']['patients'] if keep(p))
        t = np.array([blobs['avg']['patients'][p]['true_ratio'] for p in ids])
        y = (t < CUTOFF).astype(int)
        b = np.abs(t - CUTOFF) < BORDER
        lines.append('%s (n=%d)' % (name, len(ids)))
        lines.append('  %-7s %7s %8s %7s %7s' % ('', 'AUC', 'border', 'balacc', 'MAE'))
        preds = {}
        for k in arms:
            p = np.array([blobs[k]['patients'][q]['mean_predicted_ratio'] for q in ids])
            preds[k] = p
            yh = (p < CUTOFF).astype(int)
            bal = ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean()) / 2
            lines.append('  %-7s %7.4f %8.4f %7.4f %7.3f' % (k, roc_auc_score(y, -p),
                                                            roc_auc_score(y[b], -p[b]), bal,
                                                            np.abs(p - t).mean()))
        for ref in ('avg', 'plain'):
            lines.append('  guided vs %-5s MAE Wilcoxon p = %.3f' % (
                ref, wilcoxon(np.abs(preds['guided'] - t), np.abs(preds[ref] - t)).pvalue))
        lines.append('')
    lines.append('Single ensembles move ~0.02 AUC / ~0.3 MAE on the 180 by chance; the per-seed')
    lines.append('pairs are the part of this comparison that controls for that.')
    text = '\n'.join(lines)
    print('\n' + text, flush=True)
    (PUBLIC / 'comparison_avg_plain_guided.txt').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
