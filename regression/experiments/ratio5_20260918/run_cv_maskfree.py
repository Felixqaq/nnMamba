"""Cross-validated ensemble on the segmentation-free channel.

Combines the two changes that measured well on 2026-09-23: fold members
instead of seed members (balanced accuracy +0.039, CI excluding 0) and the
mask-free emphysema channel (no worse, and no segmentation network at
inference). Folds are reused from the masked CV run so the two are paired.
The comparison here is against the mask-free SEED ensemble, which isolates
what folding adds on this channel.

Original notes follow.


Two questions, answered on the 1149-patient cohort:

  Does a five-fold ensemble beat the five-seed ensemble? The seed ensemble's
  members all train on the same 883 patients and differ only in initialisation
  and data order. Fold members each train on a different 80%, so they disagree
  more -- which is what an ensemble needs -- at the cost of seeing 20% fewer
  patients each. Same number of trainings, so the comparison costs nothing
  extra.

  Can a calibrated cutoff recover the balanced accuracy that did not improve
  with more data? On the 180 shared patients the 817-patient model beat the
  700-patient one on AUC and MAE but lost 0.021 balanced accuracy at the fixed
  cutoff of 70. The cutoff cannot be tuned on training-set predictions -- the
  model has nearly memorised those (in-sample MAE ~0.8) -- and must not be tuned
  on the validation set. The fold models give an honest prediction for every
  training patient, so the cutoff is chosen on those and only then applied to
  validation.

The label stays FEV1/FVC < 70. Only the threshold applied to the *predicted*
ratio moves.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REG = Path('/home/felix/Research/nnMamba/regression')
BASE = REG / 'outputs/ratio5_expanded_20260922'
ROOT = REG / 'outputs/ratio5_expanded_20260922_cv_maskfree'
MASKFREE = REG / 'outputs/ratio5_expanded_20260922_maskfree'
MASKED_CV = REG / 'outputs/ratio5_expanded_20260922_cv'
CONFIG = MASKFREE / 'config.yaml'          # 2 channels, mask-free density
SEEDENS = MASKFREE / 'ensemble5.json'      # the arm this one must beat
PUBLIC = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260922_cv_maskfree')
FOLDS = 5
SEEDS = [72, 73, 74, 75, 76]
EPOCHS = 80
FOLD_SEED = 20260922
CUTOFF = 70.0
BORDER = 7.0


def save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')


def execute(name: str, command: list[str]) -> None:
    print('[%s] %s' % (time.strftime('%H:%M:%S'), name), flush=True)
    with (PUBLIC / f'{name}.log').open('a') as log:
        subprocess.run(command, cwd=REG, stdout=log, stderr=subprocess.STDOUT, check=True)


def score(name: str, split: Path, checkpoints: list[str], out: Path) -> dict:
    if not out.exists():
        execute(name, [sys.executable, '-u', str(BASE / 'score.py'),
                       '--config', str(CONFIG),
                       '--source-dir', str(BASE / 'ct'),
                       '--split-json', str(split),
                       '--manifest', str(ROOT / 'manifest.json'),
                       '--pft-csv', str(BASE / 'clinical.csv'),
                       '--checkpoints', *checkpoints, '--out', str(out)])
    return json.loads(out.read_text())['patients']


# ---------------------------------------------------------------- metrics ---
def metrics(true, pred, cutoff):
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    y = (true < CUTOFF).astype(int)
    yhat = (pred < cutoff).astype(int)
    sens = float((yhat[y == 1] == 1).mean())
    spec = float((yhat[y == 0] == 0).mean())
    border = np.abs(true - CUTOFF) < BORDER
    out = {'n': int(len(true)), 'auc': float(roc_auc_score(y, -pred)),
           'balacc': (sens + spec) / 2, 'sens': sens, 'spec': spec,
           'mae': float(np.abs(pred - true).mean())}
    if len(set(y[border])) > 1:
        out['border_auc'] = float(roc_auc_score(y[border], -pred[border]))
        out['border_n'] = int(border.sum())
    return out


def best_cutoff(true, pred):
    """Cutoff on the predicted ratio that maximises balanced accuracy."""
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    y = (true < CUTOFF).astype(int)
    grid = np.round(np.arange(60.0, 80.01, 0.1), 1)
    scores = []
    for c in grid:
        yhat = (pred < c).astype(int)
        scores.append(((yhat[y == 1] == 1).mean() + (yhat[y == 0] == 0).mean()) / 2)
    scores = np.array(scores)
    # Ties are common on a 0.1 grid; take the middle of the best plateau so the
    # choice is not decided by which end of the plateau comes first.
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    return float(grid[best[len(best) // 2]]), float(scores.max())


def boot(true, a, b, cutoff_a, cutoff_b, n=5000, seed=0):
    """95% CIs of (b - a) for AUC, MAE and balanced accuracy, patients resampled jointly."""
    true, a, b = (np.asarray(v, float) for v in (true, a, b))
    rng = np.random.default_rng(seed)
    y = (true < CUTOFF).astype(int)
    out = {'auc': [], 'mae': [], 'balacc': []}
    for _ in range(n):
        i = rng.integers(0, len(true), len(true))
        if len(set(y[i])) < 2:
            continue
        ma, mb = metrics(true[i], a[i], cutoff_a), metrics(true[i], b[i], cutoff_b)
        for k in out:
            out[k].append(mb[k] - ma[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in out.items()}


# ------------------------------------------------------------------- main ---
def main() -> None:
    for needed in (SEEDENS, CONFIG, BASE / 'split_scoring.json'):
        if not needed.exists():
            raise SystemExit(f'{needed} is missing; the mask-free seed run must '
                             'finish first -- it is the comparison')
    ROOT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)

    scoring = json.loads((BASE / 'split_scoring.json').read_text())
    full = json.loads((BASE / 'split.json').read_text())
    train = sorted(scoring['training_patient_ids'])
    held = sorted(full['validation_patient_ids'])
    if set(train) & set(held):
        raise SystemExit('training and held-out overlap')

    ratio = {}
    with (BASE / 'clinical.csv').open(encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            ratio[row['PatientID']] = float(row['FEV1FVC_pct'])
    strata = [2 * int(ratio[p] < CUTOFF) + int(abs(ratio[p] - CUTOFF) < BORDER) for p in train]

    folds_path = ROOT / 'folds.json'
    if not folds_path.exists() and (MASKED_CV / 'folds.json').exists():
        # Same patients in the same folds as the masked CV run: the two arms then
        # differ only by the channel, which is the whole point of the comparison.
        shutil.copy2(MASKED_CV / 'folds.json', folds_path)
    if folds_path.exists():
        folds = json.loads(folds_path.read_text())['folds']
    else:
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=FOLD_SEED)
        folds = [[train[i] for i in test] for _, test in skf.split(train, strata)]
        save(folds_path, {'seed': FOLD_SEED, 'strata': 'abnormal x borderline',
                          'folds': folds})
    if sorted(p for f in folds for p in f) != train:
        raise SystemExit('folds do not partition the training set exactly')

    # ---- train one model per fold ------------------------------------------
    checkpoints, oof = [], {}
    for k, (seed, fold) in enumerate(zip(SEEDS, folds)):
        rest = sorted(set(train) - set(fold))
        split = ROOT / f'split_fold{k}.json'
        save(split, {'training_patient_ids': rest,
                     'validation_patient_ids': sorted(set(fold) | set(held)),
                     'note': f'fold {k}: trains on the other four folds; covers the '
                             'whole cohort as set_fixed_split requires'})
        out = ROOT / f'fold{k}'
        ckpt = out / f'regressor_seed{seed}.pth'
        if not ckpt.exists():
            out.mkdir(parents=True, exist_ok=True)
            execute(f'train_fold{k}', [
                sys.executable, '-u', str(BASE / 'train.py'),
                '--config', str(CONFIG),
                '--source-dir', str(BASE / 'ct'), '--split-json', str(split),
                '--manifest', str(ROOT / 'manifest.json'),
                '--pft-csv', str(BASE / 'clinical.csv'),
                '--out', str(out), '--epochs', str(EPOCHS),
                '--seed', str(seed), '--skip-holdout'])
        checkpoints.append(str(ckpt))

        oof_split = ROOT / f'split_oof{k}.json'
        save(oof_split, {'training_patient_ids': rest, 'validation_patient_ids': sorted(fold),
                         'note': f'out-of-fold scoring for fold {k}'})
        oof.update(score(f'oof_fold{k}', oof_split, [str(ckpt)], ROOT / f'oof_fold{k}.json'))

    missing = sorted(set(train) - set(oof))
    if missing:
        raise SystemExit(f'{len(missing)} training patients have no out-of-fold prediction')

    # ---- validation ---------------------------------------------------------
    cv = score('evaluate_cv5', BASE / 'split_scoring.json', checkpoints,
               ROOT / 'ensemble5_cv.json')
    seedens = json.loads(SEEDENS.read_text())['patients']
    shutil.copy2(ROOT / 'ensemble5_cv.json', PUBLIC / 'ensemble5_cv.json')

    oof_true = [oof[p]['true_ratio'] for p in train]
    oof_pred = [oof[p]['mean_predicted_ratio'] for p in train]
    cut, oof_bal = best_cutoff(oof_true, oof_pred)

    kept = sorted(scoring['validation_patient_ids'])
    hospital66 = {str(r['patient_id']).strip() for r in json.loads(
        (REG / 'datasets/generated/rq1_nva66_manifest.image.json').read_text())['records']}
    reviewed = [p for p in kept if p not in hospital66]
    if len(reviewed) != 180:
        raise SystemExit(f'expected the 180 reviewer-kept patients, found {len(reviewed)}')

    def arrays(src, ids):
        return ([src[p]['true_ratio'] for p in ids], [src[p]['mean_predicted_ratio'] for p in ids])

    report = {'cutoff_from_oof': cut, 'oof_balacc_at_cutoff': oof_bal,
              'oof': {'at_70': metrics(oof_true, oof_pred, CUTOFF),
                      'at_cutoff': metrics(oof_true, oof_pred, cut)},
              'sets': {}}
    lines = ['OUT-OF-FOLD (883 training patients, one fold model each)',
             '  at 70       : %s' % fmt(report['oof']['at_70']),
             '  best cutoff : %.1f  (balacc %.4f)' % (cut, oof_bal), '']
    for name, ids in (('reviewed 180', reviewed), ('all validation', kept)):
        t, s = arrays(seedens, ids)
        _, c = arrays(cv, ids)
        rows = {'maskfree seed @70': metrics(t, s, CUTOFF),
                'maskfree CV   @70': metrics(t, c, CUTOFF),
                'maskfree seed @%.1f' % cut: metrics(t, s, cut),
                'maskfree CV   @%.1f' % cut: metrics(t, c, cut)}
        ci_arch = boot(t, s, c, CUTOFF, CUTOFF)
        ci_cal = boot(t, c, c, CUTOFF, cut)
        report['sets'][name] = {'rows': rows, 'ci_cv_minus_seed_at_70': ci_arch,
                                'ci_cv_calibrated_minus_cv_at_70': ci_cal}
        lines.append('%s (n=%d)' % (name.upper(), len(ids)))
        for label, m in rows.items():
            lines.append('  %-20s %s' % (label, fmt(m)))
        lines.append('  CV - seed at 70, 95%% CI   AUC [%+.4f, %+.4f]  MAE [%+.3f, %+.3f]  '
                     'balacc [%+.4f, %+.4f]' % (*ci_arch['auc'], *ci_arch['mae'], *ci_arch['balacc']))
        lines.append('  CV calibrated - CV at 70   balacc [%+.4f, %+.4f]' % tuple(ci_cal['balacc']))
        lines.append('')

    text = '\n'.join(lines)
    print('\n' + text, flush=True)
    (PUBLIC / 'cv_report.txt').write_text(text, encoding='utf-8')
    save(PUBLIC / 'cv_report.json', report)
    print('RUNCVMASKFREE_EXIT=0', flush=True)


def fmt(m: dict) -> str:
    border = ('  border AUC %.4f' % m['border_auc']) if 'border_auc' in m else ''
    return ('AUC %.4f  balacc %.4f  sens %.3f  spec %.3f  MAE %.3f%s'
            % (m['auc'], m['balacc'], m['sens'], m['spec'], m['mae'], border))


if __name__ == '__main__':
    main()
