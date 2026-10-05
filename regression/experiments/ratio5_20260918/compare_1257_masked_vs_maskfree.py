"""Masked vs mask-free emphysema channel on the 1257-patient cohort, paired by seed.

Both arms train on the same 991 patients with the same five seeds and epochs;
only channel 1 differs. Per-seed pairing is the part of this comparison that
controls for retraining noise -- a single ensemble-vs-ensemble difference on the
180 does not (see the memory note on run-to-run spread: ~0.02 AUC, ~0.3 MAE).
"""

import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

R = Path('/home/felix/Research/nnMamba/regression/outputs')
MASKED = R / 'ratio5_expanded_20260929_masked'
FREE = R / 'ratio5_expanded_20260929_maskfree'
MASKED_883 = R / 'ratio5_expanded_20260922'
OUT = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260929_masked'
           '/comparison_masked_vs_maskfree_1257.txt')
SEEDS = [72, 73, 74, 75, 76]
CUTOFF, BORDER = 70.0, 7.0

h66 = {str(r['patient_id']).strip() for r in json.loads(
    (R.parent / 'datasets/generated/rq1_nva66_manifest.image.json').read_text())['records']}


def load(d):
    return json.loads((d / 'ensemble5.json').read_text())


def check_same_split():
    a = json.loads((MASKED / 'split_scoring.json').read_text())
    b = json.loads((FREE / 'split_scoring.json').read_text())
    for key in ('training_patient_ids', 'validation_patient_ids'):
        if sorted(a[key]) != sorted(b[key]):
            raise SystemExit(f'{key} differs between the arms; the pairing is broken')


def metrics(t, p):
    y = (t < CUTOFF).astype(int)
    yh = (p < CUTOFF).astype(int)
    b = np.abs(t - CUTOFF) < BORDER
    return {'auc': roc_auc_score(y, -p), 'border': roc_auc_score(y[b], -p[b]),
            'balacc': ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean()) / 2,
            'mae': np.abs(p - t).mean()}


def main():
    check_same_split()
    m, f, m883 = load(MASKED), load(FREE), load(MASKED_883)
    lines = ['MASKED vs MASK-FREE, 1257 cohort, 991 training, paired by seed', '',
             'PER SEED (all 246 validation patients)',
             '%5s | %-26s | %s' % ('seed', 'AUC  masked -> free', 'MAE  masked -> free')]

    def members(blob):
        return {v['seed']: (v['fixed70']['auc'], v['ratio_mae']) for v in blob['per_member'].values()}
    mm, mf = members(m), members(f)
    da, dm = [], []
    for s in SEEDS:
        (a0, e0), (a1, e1) = mm[s], mf[s]
        da.append(a1 - a0)
        dm.append(e1 - e0)
        lines.append('%5d | %.4f -> %.4f  %+.4f | %6.3f -> %6.3f  %+.3f' % (s, a0, a1, a1 - a0, e0, e1, e1 - e0))
    lines += ['  AUC delta mean %+.4f (free better in %d/5) | MAE delta mean %+.3f (free better in %d/5)'
              % (st.mean(da), sum(d > 0 for d in da), st.mean(dm), sum(d < 0 for d in dm)), '']

    for name, keep in (('REVIEWED 180', lambda p: p not in h66), ('ALL 246', lambda p: True)):
        ids = sorted(p for p in m['patients'] if keep(p))
        t = np.array([m['patients'][p]['true_ratio'] for p in ids])
        rows = {'masked   883 train': np.array([m883['patients'][p]['mean_predicted_ratio'] for p in ids]),
                'masked   991 train': np.array([m['patients'][p]['mean_predicted_ratio'] for p in ids]),
                'maskfree 991 train': np.array([f['patients'][p]['mean_predicted_ratio'] for p in ids])}
        lines.append('%s (n=%d)' % (name, len(ids)))
        lines.append('  %-20s %7s %8s %7s %7s' % ('', 'AUC', 'border', 'balacc', 'MAE'))
        for label, p in rows.items():
            x = metrics(t, p)
            lines.append('  %-20s %7.4f %8.4f %7.4f %7.3f' % (label, x['auc'], x['border'], x['balacc'], x['mae']))
        e_m = np.abs(rows['masked   991 train'] - t)
        e_f = np.abs(rows['maskfree 991 train'] - t)
        lines.append('  masked vs maskfree (991): MAE Wilcoxon p = %.3f' % wilcoxon(e_f, e_m).pvalue)
        lines.append('')
    lines.append('Read against the run-to-run spread of a single 5-seed ensemble on the 180:')
    lines.append('about 0.02 AUC and 0.3 MAE. Differences inside that are indistinguishable.')
    text = '\n'.join(lines)
    print(text)
    OUT.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
