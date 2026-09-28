"""Paired 3-channel arm: CT + LAA density + LAA occupancy.

Everything is copied from the 2-channel run that precedes it -- the same
cohort, the same splits, the same seeds, the same epoch budget -- so the only
difference is the third input channel. The occupancy channel needs no new
precomputation: the stored density arrays are already (2, D, H, W), and the
2-channel arm simply reads the first of the two.

Prints the paired per-seed comparison at the end, because a single seed cannot
settle a difference this size on this cohort (member AUC sd is ~0.008, so
anything under ~0.02 needs all five).
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import time

REG = Path('/home/felix/Research/nnMamba/regression')
BASE = REG / 'outputs/ratio5_expanded_20260918'          # the 2-channel run
ROOT = REG / 'outputs/ratio5_expanded_20260918_3ch'
PUBLIC = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260918_3ch')
SEEDS = [72, 73, 74, 75, 76]
EPOCHS = 80


def execute(name: str, command: list[str]) -> None:
    print('[%s] %s' % (time.strftime('%H:%M:%S'), name), flush=True)
    with (PUBLIC / f'{name}.log').open('a') as log:
        subprocess.run(command, cwd=REG, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    if not (BASE / 'ensemble5.json').exists():
        raise SystemExit(f'{BASE}/ensemble5.json is missing; the 2-channel arm has '
                         'not finished, and this comparison is only meaningful '
                         'against it')
    ROOT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)

    # Reuse the 2-channel run's cohort and splits verbatim. Rebuilding them would
    # risk a different split and silently break the pairing.
    import yaml
    config = yaml.safe_load((BASE / 'config.yaml').read_text())
    if int(config['model']['in_channels']) != 2:
        raise SystemExit('the base run is not 2-channel; refusing to guess')
    config['model']['in_channels'] = 3
    config['data']['laa_density_mode'] = 'density_and_occupancy'
    (ROOT / 'config.yaml').write_text(yaml.safe_dump(config))
    for name in ('train.py', 'score.py'):
        shutil.copy2(BASE / name, ROOT / name)

    common = ['--source-dir', str(BASE / 'ct'),
              '--manifest', str(ROOT / 'manifest.json'),
              '--pft-csv', str(BASE / 'clinical.csv')]

    for seed in SEEDS:
        out = ROOT / f'seed{seed}'
        if (out / f'regressor_seed{seed}.pth').exists():
            print('seed %d already trained, skipping' % seed, flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        execute(f'train_seed{seed}', [
            sys.executable, '-u', str(ROOT / 'train.py'),
            '--config', str(ROOT / 'config.yaml'),
            '--split-json', str(BASE / 'split.json'), *common,
            '--out', str(out), '--epochs', str(EPOCHS),
            '--seed', str(seed), '--skip-holdout'])

    checkpoints = [str(ROOT / f'seed{s}/regressor_seed{s}.pth') for s in SEEDS]
    for n in (5, 3):
        output = ROOT / f'ensemble{n}.json'
        if not output.exists():
            execute(f'evaluate_{n}', [
                sys.executable, '-u', str(ROOT / 'score.py'),
                '--config', str(ROOT / 'config.yaml'),
                '--split-json', str(BASE / 'split_scoring.json'), *common,
                '--checkpoints', *checkpoints[:n], '--out', str(output)])
        shutil.copy2(output, PUBLIC / output.name)

    # ---- paired comparison --------------------------------------------------
    def members(path):
        blob = json.loads(Path(path).read_text())
        return ({v['seed']: (v['fixed70']['auc'], v['ratio_mae']) for v in blob['per_member'].values()},
                blob['fixed70']['mean_ratio'])

    two, two_ens = members(BASE / 'ensemble5.json')
    three, three_ens = members(ROOT / 'ensemble5.json')

    lines = []
    lines.append('paired per seed (same cohort, same split, same epochs)')
    lines.append('%5s | %-24s | %s' % ('seed', 'AUC  2ch -> 3ch', 'MAE  2ch -> 3ch'))
    da, dm = [], []
    for seed in SEEDS:
        if seed not in two or seed not in three:
            continue
        a0, m0 = two[seed]
        a1, m1 = three[seed]
        da.append(a1 - a0)
        dm.append(m1 - m0)
        lines.append('%5d | %.4f -> %.4f  %+.4f | %6.3f -> %6.3f  %+.3f'
                     % (seed, a0, a1, a1 - a0, m0, m1, m1 - m0))
    if da:
        lines.append('')
        lines.append('AUC delta mean %+.4f  (improved in %d/%d seeds)'
                     % (st.mean(da), sum(d > 0 for d in da), len(da)))
        lines.append('MAE delta mean %+.3f  (improved in %d/%d seeds)'
                     % (st.mean(dm), sum(d < 0 for d in dm), len(dm)))
        if len(da) > 1:
            lines.append('member AUC sd: 2ch %.4f  3ch %.4f'
                         % (st.stdev([two[s][0] for s in SEEDS if s in two]),
                            st.stdev([three[s][0] for s in SEEDS if s in three])))
    lines.append('')
    lines.append('5-member ensemble  2ch AUC %.4f balacc %.4f | 3ch AUC %.4f balacc %.4f'
                 % (two_ens['auc'], two_ens['balanced_accuracy'],
                    three_ens['auc'], three_ens['balanced_accuracy']))
    for band in ('borderline', 'clear'):
        a = two_ens.get('by_band', {}).get(band, {}).get('auc')
        b = three_ens.get('by_band', {}).get(band, {}).get('auc')
        if a is not None and b is not None:
            lines.append('  %-11s 2ch %.4f -> 3ch %.4f  %+.4f' % (band, a, b, b - a))

    report = '\n'.join(lines)
    print('\n' + report, flush=True)
    (PUBLIC / 'comparison_2ch_vs_3ch.txt').write_text(report, encoding='utf-8')
    print('\nRUN3CH_EXIT=0', flush=True)


if __name__ == '__main__':
    main()
