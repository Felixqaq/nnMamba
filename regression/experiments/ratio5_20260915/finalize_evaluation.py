"""Verify finished weights and score the original holdout alongside reviewed cases."""
from __future__ import annotations

import json
import shutil
import sys
import time

import torch

from run import CODE, DATA, PUBLIC, ROOT, execute, save


def main() -> None:
    split = json.loads((ROOT / 'split.json').read_text())
    reviewed = json.loads((ROOT / 'split_scoring.json').read_text())
    train = set(split['training_patient_ids'])
    test = set(split['validation_patient_ids'])
    kept = set(reviewed['validation_patient_ids'])
    assert len(train) == 700 and len(test) == 200 and not train & test
    assert kept <= test and len(kept) == 188
    assert set(reviewed['training_patient_ids']) == train
    checkpoints = []
    for seed in range(72, 77):
        path = ROOT / f'seed{seed}/regressor_seed{seed}.pth'
        blob = torch.load(path, map_location='cpu', weights_only=False)
        assert blob['seed'] == seed and blob['epochs'] == 80
        checkpoints.append(str(path))
    common = ['--config', str(ROOT / 'config.yaml'), '--source-dir', str(DATA),
              '--split-json', str(ROOT / 'split.json'),
              '--manifest', str(ROOT / 'manifest.json'),
              '--pft-csv', str(ROOT / 'clinical.csv')]
    for n in (5, 3):
        output = ROOT / f'ensemble{n}_official200.json'
        if not output.exists():
            execute(f'evaluate_{n}_official200', [sys.executable, str(ROOT / 'score.py'),
                    *common, '--checkpoints', *checkpoints[:n], '--out', str(output)])
        result = json.loads(output.read_text())
        assert set(result['patients']) == test
        assert result['fixed70']['mean_ratio']['n'] == 200
        reviewed_result = json.loads((PUBLIC / f'ensemble{n}.json').read_text())
        assert set(reviewed_result['patients']) == kept
        assert reviewed_result['fixed70']['mean_ratio']['n'] == 188
        shutil.copy2(output, PUBLIC / output.name)
    complete = json.loads((PUBLIC / 'complete.json').read_text())
    complete.update(test_reserved=200, test_scored_original=200,
                    test_scored_reviewed=188, review_excluded=12)
    save(PUBLIC / 'complete.json', complete)
    execute('report_verified', [sys.executable, str(CODE / 'report.py'), str(PUBLIC)])
    save(PUBLIC / 'status.json', {'status': 'complete', 'stage': 'verified',
         'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'train': 700, 'test_original': 200, 'test_reviewed': 188})


if __name__ == '__main__':
    main()
