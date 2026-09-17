"""Isolated expanded-cohort CT+LAA five-member evaluation with fixed official200."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import yaml

CODE = Path(__file__).resolve().parent
REPO = Path('/home/felix/Research/nnMamba')
REG = REPO / 'regression'
ROOT = REG / 'outputs/ratio5_expanded_20260915'
PUBLIC = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260915')
SOURCE = REPO / 'classification/datasets/normal_v_abnormal_fev1fvc70'
DATA = ROOT / 'ct'
OLD_SPLIT = Path('/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831/split.json')


DECISIONS = REG / 'cohort_decisions.local.json'
REVIEW = CODE / 'doctor_review_exclusions.local.json'


def _excluded_ids() -> set:
    """Patient IDs dropped from the cohort, from the gitignored decisions file.

    A missing file is fatal rather than an empty set. An empty set would let a
    patient the build already rejected back into the snapshot, and nothing
    downstream would report it.
    """
    if not DECISIONS.is_file():
        raise SystemExit(
            f'{DECISIONS} is missing; refusing to snapshot without the cohort '
            'exclusion list'
        )
    return set(json.loads(DECISIONS.read_text(encoding='utf-8'))['excluded'])


_EXCLUDED_IDS = _excluded_ids()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def stage(name: str, **extra) -> None:
    current = {'status': 'running', 'stage': name, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra}
    save(PUBLIC / 'status.json', current)
    print(json.dumps(current, ensure_ascii=False), flush=True)


def execute(name: str, command: list[str]) -> None:
    stage(name)
    with (PUBLIC / f'{name}.log').open('a') as log:
        subprocess.run(command, cwd=REG, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    if (PUBLIC / 'status.json').exists() and json.loads((PUBLIC / 'status.json').read_text()).get('status') == 'complete':
        print('Already complete', flush=True)
        return
    if not (PUBLIC / 'started.json').exists():
        save(PUBLIC / 'started.json', {'unix': time.time()})
    stage('snapshot_inputs')
    if not (ROOT / 'snapshot_ready.json').exists():
        summary = json.loads((SOURCE / 'build_summary.json').read_text())
        rows = [r for r in summary['records'] if r['ok']]
        assert len({r['patient_id'] for r in rows}) == len(rows)
        for row in rows:
            src = Path(row['path'])
            # Excluded IDs live in the gitignored decisions file: this
            # repository is public and an ID beside a clinical reason
            # identifies a person.
            assert row['patient_id'] not in _EXCLUDED_IDS
            dest = DATA / row['label'] / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(src, dest)
            row['path'] = str(dest)
        summary['records'] = rows
        save(DATA / 'build_summary.json', summary)
        shutil.copy2(OLD_SPLIT, ROOT / 'official200_original.json')
        shutil.copy2('/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv', ROOT / 'pft_original.csv')
        # Snapshot training code, while resolving existing reusable modules in WSL.
        for source, target in [(REG / 'scripts/train_fixed_holdout_ratio_regression.py', 'train.py'),
                               (CODE / 'score_snapshot.py', 'score.py')]:
            code = source.read_text(encoding='utf-8-sig')
            code = code.replace('ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(REG)!r})')
            (ROOT / target).write_text(code)
        save(ROOT / 'snapshot_ready.json', {'initial_n': len(rows)})

    if not (ROOT / 'conversion_done.json').exists():
        command = [sys.executable, str(CODE / 'build_snapshot.py'), '--out', str(DATA),
                   '--gold-json', str(REG / 'GOLD_2026_classification.json'),
                   '--series-hints', str(REG / 'datasets/generated/rq1_nva66_manifest.image.json'),
                   '--csv', str(ROOT / 'pft_original.csv'), '--workers', '2']
        execute('conversion_dry_run', command + ['--dry-run'])
        execute('conversion', command)
        save(ROOT / 'conversion_done.json', {'done': True})

    stage('audit_cohort')
    summary = json.loads((DATA / 'build_summary.json').read_text())
    rows = [r for r in summary['records'] if r['ok']]
    ids = {r['patient_id'] for r in rows}
    assert len(ids) == len(rows)
    assert all('/Test/' not in r['dicom_dir'] and '/non_PFT/' not in r['dicom_dir'] for r in rows)
    test_ids = json.loads((ROOT / 'official200_original.json').read_text())['validation_patient_ids']
    assert len(test_ids) == len(set(test_ids)) == 200 and set(test_ids) <= ids
    paths = list(DATA.glob('*/*.nii.gz'))
    assert len(paths) == len(ids), 'Duplicate or stale NIfTI in snapshot'
    for row in rows:
        assert Path(row['path']).is_file()
        assert row['label'] == ('Abnormal' if float(row['fev1_fvc_pct']) < 70 else 'Normal')
    # Patients the physician marked red: the CT or the spirometry cannot describe
    # uncomplicated COPD (inadequate effort, tumour, resected lobe, metal artefact).
    # They leave the validation half and are NOT added to training -- `ids-test_ids`
    # would otherwise sweep them straight into it, changing the 700 patients that
    # models are already being trained on and silently invalidating those weights.
    # Two split files, because training and scoring need different things.
    #
    # train.py calls set_fixed_split, which requires training + validation to name
    # exactly the volumes on disk and aborts on any extra. So the training split
    # must still cover all 900, and it does -- training never reads the validation
    # half anyway, every seed runs with --skip-holdout.
    #
    # Scoring is where the physician's review applies. Twelve patients were marked
    # red because the CT or the spirometry cannot describe uncomplicated COPD, and
    # they are dropped from the scored set. They are not moved into training:
    # `ids-test_ids` already excludes them, so the 700 training patients are
    # unchanged and weights trained before this review stay valid.
    split = {'training_patient_ids': sorted(ids-set(test_ids)),
             'validation_patient_ids': test_ids}
    save(ROOT / 'split.json', split)

    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    dropped = {row['patient_id'] for row in review['excluded_from_validation']}
    stray = dropped - set(test_ids)
    assert not stray, f'review lists {sorted(stray)}, which are not in the validation half'
    kept = [pid for pid in test_ids if pid not in dropped]
    save(ROOT / 'split_scoring.json', {
        'training_patient_ids': split['training_patient_ids'],
        'validation_patient_ids': kept,
        'doctor_excluded_from_validation': sorted(dropped),
        'note': 'scoring only; train.py must read split.json, which covers all 900'})

    save(PUBLIC / 'cohort_audit.json', {'n': len(ids), 'train': len(split['training_patient_ids']),
        'test_trained_against': len(test_ids), 'test_scored': len(kept),
        'doctor_excluded': len(dropped),
        'excluded_test_folder': True, 'conversion_failed': summary['failed'],
        'split_sha256': hashlib.sha256((ROOT / 'split.json').read_bytes()).hexdigest(),
        'label_rule': 'FEV1/FVC < 70; labels frozen from audited build summary',
        'review_source': review['source']})
    with (ROOT / 'pft_original.csv').open(encoding='utf-8-sig', newline='') as stream:
        clinical_rows = [{(k or '').strip(): (v or '').strip() for k, v in r.items()}
                         for r in csv.DictReader(stream)]
        clinical = {r['PatientID']: r for r in clinical_rows}
    # Respect reconciled hospital66 labels in build_summary, never override with a
    # second encounter's CSV ratio. GLI is blank unless the selected ratio agrees.
    with (ROOT / 'clinical.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['PatientID', 'FEV1FVC_pct', 'FEV1FVC_LLN_GLI'])
        writer.writeheader()
        for row in rows:
            original = clinical.get(row['patient_id'], {})
            same = bool(original.get('FEV1FVC_pct')) and float(original['FEV1FVC_pct']) == float(row['fev1_fvc_pct'])
            writer.writerow({'PatientID': row['patient_id'], 'FEV1FVC_pct': row['fev1_fvc_pct'],
                'FEV1FVC_LLN_GLI': original.get('FEV1FVC_LLN_GLI', '') if same else ''})

    density = ROOT / 'density'
    density.mkdir(exist_ok=True)
    old_density = REG / 'datasets/generated/laa_density_112x136x112'
    for pid in ids:
        target = density / f'{pid}.npy'
        if not target.exists() and (old_density / target.name).exists():
            os.link(old_density / target.name, target)
    missing = [r for r in rows if not (density / f"{r['patient_id']}.npy").exists()]
    save(ROOT / 'missing_density_manifest.json', {'records': missing})
    if missing:
        masks = ROOT / 'masks'
        (masks / 'lung').mkdir(parents=True, exist_ok=True)
        for row in missing:
            target = masks / 'lung' / f"{row['patient_id']}.nii.gz"
            previous = REG / 'masks/totalseg/lung' / target.name
            if previous.exists() and not target.exists():
                os.link(previous, target)
        execute('lung_masks', [sys.executable, str(REG / 'scripts/segment_lungs_totalseg.py'),
            '--manifest', str(ROOT / 'missing_density_manifest.json'), '--out', str(masks), '--fast'])
        stage('laa_density', missing=len(missing))
        sys.path.insert(0, str(REG / 'scripts'))
        import precompute_laa_density as laa
        import nibabel as nib
        derived = []
        for i, row in enumerate(missing):
            pid = row['patient_id']
            ct_image = nib.load(row['path'])
            mask_image = nib.load(str(masks / 'lung' / f'{pid}.nii.gz'))
            if ct_image.shape != mask_image.shape or not np.allclose(ct_image.affine, mask_image.affine, atol=1e-3):
                raise ValueError(f'CT and lung mask grids differ for {pid}')
            result = laa.one((pid, row['path'], str(masks / 'lung' / f'{pid}.nii.gz'),
                              (112, 136, 112), -950., density, None))
            if 'error' in result:
                raise RuntimeError(f'LAA failed: {result}')
            derived.append(result)
            stage('laa_density', done=i+1, total=len(missing))
        save(ROOT / 'new_density_audit.json', {'records': derived})
    for pid in ids:
        array = np.load(density / f'{pid}.npy', mmap_mode='r')
        assert array.shape == (2, 112, 136, 112) and array.dtype == np.uint8
        assert array[1].sum() > 0
    config = yaml.safe_load((REG / 'config.rq1.ratio_regression.2ch.yaml').read_text())
    config['data'].update(source_dir=str(DATA), manifest=str(ROOT / 'manifest.json'),
                          laa_density_dir=str(density), num_workers=4)
    config['training']['epochs'] = 80
    cfg_path = ROOT / 'config.yaml'
    cfg_path.write_text(yaml.safe_dump(config, allow_unicode=True))
    common = ['--config', str(cfg_path), '--source-dir', str(DATA),
              '--split-json', str(ROOT / 'split.json'), '--manifest', str(ROOT / 'manifest.json'),
              '--pft-csv', str(ROOT / 'clinical.csv')]
    checkpoints = []
    for seed in range(72, 77):
        output = ROOT / f'seed{seed}'
        checkpoint = output / f'regressor_seed{seed}.pth'
        if not checkpoint.exists():
            execute(f'train_seed{seed}', [sys.executable, '-u', str(ROOT / 'train.py'),
                *common, '--out', str(output), '--epochs', '80', '--seed', str(seed), '--skip-holdout'])
        checkpoints.append(str(checkpoint))
    # `common` carries split.json for training; scoring overrides it with the
    # reviewed set. argparse keeps the last value, so the trailing flag wins.
    scoring = [*common, '--split-json', str(ROOT / 'split_scoring.json')]
    execute('evaluate_five', [sys.executable, str(ROOT / 'score.py'), *scoring,
        '--checkpoints', *checkpoints, '--out', str(ROOT / 'ensemble5.json')])
    execute('evaluate_three', [sys.executable, str(ROOT / 'score.py'), *scoring,
        '--checkpoints', *checkpoints[:3], '--out', str(ROOT / 'ensemble3.json')])
    shutil.copy2(ROOT / 'ensemble5.json', PUBLIC / 'ensemble5.json')
    shutil.copy2(ROOT / 'ensemble3.json', PUBLIC / 'ensemble3.json')
    save(PUBLIC / 'complete.json', {'n': len(ids), 'train': len(ids)-200, 'test': 200,
        'seeds': list(range(72,77)), 'epochs': 80, 'artifact_root': str(ROOT),
        'elapsed_seconds': time.time()-json.loads((PUBLIC / 'started.json').read_text())['unix']})
    execute('report', [sys.executable, str(CODE / 'report.py'), str(PUBLIC)])
    save(PUBLIC / 'status.json', {'status': 'complete', 'stage': 'done'})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(PUBLIC / 'status.json', {'status': 'failed', 'error': str(error)})
        traceback.print_exc()
        raise
