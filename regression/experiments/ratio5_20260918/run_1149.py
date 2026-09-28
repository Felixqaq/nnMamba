"""Expanded 1149-patient CT+LAA five-member run, with hospital66 held out.

Same pipeline as the 1083-patient run of 2026-09-18, plus the four August 2025
pulls added on 2026-09-22 (66 labelled patients). Validation is unchanged -- the
same 180 reviewer-kept patients plus the same 66 hospital66 patients -- so the
scores are directly comparable with that run. Only training grows, 817 -> 883.

Original notes follow.

Two changes from the 900-patient run:

  the cohort grew by 183 patients from the twelve weekly pulls that predate
  20251127, whose PFT pages were curated on 2026-09-18;

  the 66 physician-confirmed hospital66 patients move out of training and join
  the validation set, which the reviewer's 20 exclusions had left at 180.

Validation is therefore 180 + 66 = 246. Those 66 are a much easier case mix --
9% borderline against 44% -- so an AUC here is not comparable with the 0.8222
measured on the 188 or the 0.8158 on the 180. The per-band numbers in the score
output are the ones to read across runs.

Resumable: every stage writes a marker and is skipped if the marker exists.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import time

REPO = Path('/home/felix/Research/nnMamba')
REG = REPO / 'regression'
ROOT = REG / 'outputs/ratio5_expanded_20260922'
PUBLIC = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260922')
SOURCE = REPO / 'classification/datasets/normal_v_abnormal_fev1fvc70'
PREV = REG / 'outputs/ratio5_expanded_20260918'
PREV2 = REG / 'outputs/ratio5_expanded_20260915'
DATA = ROOT / 'ct'
DENSITY = ROOT / 'density'

DECISIONS = REG / 'cohort_decisions.local.json'
REVIEW = Path('/mnt/d/Felix/Hospital/nnMamba/regression/experiments/ratio5_20260915'
              '/doctor_review_exclusions.local.json')
HOSPITAL66 = REG / 'datasets/generated/rq1_nva66_manifest.image.json'
SEEDS = [72, 73, 74, 75, 76]
EPOCHS = 80


def save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')


def stage(name: str, **extra) -> None:
    save(PUBLIC / 'status.json', {'stage': name, 'unix': time.time(), **extra})
    print('[%s] %s %s' % (time.strftime('%H:%M:%S'), name,
                          json.dumps(extra, ensure_ascii=False) if extra else ''),
          flush=True)


def execute(name: str, command: list[str]) -> None:
    stage(name)
    with (PUBLIC / f'{name}.log').open('a') as log:
        subprocess.run(command, cwd=REG, stdout=log, stderr=subprocess.STDOUT, check=True)


def excluded_ids() -> set:
    if not DECISIONS.is_file():
        raise SystemExit(f'{DECISIONS} is missing; refusing to build a cohort '
                         'without the exclusion list')
    return set(json.loads(DECISIONS.read_text(encoding='utf-8'))['excluded'])


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)

    # ---- 1. snapshot the converted CT ---------------------------------------
    stage('snapshot_inputs')
    summary = json.loads((SOURCE / 'build_summary.json').read_text(encoding='utf-8-sig'))
    rows = [r for r in summary['records'] if r['ok']]
    if len({r['patient_id'] for r in rows}) != len(rows):
        raise SystemExit('duplicate patient ids in the build summary')
    dropped = excluded_ids()
    leaked = sorted({r['patient_id'] for r in rows} & dropped)
    if leaked:
        raise SystemExit(f'excluded patients present in the build: {leaked}')

    for row in rows:
        src = Path(row['path'])
        dest = DATA / row['label'] / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(src, dest)
        row['path'] = str(dest)
    summary['records'] = rows
    save(DATA / 'build_summary.json', summary)
    ids = {r['patient_id'] for r in rows}
    stage('snapshot_inputs', patients=len(ids))

    # ---- 2. clinical table --------------------------------------------------
    if not (ROOT / 'pft_original.csv').exists():
        shutil.copy2('/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv',
                     ROOT / 'pft_original.csv')
    with (ROOT / 'pft_original.csv').open(encoding='utf-8-sig', newline='') as stream:
        clinical = {r['PatientID']: r for r in
                    ({(k or '').strip(): (v or '').strip() for k, v in row.items()}
                     for row in csv.DictReader(stream)) if r.get('PatientID')}
    # Respect the reconciled label in build_summary; never overwrite it with a
    # second encounter's CSV ratio. GLI is left blank unless the two agree.
    with (ROOT / 'clinical.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['PatientID', 'FEV1FVC_pct',
                                                    'FEV1FVC_LLN_GLI'])
        writer.writeheader()
        for row in rows:
            original = clinical.get(row['patient_id'], {})
            same = (bool(original.get('FEV1FVC_pct'))
                    and float(original['FEV1FVC_pct']) == float(row['fev1_fvc_pct']))
            writer.writerow({
                'PatientID': row['patient_id'],
                'FEV1FVC_pct': row['fev1_fvc_pct'],
                'FEV1FVC_LLN_GLI': original.get('FEV1FVC_LLN_GLI', '') if same else ''})

    # ---- 3. splits ----------------------------------------------------------
    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    kept = set(review['validation_patient_ids'])
    doctor_excluded = {e['patient_id'] for e in review['excluded_from_validation']}
    ids66 = {str(r['patient_id']).strip()
             for r in json.loads(HOSPITAL66.read_text(encoding='utf-8'))['records']}

    for name, group in (('reviewer-kept', kept), ('reviewer-excluded', doctor_excluded),
                        ('hospital66', ids66)):
        missing = sorted(group - ids)
        if missing:
            raise SystemExit(f'{name} patients absent from the cohort: {missing}')

    validation = sorted(kept | ids66)
    held = sorted(kept | ids66 | doctor_excluded)
    training = sorted(ids - set(held))
    if set(training) & set(validation):
        raise SystemExit('a patient cannot be in both halves')
    if len(training) + len(held) != len(ids):
        raise SystemExit('split does not cover the cohort exactly')

    save(ROOT / 'split.json', {
        'training_patient_ids': training,
        'validation_patient_ids': held,
        'note': 'covers the whole cohort, as set_fixed_split requires. Held out = '
                'the reviewer-kept 180, the 66 hospital66 patients moved out of '
                'training, and the 20 the reviewer excluded.'})

    ratio = {}
    with (ROOT / 'clinical.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            ratio[row['PatientID']] = float(row['FEV1FVC_pct'])

    def mix(group):
        vals = [ratio[p] for p in group if p in ratio]
        return {'n': len(vals),
                'abnormal_pct': round(100 * sum(v < 70 for v in vals) / len(vals), 1),
                'borderline_pct': round(100 * sum(abs(v - 70) < 7 for v in vals) / len(vals), 1),
                'ratio_mean': round(st.mean(vals), 1),
                'ratio_sd': round(st.stdev(vals), 1)}

    save(ROOT / 'split_scoring.json', {
        'training_patient_ids': training,
        'validation_patient_ids': validation,
        'doctor_excluded_from_validation': sorted(doctor_excluded),
        'note': 'scoring only; train.py must read split.json, which covers the '
                'whole cohort.',
        'case_mix': {'reviewer_kept_180': mix(sorted(kept)),
                     'hospital66_added': mix(sorted(ids66)),
                     'combined_validation': mix(validation),
                     'training': mix(training),
                     'warning': 'hospital66 is far easier than the reviewed set, so '
                                'the headline AUC here is not comparable with the '
                                '0.8222 on the 188 or the 0.8158 on the 180. Compare '
                                'the per-band numbers instead.'}})

    save(PUBLIC / 'cohort_audit.json', {
        'n': len(ids), 'train': len(training), 'validation': len(validation),
        'doctor_excluded': len(doctor_excluded), 'hospital66_moved_to_validation': len(ids66),
        'previous_run': {'n': 900, 'train': 700, 'validation': 180},
        'conversion_failed': summary.get('failed'),
        'split_sha256': hashlib.sha256((ROOT / 'split.json').read_bytes()).hexdigest(),
        'label_rule': 'FEV1/FVC < 70; labels frozen from the audited build summary',
        'review_source': review['source'],
        'case_mix': json.loads((ROOT / 'split_scoring.json').read_text())['case_mix']})
    stage('splits', train=len(training), validation=len(validation),
          excluded=len(doctor_excluded))

    # ---- 4. emphysema density channel ---------------------------------------
    DENSITY.mkdir(exist_ok=True)
    for source in (PREV / 'density', PREV2 / 'density',
                   REG / 'datasets/generated/laa_density_112x136x112'):
        if not source.is_dir():
            continue
        for row in rows:
            target = DENSITY / f"{row['patient_id']}.npy"
            if not target.exists() and (source / target.name).exists():
                os.link(source / target.name, target)
    missing = [r for r in rows if not (DENSITY / f"{r['patient_id']}.npy").exists()]
    save(ROOT / 'missing_density_manifest.json', {'records': missing})
    stage('laa_density', missing=len(missing))

    if missing:
        masks = ROOT / 'masks'
        (masks / 'lung').mkdir(parents=True, exist_ok=True)
        for row in missing:
            target = masks / 'lung' / f"{row['patient_id']}.nii.gz"
            for previous in (PREV / 'masks/lung' / target.name,
                             PREV2 / 'masks/lung' / target.name,
                             REG / 'masks/totalseg/lung' / target.name):
                if not target.exists() and previous.exists():
                    os.link(previous, target)
        execute('lung_masks', [sys.executable, str(REG / 'scripts/segment_lungs_totalseg.py'),
                               '--manifest', str(ROOT / 'missing_density_manifest.json'),
                               '--out', str(masks), '--fast'])

        sys.path.insert(0, str(REG / 'scripts'))
        import nibabel as nib
        import numpy as np
        import precompute_laa_density as laa
        derived = []
        for i, row in enumerate(missing):
            pid = row['patient_id']
            mask_path = masks / 'lung' / f'{pid}.nii.gz'
            ct_image = nib.load(row['path'])
            mask_image = nib.load(str(mask_path))
            if ct_image.shape != mask_image.shape or not np.allclose(
                    ct_image.affine, mask_image.affine, atol=1e-3):
                raise ValueError(f'CT and lung mask grids differ for {pid}')
            result = laa.one((pid, row['path'], str(mask_path),
                              (112, 136, 112), -950., DENSITY, None))
            if not (DENSITY / f'{pid}.npy').exists():
                raise RuntimeError(f'LAA failed for {pid}: {result}')
            derived.append({'patient_id': pid})
            if (i + 1) % 25 == 0 or i + 1 == len(missing):
                stage('laa_density', done=i + 1, total=len(missing))
        save(ROOT / 'new_density_audit.json', {'records': derived})

    absent = [r['patient_id'] for r in rows if not (DENSITY / f"{r['patient_id']}.npy").exists()]
    if absent:
        raise SystemExit(f'{len(absent)} patients still have no density channel: {absent[:10]}')

    # ---- 5. config and snapshot of the training code ------------------------
    import yaml
    config = yaml.safe_load((REG / 'config.rq1.ratio_regression.2ch.yaml').read_text())
    config['data']['laa_density_dir'] = str(DENSITY)
    # Eight workers plus an 11 GB shared cache ran WSL out of memory on the first
    # attempt and the OOM killer took the trainer down during epoch 1. The cache
    # is shared between workers, but each worker's prefetch buffers are not.
    config['data']['num_workers'] = 4
    config['data']['prefetch_factor'] = 2
    (ROOT / 'config.yaml').write_text(yaml.safe_dump(config))
    for source, target in ((REG / 'scripts/train_fixed_holdout_ratio_regression.py', 'train.py'),
                           (Path('/mnt/d/Felix/Hospital/nnMamba/regression/experiments'
                                 '/ratio5_20260915/score_snapshot.py'), 'score.py')):
        code = source.read_text(encoding='utf-8-sig')
        code = code.replace('ROOT = Path(__file__).resolve().parents[1]',
                            f'ROOT = Path({str(REG)!r})')
        (ROOT / target).write_text(code)

    # ---- 6. train -----------------------------------------------------------
    for seed in SEEDS:
        out = ROOT / f'seed{seed}'
        if (out / f'regressor_seed{seed}.pth').exists():
            stage('train', seed=seed, skipped=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        stage('train', seed=seed, training_patients=len(training))
        execute(f'train_seed{seed}', [
            sys.executable, '-u', str(ROOT / 'train.py'),
            '--config', str(ROOT / 'config.yaml'),
            '--source-dir', str(DATA),
            '--split-json', str(ROOT / 'split.json'),
            '--manifest', str(ROOT / 'manifest.json'),
            '--pft-csv', str(ROOT / 'clinical.csv'),
            '--out', str(out), '--epochs', str(EPOCHS),
            '--seed', str(seed), '--skip-holdout'])

    # ---- 7. score -----------------------------------------------------------
    checkpoints = [str(ROOT / f'seed{s}/regressor_seed{s}.pth') for s in SEEDS]
    for n in (5, 3):
        output = ROOT / f'ensemble{n}.json'
        if output.exists():
            continue
        execute(f'evaluate_{n}', [
            sys.executable, '-u', str(ROOT / 'score.py'),
            '--config', str(ROOT / 'config.yaml'),
            '--source-dir', str(DATA),
            '--split-json', str(ROOT / 'split_scoring.json'),
            '--manifest', str(ROOT / 'manifest.json'),
            '--pft-csv', str(ROOT / 'clinical.csv'),
            '--checkpoints', *checkpoints[:n], '--out', str(output)])
        shutil.copy2(output, PUBLIC / output.name)

    save(PUBLIC / 'complete.json', {
        'n': len(ids), 'train': len(training), 'validation': len(validation),
        'doctor_excluded': len(doctor_excluded), 'seeds': SEEDS, 'epochs': EPOCHS})
    stage('complete')
    print('RUN1149_EXIT=0', flush=True)


if __name__ == '__main__':
    main()
