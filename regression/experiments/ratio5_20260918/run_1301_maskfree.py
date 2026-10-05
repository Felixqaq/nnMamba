"""1301-patient mask-free 2-channel five-seed run -- the deployment candidate.

Derived from run_1257_maskfree.py on 2026-09-29: same recipe, same 246 validation
patients, 44 more training patients from the late-May and early-June 2025 pulls
(991 -> 1035). Compared at the end against the 1257-patient run.

Original notes follow.

1257-patient mask-free 2-channel five-seed run -- the deployment candidate.

The recipe chosen on 2026-09-29: two-stage ratio regression, CT plus the
segmentation-free emphysema channel, five seeds. Nothing in this pipeline needs
a segmentation network, which is the point: inference at the hospital must not
need one either.

Validation is held exactly as in every run since 2026-09-18 -- the 180
reviewer-kept patients plus the 66 hospital66 patients -- and the script refuses
to run if it differs, so the scores stay comparable with the 1149-patient
mask-free run. Training grows 883 -> 991.

Resumable: every stage skips what already exists.
"""

from __future__ import annotations

import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path
import shutil
import statistics as st
import subprocess
import sys
import time

import numpy as np

REPO = Path('/home/felix/Research/nnMamba')
REG = REPO / 'regression'
ROOT = REG / 'outputs/ratio5_expanded_20260930_maskfree'
PUBLIC = Path('/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260930_maskfree')
SOURCE = REPO / 'classification/datasets/normal_v_abnormal_fev1fvc70'
PREV = REG / 'outputs/ratio5_expanded_20260929_maskfree'   # 1257-patient mask-free run
PREV_BASE = REG / 'outputs/ratio5_expanded_20260922'
DATA = ROOT / 'ct'
DENSITY = ROOT / 'density_maskfree'
DECISIONS = REG / 'cohort_decisions.local.json'
REVIEW = Path('/mnt/d/Felix/Hospital/nnMamba/regression/experiments/ratio5_20260915'
              '/doctor_review_exclusions.local.json')
HOSPITAL66 = REG / 'datasets/generated/rq1_nva66_manifest.image.json'
SEEDS = [72, 73, 74, 75, 76]
EPOCHS = 80
IMAGE_SIZE = (112, 136, 112)
THRESHOLDS = (-950.0, -910.0)
CUTOFF, BORDER = 70.0, 7.0


def save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')


def stage(name: str, **extra) -> None:
    save(PUBLIC / 'status.json', {'stage': name, 'unix': time.time(), **extra})
    print('[%s] %s %s' % (time.strftime('%H:%M:%S'), name,
                          json.dumps(extra, ensure_ascii=False) if extra else ''), flush=True)


def execute(name: str, command: list[str]) -> None:
    stage(name)
    with (PUBLIC / f'{name}.log').open('a') as fh:
        subprocess.run(command, cwd=REG, stdout=fh, stderr=subprocess.STDOUT, check=True)


def maskfree(job):
    """Fraction of each block below -950 (and -910) HU over the whole volume.

    Identical to build_maskfree_density.py, which built the channel the
    1149-patient run was scored on. The deployed app must compute exactly this.
    """
    pid, path = job
    target = DENSITY / f'{pid}.npy'
    if target.exists():
        return pid, None
    try:
        import nibabel as nib
        from skimage import transform
        ct = np.asanyarray(nib.load(path).dataobj).astype(np.float32)
        if ct.ndim > 3:
            ct = ct[..., 0]
        chans = []
        for hu in THRESHOLDS:
            frac = transform.resize((ct < hu).astype(np.float32), IMAGE_SIZE, order=1,
                                    preserve_range=True, anti_aliasing=True)
            chans.append(np.clip(np.rint(frac * 255.0), 0, 255).astype(np.uint8))
        tmp = target.with_suffix('.tmp.npy')
        np.save(tmp, np.stack(chans, axis=0))
        tmp.rename(target)
        return pid, None
    except Exception as exc:
        return pid, repr(exc)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    if not DECISIONS.is_file():
        raise SystemExit(f'{DECISIONS} is missing; refusing to build without the exclusion list')
    excluded = set(json.loads(DECISIONS.read_text(encoding='utf-8'))['excluded'])

    # ---- 1. snapshot ---------------------------------------------------------
    stage('snapshot_inputs')
    summary = json.loads((SOURCE / 'build_summary.json').read_text(encoding='utf-8-sig'))
    rows = [r for r in summary['records'] if r['ok']]
    ids = {r['patient_id'] for r in rows}
    if len(ids) != len(rows):
        raise SystemExit('duplicate patient ids in the build summary')
    if ids & excluded:
        raise SystemExit(f'excluded patients present in the build: {sorted(ids & excluded)}')
    for row in rows:
        src = Path(row['path'])
        dest = DATA / row['label'] / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(src, dest)
        row['path'] = str(dest)
    summary['records'] = rows
    save(DATA / 'build_summary.json', summary)
    stage('snapshot_inputs', patients=len(ids))

    # ---- 2. clinical table ---------------------------------------------------
    if not (ROOT / 'pft_original.csv').exists():
        shutil.copy2('/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv',
                     ROOT / 'pft_original.csv')
    with (ROOT / 'pft_original.csv').open(encoding='utf-8-sig', newline='') as fh:
        clinical = {r['PatientID']: r for r in
                    ({(k or '').strip(): (v or '').strip() for k, v in row.items()}
                     for row in csv.DictReader(fh)) if r.get('PatientID')}
    with (ROOT / 'clinical.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['PatientID', 'FEV1FVC_pct', 'FEV1FVC_LLN_GLI'])
        writer.writeheader()
        for row in rows:
            original = clinical.get(row['patient_id'], {})
            same = (bool(original.get('FEV1FVC_pct'))
                    and float(original['FEV1FVC_pct']) == float(row['fev1_fvc_pct']))
            writer.writerow({'PatientID': row['patient_id'], 'FEV1FVC_pct': row['fev1_fvc_pct'],
                             'FEV1FVC_LLN_GLI': original.get('FEV1FVC_LLN_GLI', '') if same else ''})

    # ---- 3. splits: validation must be the same 246 as before ----------------
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
    previous = json.loads((PREV_BASE / 'split_scoring.json').read_text())
    if validation != sorted(previous['validation_patient_ids']):
        raise SystemExit('validation set differs from the 1149-patient run; scores would '
                         'not be comparable')
    save(ROOT / 'split.json', {'training_patient_ids': training, 'validation_patient_ids': held,
                               'note': 'covers the whole cohort, as set_fixed_split requires'})
    save(ROOT / 'split_scoring.json', {'training_patient_ids': training,
                                       'validation_patient_ids': validation,
                                       'doctor_excluded_from_validation': sorted(doctor_excluded),
                                       'note': 'scoring only; train.py reads split.json'})
    save(PUBLIC / 'cohort_audit.json', {
        'n': len(ids), 'train': len(training), 'validation': len(validation),
        'doctor_excluded': len(doctor_excluded), 'previous_train': len(previous['training_patient_ids']),
        'channel': 'mask-free: fraction below -950 HU over the whole volume, no segmentation'})
    stage('splits', train=len(training), validation=len(validation), excluded=len(doctor_excluded))

    # ---- 4. mask-free channel -------------------------------------------------
    DENSITY.mkdir(parents=True, exist_ok=True)
    for row in rows:
        target = DENSITY / f"{row['patient_id']}.npy"
        prev = PREV / 'density_maskfree' / target.name
        if not target.exists() and prev.exists():
            os.link(prev, target)
    jobs = [(r['patient_id'], r['path']) for r in rows
            if not (DENSITY / f"{r['patient_id']}.npy").exists()]
    stage('maskfree_channel', to_compute=len(jobs))
    failed = []
    if jobs:
        with Pool(4) as pool:
            for pid, err in pool.imap_unordered(maskfree, jobs):
                if err:
                    failed.append((pid, err))
    if failed:
        raise SystemExit(f'{len(failed)} mask-free channels failed: {failed[:5]}')
    absent = [r['patient_id'] for r in rows if not (DENSITY / f"{r['patient_id']}.npy").exists()]
    if absent:
        raise SystemExit(f'{len(absent)} patients still have no channel: {absent[:5]}')

    # ---- 5. config and code snapshot -------------------------------------------
    import yaml
    config = yaml.safe_load((PREV / 'config.yaml').read_text())
    if int(config['model']['in_channels']) != 2 or config['data'].get('laa_density_mode') != 'density':
        raise SystemExit('the mask-free reference config is not 2-channel density mode')
    config['data']['laa_density_dir'] = str(DENSITY)
    config['data']['num_workers'] = 4
    config['data']['prefetch_factor'] = 2
    (ROOT / 'config.yaml').write_text(
        '# Channel 1 is the fraction below -950 HU over the WHOLE volume, with no lung\n'
        '# mask. The deployed app must compute exactly this; see maskfree() in\n'
        '# run_1257_maskfree.py.\n' + yaml.safe_dump(config))
    for name in ('train.py', 'score.py'):
        shutil.copy2(PREV_BASE / name, ROOT / name)

    # ---- 6. train ---------------------------------------------------------------
    common = ['--source-dir', str(DATA), '--manifest', str(ROOT / 'manifest.json'),
              '--pft-csv', str(ROOT / 'clinical.csv')]
    for seed in SEEDS:
        out = ROOT / f'seed{seed}'
        if (out / f'regressor_seed{seed}.pth').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        stage('train', seed=seed, training_patients=len(training))
        execute(f'train_seed{seed}', [sys.executable, '-u', str(ROOT / 'train.py'),
                                      '--config', str(ROOT / 'config.yaml'),
                                      '--split-json', str(ROOT / 'split.json'), *common,
                                      '--out', str(out), '--epochs', str(EPOCHS),
                                      '--seed', str(seed), '--skip-holdout'])

    # ---- 7. score and compare ---------------------------------------------------
    checkpoints = [str(ROOT / f'seed{s}/regressor_seed{s}.pth') for s in SEEDS]
    output = ROOT / 'ensemble5.json'
    if not output.exists():
        execute('evaluate_5', [sys.executable, '-u', str(ROOT / 'score.py'),
                               '--config', str(ROOT / 'config.yaml'),
                               '--split-json', str(ROOT / 'split_scoring.json'), *common,
                               '--checkpoints', *checkpoints, '--out', str(output)])
    shutil.copy2(output, PUBLIC / output.name)
    for s in SEEDS:
        shutil.copy2(ROOT / f'seed{s}/regressor_seed{s}.pth', PUBLIC / f'regressor_seed{s}.pth')
    shutil.copy2(ROOT / 'config.yaml', PUBLIC / 'config.yaml')
    compare(json.loads((PREV / 'ensemble5.json').read_text()), json.loads(output.read_text()))
    stage('complete')
    print('RUN1301MF_EXIT=0', flush=True)


def compare(before: dict, after: dict) -> None:
    from scipy.stats import wilcoxon
    from sklearn.metrics import roc_auc_score

    hospital66 = {str(r['patient_id']).strip()
                  for r in json.loads(HOSPITAL66.read_text())['records']}
    lines = ['MASK-FREE 5 seeds: 991 training (1257 cohort) -> 1035 training (1301 cohort)', '']
    rng = np.random.default_rng(0)
    for name, keep in (('reviewed 180', lambda p: p not in hospital66),
                       ('all validation', lambda p: True)):
        ids = sorted(p for p in after['patients'] if keep(p))
        t = np.array([after['patients'][p]['true_ratio'] for p in ids])
        if not np.allclose(t, [before['patients'][p]['true_ratio'] for p in ids]):
            lines.append('%s: validation labels differ between runs; not compared' % name)
            continue
        p0 = np.array([before['patients'][p]['mean_predicted_ratio'] for p in ids])
        p1 = np.array([after['patients'][p]['mean_predicted_ratio'] for p in ids])
        y = (t < CUTOFF).astype(int)
        b = np.abs(t - CUTOFF) < BORDER

        def bal(pred):
            yh = (pred < CUTOFF).astype(int)
            return ((yh[y == 1] == 1).mean() + (yh[y == 0] == 0).mean()) / 2

        e0, e1 = np.abs(p0 - t), np.abs(p1 - t)
        boot = rng.integers(0, len(ids), size=(5000, len(ids)))
        dmae = np.percentile(e1[boot].mean(1) - e0[boot].mean(1), [2.5, 97.5])
        lines.append('%s (n=%d)' % (name.upper(), len(ids)))
        lines.append('  AUC        %.4f -> %.4f' % (roc_auc_score(y, -p0), roc_auc_score(y, -p1)))
        lines.append('  border AUC %.4f -> %.4f  (n=%d)'
                     % (roc_auc_score(y[b], -p0[b]), roc_auc_score(y[b], -p1[b]), b.sum()))
        lines.append('  balacc     %.4f -> %.4f' % (bal(p0), bal(p1)))
        lines.append('  MAE        %.3f -> %.3f  (diff CI [%+.3f, %+.3f], Wilcoxon p=%.3f)'
                     % (e0.mean(), e1.mean(), *dmae, wilcoxon(e1, e0).pvalue))
        lines.append('')

    def members(blob):
        return [(v['seed'], v['fixed70']['auc'], v['ratio_mae']) for v in blob['per_member'].values()]
    m0, m1 = members(before), members(after)
    lines.append('member AUC %.4f +- %.4f -> %.4f +- %.4f | member MAE %.3f -> %.3f' % (
        st.mean(a for _, a, _ in m0), st.stdev(a for _, a, _ in m0),
        st.mean(a for _, a, _ in m1), st.stdev(a for _, a, _ in m1),
        st.mean(e for _, _, e in m0), st.mean(e for _, _, e in m1)))
    text = '\n'.join(lines)
    print('\n' + text, flush=True)
    (PUBLIC / 'comparison_1257_vs_1301.txt').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
