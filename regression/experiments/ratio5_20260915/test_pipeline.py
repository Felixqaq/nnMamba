"""Smoke-test real CT+LAA training and voting without touching official200."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path('/home/felix/Research/nnMamba/regression/outputs/ratio5_expanded_20260915')
REG = ROOT.parents[1]
SMOKE = ROOT / 'smoke_not_for_performance'
SMOKE.mkdir(exist_ok=True)
old = json.loads((REG.parent / 'classification/datasets/normal_v_abnormal_fev1fvc70/build_summary.json').read_text())
hold = set(json.loads((ROOT / 'official200_original.json').read_text())['validation_patient_ids'])
groups = {name: [r for r in old['records'] if r['ok'] and r['label'] == name
                 and r['patient_id'] not in hold
                 and (REG / 'datasets/generated/laa_density_112x136x112' / (r['patient_id']+'.npy')).exists()][:6]
          for name in ['Normal','Abnormal']}
train, test = [], []
with (SMOKE / 'clinical.csv').open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(['PatientID','FEV1FVC_pct','FEV1FVC_LLN_GLI'])
    for label, rows in groups.items():
        assert len(rows) == 6
        for i, row in enumerate(rows):
            src = Path(row['path'])
            target = SMOKE / 'ct' / label / src.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(src, target)
            (train if i<4 else test).append(row['patient_id'])
            writer.writerow([row['patient_id'], row['fev1_fvc_pct'], ''])
(SMOKE / 'split.json').write_text(json.dumps({'training_patient_ids':train,'validation_patient_ids':test}))
cfg = yaml.safe_load((REG / 'config.rq1.ratio_regression.2ch.yaml').read_text())
cfg['data'].update(num_workers=0, laa_density_dir=str(REG / 'datasets/generated/laa_density_112x136x112'))
(SMOKE / 'config.yaml').write_text(yaml.safe_dump(cfg))
common = ['--config',str(SMOKE/'config.yaml'),'--source-dir',str(SMOKE/'ct'),
          '--split-json',str(SMOKE/'split.json'),'--pft-csv',str(SMOKE/'clinical.csv'),
          '--manifest',str(SMOKE/'manifest.json')]
subprocess.run([sys.executable,str(ROOT/'train.py'),*common,'--out',str(SMOKE/'model'),
                '--epochs','1','--seed','72','--skip-holdout'],cwd=REG,check=True)
subprocess.run([sys.executable,str(ROOT/'score.py'),*common,'--checkpoints',
                str(SMOKE/'model/regressor_seed72.pth'),'--out',str(SMOKE/'score.json')],cwd=REG,check=True)
result=json.loads((SMOKE/'score.json').read_text())
assert result['fixed70']['mean_ratio']['n'] == 4
print('PASS real CT+LAA train and score; no official200 patients used; smoke metrics not clinical results')
