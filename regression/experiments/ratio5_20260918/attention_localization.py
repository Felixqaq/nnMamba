"""Does attention pooling actually look at disease? Measured, not eyeballed.

For every validation patient, average the five members' attention maps per
stage, then compare where the weight goes against arrays already on disk:

  lung        lung occupancy per block (channel 1 of the masked density run)
  emphysema   fraction of each block below -950 HU inside the lung (channel 0)
  stray air   below -950 HU but outside the lung: outside the body, trachea,
              bowel gas (mask-free channel 0 minus masked channel 0)

Each is reported against the uniform map, which is what average pooling does,
so a number means something only relative to that reference. The lung masks
serve this evaluation only; the model under test never sees them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

REG = Path('/home/felix/Research/nnMamba/regression')
sys.path.insert(0, str(REG))
from core.checkpoints import load_model_weights  # noqa: E402
from core.config import Config  # noqa: E402
from core.runtime import configure_torch_runtime  # noqa: E402
from data.loader import RegressionLoaderHelper as LoaderHelper  # noqa: E402
from models import build_model  # noqa: E402

MASKED_DENSITY = (REG / 'outputs/ratio5_expanded_20260929_masked/density',
                  REG / 'outputs/ratio5_expanded_20260922/density')
CUTOFF = 70.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--split-json', type=Path, required=True)
    p.add_argument('--maskfree-dir', type=Path, required=True)
    p.add_argument('--clinical', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    return p.parse_args()


def masked_array(pid):
    for d in MASKED_DENSITY:
        f = d / f'{pid}.npy'
        if f.exists():
            return np.load(f).astype(np.float32) / 255.0
    raise SystemExit(f'{pid}: no masked density array in {MASKED_DENSITY}')


def to_grid(volume, shape):
    t = torch.from_numpy(volume)[None, None]
    return F.adaptive_avg_pool3d(t, shape)[0, 0].numpy()


def main():
    args = parse_args()
    import csv
    truth = {r['PatientID']: float(r['FEV1FVC_pct'])
             for r in csv.DictReader(args.clinical.open(encoding='utf-8-sig'))}
    pids = json.loads(args.split_json.read_text())['validation_patient_ids']

    config = Config.from_yaml(str(args.config))
    if str(config.model.name).lower() != 'hybrid_mamba_attnpool':
        raise SystemExit('this analysis needs an attention-pooling model (hybrid_mamba_attnpool)')
    config = replace(config, data=replace(config.data, source_dir=str(args.source_dir),
                                          manifest=str(args.manifest), cache_data=False))
    configure_torch_runtime()
    helper = LoaderHelper(config)
    index = {p: i for i, p in enumerate(helper.patient_ids)}
    loader = helper._build_loader([index[p] for p in pids], batch_size=4, shuffle=False,
                                  drop_last=False, augmentation=None)
    device = torch.device('cuda')

    maps: dict[str, list[np.ndarray]] = {}
    for path in args.checkpoints:
        blob = torch.load(path, map_location='cpu', weights_only=False)
        model = build_model(config.model, output_dim=config.model_output_dim()).to(device)
        load_model_weights(model, blob['state_dict'], str(path))
        model.eval()
        for batch in loader:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                _, weights = model.forward_with_attention(batch['ct'].to(device))
            for i, pid in enumerate(batch['patient_id']):
                per_stage = [w[i].float().cpu().numpy() for w in weights]
                if pid not in maps:
                    maps[pid] = [np.zeros_like(a) for a in per_stage]
                for acc, a in zip(maps[pid], per_stage):
                    acc += a / len(args.checkpoints)

    stages = ('stage1', 'stage2', 'stage3', 'attn')
    rows = []
    for pid in pids:
        masked = masked_array(pid)
        free = np.load(args.maskfree_dir / f'{pid}.npy').astype(np.float32) / 255.0
        stray = np.clip(free[0] - masked[0], 0.0, None)
        rec = {'patient_id': pid, 'abnormal': truth[pid] < CUTOFF,
               'border': abs(truth[pid] - CUTOFF) < 7}
        for name, w in zip(stages, maps[pid]):
            occ, laa, air = (to_grid(v, w.shape) for v in (masked[1], masked[0], stray))
            uniform = np.full_like(w, 1.0 / w.size)
            for tag, weights in (('attn', w), ('uniform', uniform)):
                in_lung = float((weights * occ).sum())
                laa_in_attended = float((weights * laa).sum()) / max(in_lung, 1e-8)
                rec[f'{name}_{tag}_in_lung'] = in_lung
                rec[f'{name}_{tag}_stray_air'] = float((weights * air).sum())
                rec[f'{name}_{tag}_laa'] = laa_in_attended
            lung_laa = float(laa.sum()) / max(float(occ.sum()), 1e-8)
            rec[f'{name}_laa_enrichment'] = rec[f'{name}_attn_laa'] / max(lung_laa, 1e-8)
            rec[f'{name}_entropy_ratio'] = float(-(w * np.log(w + 1e-12)).sum() / np.log(w.size))
        rows.append(rec)

    lines = ['WHERE DOES ATTENTION GO? validation n=%d, maps averaged over %d members'
             % (len(rows), len(args.checkpoints)),
             'uniform = what average pooling does. enrichment > 1 = attention favours',
             'emphysematous lung over the patient\'s own lung average. entropy 1 = uniform.', '']
    groups = (('abnormal', lambda r: r['abnormal']), ('normal', lambda r: not r['abnormal']))
    for name in stages:
        lines.append(name.upper())
        lines.append('  %-9s %-17s %-17s %-12s %s'
                     % ('group', 'in lung attn/unif', 'stray air a/u', 'LAA enrich', 'entropy'))
        for gname, keep in groups:
            g = [r for r in rows if keep(r)]
            med = lambda k: float(np.median([r[k] for r in g]))
            lines.append('  %-9s %.3f / %.3f     %.4f / %.4f   %.2f         %.3f' % (
                f'{gname}({len(g)})', med(f'{name}_attn_in_lung'), med(f'{name}_uniform_in_lung'),
                med(f'{name}_attn_stray_air'), med(f'{name}_uniform_stray_air'),
                med(f'{name}_laa_enrichment'), med(f'{name}_entropy_ratio')))
        lines.append('')
    text = '\n'.join(lines)
    print(text, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'attention_localization.txt').write_text(text, encoding='utf-8')
    (args.out / 'attention_localization.json').write_text(json.dumps(rows, indent=1))
    np.savez_compressed(args.out / 'attention_maps_validation.npz',
                        **{f'{pid}__{s}': m.astype(np.float16)
                           for pid, ms in maps.items() for s, m in zip(stages, ms)})


if __name__ == '__main__':
    main()
