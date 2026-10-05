#!/usr/bin/env python
"""Two-stage FEV1/FVC regression with guided attention pooling.

Everything is the ratio trainer (train_fixed_holdout_ratio_regression.py) --
splits, standardisation, scoring, checkpoint format -- except the training loop,
which adds two terms on the attention maps of an attention-pooling model:

  lung prior    -log(attention mass inside the lung), averaged over stages. It
                asks only that the weight fall inside the lung, not where inside,
                so the model stays free to focus within it.
  entropy       max(0, H - target) on stages 1 and 2, H the normalised entropy
                of the map (1 = uniform). It penalises maps that stay too
                uniform without pushing them to collapse onto a few voxels.

Why: on 2026-09-29 unguided attention pooling finished with normalised entropy
0.999, 10% less weight on the lung than average pooling and 5% more on soft
tissue and bone. The regression loss alone gave the scorer no consistent
direction; these terms supply one.

The lung masks are used here, in training, only. The model sees the same two
input channels as always, and inference needs no segmentation.

The mask travels through augmentation as a third channel of the batch: the
spatial transform samples every channel on one shared grid, and intensity
augmentation touches channel 0 only, so the mask moves with the CT and keeps its
values. It is split off before the model sees the batch.

The original trainer is imported, not edited: its train_regressor is replaced
for this process only, and its main() runs unchanged.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_fixed_holdout_ratio_regression as base  # noqa: E402

GUIDE: dict = {}
ENTROPY_STAGES = (0, 1)          # 28x34x28 and 14x17x14: where localisation lives


def parse_guidance_args() -> list[str]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--lung-occupancy-dir", type=Path, action="append", required=True,
                   help="directories of (2, D, H, W) uint8 arrays whose channel 1 is "
                        "lung occupancy; searched in order")
    p.add_argument("--lung-weight", type=float, default=0.1)
    p.add_argument("--entropy-weight", type=float, default=0.1)
    p.add_argument("--entropy-target", type=float, default=0.8,
                   help="normalised entropy above which maps are penalised. Uniform "
                        "over a 17%% lung is 0.83 at stage 1, so 0.8 asks for a "
                        "little focus inside the lung, no more")
    known, rest = p.parse_known_args()
    GUIDE.update(dirs=known.lung_occupancy_dir, lung_weight=known.lung_weight,
                 entropy_weight=known.entropy_weight, entropy_target=known.entropy_target)
    return rest


class Occupancy:
    """Lung occupancy per patient on the model grid, cached as uint8."""

    def __init__(self, dirs, shape):
        self.dirs, self.shape, self.cache = list(dirs), tuple(shape), {}

    def _load(self, pid):
        for d in self.dirs:
            f = Path(d) / f"{pid}.npy"
            if f.exists():
                a = np.load(f)
                occ = a[1] if a.ndim == 4 else a
                if occ.shape != self.shape or occ.dtype != np.uint8:
                    raise SystemExit(f"{f}: expected uint8 {self.shape}, got {occ.dtype} {occ.shape}")
                return occ
        raise SystemExit(f"{pid}: no lung occupancy in {self.dirs}")

    def batch(self, pids, device):
        for p in pids:
            if p not in self.cache:
                self.cache[p] = self._load(p)
        arr = np.stack([self.cache[p] for p in pids])[:, None]
        return torch.from_numpy(arr).to(device, non_blocking=True).float() / 255.0


def guidance_losses(weights, occupancy):
    """Lung prior and entropy hinge for one batch of attention maps."""
    lung_terms, entropy_terms, stats = [], [], {}
    for s, w in enumerate(weights):
        w = w.float()
        occ = F.adaptive_avg_pool3d(occupancy, w.shape[-3:])[:, 0]
        mass = (w * occ).flatten(1).sum(1).clamp_min(1e-6)
        lung_terms.append(-torch.log(mass).mean())
        h = -(w * torch.log(w.clamp_min(1e-12))).flatten(1).sum(1) / math.log(w[0].numel())
        if s in ENTROPY_STAGES:
            entropy_terms.append(F.relu(h - GUIDE["entropy_target"]).mean())
        stats[f"mass{s + 1}"] = float(mass.mean())
        stats[f"H{s + 1}"] = float(h.mean())
    return torch.stack(lung_terms).mean(), torch.stack(entropy_terms).mean(), stats


def guided_train_regressor(config, helper, clinical, seed, epochs, device, mean, std,
                           eval_hook=None, eval_every=0, aux=None):
    if aux is not None:
        raise SystemExit("--aux-pft-csv is not supported together with guided attention")
    base.set_seed(seed)
    model = base.build_model(config.model, output_dim=config.model_output_dim()).to(device)
    if not hasattr(model, "forward_with_attention"):
        raise SystemExit(f"{config.model.name} exposes no attention maps; use an "
                         "attention-pooling model such as hybrid_mamba_attnpool_guided")
    occupancy = Occupancy(GUIDE["dirs"], tuple(int(v) for v in config.data.image_size))
    train_loader = helper.get_train_dl(0, shuffle=True)
    optimizer = base.build_optimizer(model, float(config.training.learning_rate),
                                     float(config.training.weight_decay))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    augmentation = getattr(getattr(train_loader, "dataset", None), "augmentation", None)
    if augmentation is not None and getattr(augmentation, "enabled", False) \
            and not getattr(augmentation, "defer_to_device", False):
        # CPU-side augmentation would move the CT in the worker, before the mask
        # is attached, and the prior would then point at the wrong voxels.
        raise SystemExit("guided attention needs augmentation deferred to the GPU "
                         "(defer_to_device); CPU augmentation would misalign the lung mask")
    if not getattr(augmentation, "defer_to_device", False):
        augmentation = None
    lw, ew = GUIDE["lung_weight"], GUIDE["entropy_weight"]
    print(f"guided attention: lung weight {lw}, entropy weight {ew} above "
          f"{GUIDE['entropy_target']}, occupancy from {[str(d) for d in GUIDE['dirs']]}",
          flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        sums = {"loss": 0.0, "ratio": 0.0, "lung": 0.0, "entropy": 0.0}
        track, batches = {}, 0
        for batch in tqdm(train_loader, leave=False, desc=f"seed {seed} epoch {epoch}/{epochs}"):
            ct = batch["ct"].to(device, non_blocking=True)
            occ = occupancy.batch(batch["patient_id"], device)
            if augmentation is not None:
                both = augmentation.apply_batch(torch.cat([ct.float(), occ], dim=1),
                                                batch.get("augment"))
                ct, occ = both[:, :-1], both[:, -1:].clamp(0.0, 1.0)
            raw = torch.tensor([base.ratio_of(clinical, pid) for pid in batch["patient_id"]],
                               dtype=torch.float32, device=device)
            target = (raw - mean) / std
            optimizer.zero_grad(set_to_none=True)

            def compute(use_amp: bool):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_amp):
                    out, weights = model.forward_with_attention(ct)
                    main = loss_fn(out.view(-1).float(), target)
                lung, ent, stats = guidance_losses(weights, occ)
                return main + lw * lung + ew * ent, main, lung, ent, stats

            use_amp = bool(config.training.amp and device.type == "cuda")
            loss, main, lung, ent, stats = compute(use_amp)
            if not torch.isfinite(loss) and use_amp:
                print("Non-finite loss under bf16; retrying this batch in fp32", flush=True)
                loss, main, lung, ent, stats = compute(False)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss after fp32 retry")
            loss.backward()
            clip = float(config.training.clip_grad_norm)
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            for k, v in (("loss", loss), ("ratio", main), ("lung", lung), ("entropy", ent)):
                sums[k] += float(v.item())
            for k, v in stats.items():
                track[k] = track.get(k, 0.0) + v
            batches += 1
        scheduler.step()
        n = max(batches, 1)
        print(f"seed={seed} epoch={epoch}/{epochs} loss={sums['loss']/n:.6f} "
              f"ratio={sums['ratio']/n:.6f} lung={sums['lung']/n:.4f} "
              f"entropy={sums['entropy']/n:.4f} "
              f"lung_mass s1={track['mass1']/n:.3f} s2={track['mass2']/n:.3f} "
              f"H s1={track['H1']/n:.3f} s2={track['H2']/n:.3f} "
              f"lr={scheduler.get_last_lr()[0]:.8g}", flush=True)
        if eval_hook is not None and eval_every > 0 and (epoch % eval_every == 0 or epoch == epochs):
            model.eval()
            eval_hook(epoch, model, sums["loss"] / n)
            model.train()
    model.eval()
    return model


def main() -> None:
    sys.argv = [sys.argv[0]] + parse_guidance_args()
    base.train_regressor = guided_train_regressor
    base.main()


if __name__ == "__main__":
    main()
