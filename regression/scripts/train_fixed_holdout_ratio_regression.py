#!/usr/bin/env python3
"""Two-stage FEV1/FVC: regress the ratio end to end, then apply the clinical cutoff.

Classifying at 70 makes a patient at 69 and one at 71 maximally different while
their lungs are not, and every classifier tried on this cohort sits at chance in
that band. Regressing the ratio removes the discontinuity from the training
signal: 69 and 71 are two apart, and the cutoff is applied only afterwards.

That separation buys something the classifiers cannot have. The decision rule is
supplied by the clinical definition rather than fitted, so the same trained model
can be read out under the GOLD fixed ratio of 70 and under each patient's own GLI
lower limit of normal, with nothing refitted between them. This matters here: on
this cohort a logistic regression on age, sex and height alone reaches holdout
AUC 0.630 against the fixed ratio -- essentially matching the 3D classifier --
but only 0.560 against GLI, because GLI already adjusts for those variables. The
fixed-ratio label is partly an age label, and a single regression model lets both
readings be compared on identical predictions.

Selection discipline: the model never sees the holdout during training. AUC is
threshold-free and is the number to compare across runs. Cutoffs derived from
training predictions are in-sample and are labelled as such in the output; the
label-rule readings (70, and each patient's GLI LLN) are fitted to nothing at all
and are the honest operating points.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import Config  # noqa: E402
from core.runtime import configure_torch_runtime  # noqa: E402
from data.loader import RegressionLoaderHelper as LoaderHelper  # noqa: E402
from models import build_model  # noqa: E402
from networks.hybrid_mamba_attention_regressor import AUX_PFT_TARGETS  # noqa: E402

FIXED_CUTOFF = 70.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "datasets/generated/doctor_validation_manifest.json")
    p.add_argument("--pft-csv", type=Path, required=True)
    p.add_argument("--aux-pft-csv", type=Path,
                   help="the full PFT export. Supplying it turns on the auxiliary "
                        "spirometry heads: the trunk predicts FEV1, FVC and both "
                        "%%predicted values alongside the ratio. The point is "
                        "regularisation -- in-sample ratio MAE runs near 1.1 "
                        "against 7.1 held out, so the trunk has room to memorise a "
                        "single target. Omit it and the run is single-target, "
                        "exactly as before.")
    p.add_argument("--aux-weight", type=float, default=0.25,
                   help="weight on each auxiliary target, against 1.0 on the ratio")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=72)
    p.add_argument("--margin", type=float, default=7.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--internal-val-size", type=int, default=0,
                   help="hold this many training patients out of training and score "
                        "them every --eval-every epochs. Training loss on this task "
                        "keeps falling to epoch 80 without a plateau, so the epoch "
                        "budget cannot be read off it; this measures generalisation "
                        "directly instead, and never touches the frozen holdout.")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--internal-val-seed", type=int, default=20260904,
                   help="kept separate from --seed so every arm carves the same "
                        "internal validation patients and the arms stay comparable")
    p.add_argument("--skip-holdout", action="store_true",
                   help="do not score the frozen holdout at all. Use it while "
                        "choosing epochs or regularisation: that holdout has been "
                        "scored many times already and every extra look inflates it.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_clinical(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = {(f or "").strip() for f in (reader.fieldnames or [])}
        for required in ("PatientID", "FEV1FVC_pct"):
            if required not in fields:
                raise SystemExit(f"{path}: no {required!r} column")
        for row in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            rows[row["PatientID"]] = row
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def ratio_of(clinical: dict, pid: str) -> float:
    row = clinical.get(pid)
    if row is None:
        raise SystemExit(f"{pid}: no row in the PFT csv")
    value = row.get("FEV1FVC_pct", "")
    if not value:
        raise SystemExit(f"{pid}: FEV1FVC_pct is empty")
    return float(value)


# The ratio label is Pre for some patients and Post for others -- the export
# says which in its Source column -- so the volumes have to be taken from the
# matching half of the test. The %predicted columns already follow Source.
AUX_SOURCES = {
    "fev1": ("FEV1_pre", "FEV1_post"),
    "fvc": ("FVC_pre", "FVC_post"),
    "fev1_pctpred": ("FEV1_pctpred", "FEV1_pctpred"),
    "fvc_pctpred": ("FVC_pctpred", "FVC_pctpred"),
}
# How far the volumes may disagree with the ratio label before the row is taken
# to describe a different session. Labels are stored to whole percent, so ~0.5
# of disagreement is arithmetic; 1.5 leaves room for that and still catches the
# real case (one patient whose CT and PFT were captured months apart, kept in
# the cohort under the ratio the reviewer chose, whose volumes are the other
# session's and would be a wrong target).
AUX_RATIO_TOLERANCE = 1.5


def load_aux_targets(path: Path, order: tuple) -> tuple[dict, dict]:
    """Per-patient spirometry targets from the full export, plus a reject log."""
    rows: dict[str, dict[str, str]] = {}
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if row.get("PatientID"):
                rows[row["PatientID"]] = row
    if not rows:
        raise SystemExit(f"{path}: no rows")
    missing = [c for name in order for c in set(AUX_SOURCES[name])
               if c not in next(iter(rows.values()))]
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(set(missing))}")
    return rows, {}


def aux_of(aux_rows: dict, clinical: dict, pid: str, order: tuple):
    """The auxiliary targets for one patient, or None with the reason why not.

    Returning a reason rather than a silent zero matters: an unnoticed gap here
    would train the heads towards a value nobody measured.
    """
    row = aux_rows.get(pid)
    if row is None:
        return None, "not in the auxiliary export"
    post = row.get("Source", "") == "Post"
    values = []
    for name in order:
        column = AUX_SOURCES[name][1 if post else 0]
        raw = row.get(column, "")
        if not raw:
            return None, f"{column} empty"
        try:
            values.append(float(raw))
        except ValueError:
            return None, f"{column} not a number ({raw!r})"
    fev1, fvc = values[0], values[1]
    if fvc <= 0:
        return None, "FVC is not positive"
    drift = abs(100.0 * fev1 / fvc - ratio_of(clinical, pid))
    if drift > AUX_RATIO_TOLERANCE:
        return None, f"volumes imply ratio {100.0 * fev1 / fvc:.1f}, label says " \
                     f"{ratio_of(clinical, pid):.1f}"
    return values, ""


def gli_cutoff_of(clinical: dict, pid: str) -> float:
    """Each patient's own GLI lower limit of normal, in ratio points."""
    row = clinical[pid]
    value = (row.get("FEV1FVC_LLN_GLI") or "").strip()
    if not value:
        raise SystemExit(f"{pid}: no FEV1FVC_LLN_GLI, cannot apply the GLI rule")
    return float(value)


def set_fixed_split(helper: LoaderHelper, split: dict) -> tuple[list[int], list[int]]:
    index_by_id = {pid: index for index, pid in enumerate(helper.patient_ids)}
    if len(index_by_id) != len(helper.patient_ids):
        raise SystemExit("dataset contains duplicate patient IDs")
    train_ids = list(split["training_patient_ids"])
    holdout_ids = list(split["validation_patient_ids"])
    if set(train_ids) & set(holdout_ids):
        raise SystemExit("train and holdout IDs overlap")
    expected = set(train_ids) | set(holdout_ids)
    missing = sorted(expected - set(index_by_id))
    extra = sorted(set(index_by_id) - expected)
    if missing or extra:
        raise SystemExit(f"split/dataset mismatch: missing={missing} extra={extra}")
    train_indices = [index_by_id[pid] for pid in train_ids]
    holdout_indices = [index_by_id[pid] for pid in holdout_ids]
    helper.fold_indices = [(train_indices, holdout_indices)]
    helper.k_folds = 1
    return train_indices, holdout_indices


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def stratified_carve(ids, clinical, size, seed):
    """Take `size` patients out of the training set, keeping the class ratio.

    The carve is driven by its own seed so that every arm of a comparison holds
    out the same people; if it followed the model seed, two arms would be scored
    on different patients and the difference between them would be unreadable.
    """
    if size <= 0:
        return list(ids), []
    if size >= len(ids):
        raise SystemExit(f"--internal-val-size {size} leaves no training patients")
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[str]] = {0: [], 1: []}
    for pid in ids:
        by_class[int(ratio_of(clinical, pid) < FIXED_CUTOFF)].append(pid)
    held: list[str] = []
    for label, members in by_class.items():
        take = int(round(size * len(members) / len(ids)))
        order = rng.permutation(len(members))
        held += [members[i] for i in order[:take]]
    held_set = set(held)
    return [p for p in ids if p not in held_set], sorted(held_set)


def train_regressor(config, helper, clinical, seed, epochs, device, mean, std,
                    eval_hook=None, eval_every=0, aux=None):
    set_seed(seed)
    model = build_model(config.model, output_dim=config.model_output_dim()).to(device)
    if aux is not None and not hasattr(model, "forward_multitarget"):
        raise SystemExit(f"{config.model.name} has no forward_multitarget; it cannot "
                         "train the auxiliary spirometry heads. Drop --aux-pft-csv or "
                         "use a model that implements them.")
    train_loader = helper.get_train_dl(0, shuffle=True)
    optimizer = build_optimizer(model, float(config.training.learning_rate),
                                float(config.training.weight_decay))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    # Huber rather than plain squared error: a handful of ratios sit far from the
    # mean and squared error would let them dominate the gradient.
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    augmentation = getattr(getattr(train_loader, "dataset", None), "augmentation", None)
    if not getattr(augmentation, "defer_to_device", False):
        augmentation = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_main, total_aux, batches = 0.0, 0.0, 0.0, 0
        for batch in tqdm(train_loader, leave=False,
                          desc=f"seed {seed} epoch {epoch}/{epochs}"):
            ct = batch["ct"].to(device, non_blocking=True)
            if augmentation is not None:
                ct = augmentation.apply_batch(ct, batch.get("augment"))
            raw = torch.tensor([ratio_of(clinical, pid) for pid in batch["patient_id"]],
                               dtype=torch.float32, device=device)
            target = (raw - mean) / std
            if aux is not None:
                aux_target, aux_mask = aux.batch(batch["patient_id"], device)
            optimizer.zero_grad(set_to_none=True)

            def compute_loss(use_amp: bool):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_amp):
                    if aux is None:
                        out = model(ct)
                        main = loss_fn(out.view(-1), target)
                        return main, main, main.new_zeros(())
                    out, aux_out = model.forward_multitarget(ct)
                    main = loss_fn(out.view(-1), target)
                    # Averaged over the patients that have each value, not over
                    # the batch: patients without spirometry must not pull the
                    # heads towards zero.
                    penalty = main.new_zeros(())
                    for i, name in enumerate(aux.order):
                        keep = aux_mask[:, i]
                        if not bool(keep.any()):
                            continue
                        per = nn.functional.smooth_l1_loss(
                            aux_out[name].view(-1)[keep], aux_target[keep, i],
                            beta=1.0, reduction="mean")
                        penalty = penalty + per
                    penalty = penalty / max(len(aux.order), 1)
                    return main + aux.weight * penalty, main, penalty

            use_amp = bool(config.training.amp and device.type == "cuda")
            loss, main_loss, aux_loss = compute_loss(use_amp)
            if not torch.isfinite(loss) and use_amp:
                print("Non-finite loss under bf16; retrying this batch in fp32", flush=True)
                loss, main_loss, aux_loss = compute_loss(False)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss after fp32 retry")
            scaler.scale(loss).backward()
            clip = float(config.training.clip_grad_norm)
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            total_main += float(main_loss.item())
            total_aux += float(aux_loss.item())
            batches += 1
        scheduler.step()
        # The combined loss is not comparable with a single-target run's, so the
        # ratio term is printed on its own beside it.
        detail = ("" if aux is None else
                  f" ratio={total_main/max(batches,1):.6f} "
                  f"aux={total_aux/max(batches,1):.6f}")
        print(f"seed={seed} epoch={epoch}/{epochs} loss={total_loss/max(batches,1):.6f}"
              f"{detail} lr={scheduler.get_last_lr()[0]:.8g}", flush=True)
        if eval_hook is not None and eval_every > 0 and (
                epoch % eval_every == 0 or epoch == epochs):
            model.eval()
            eval_hook(epoch, model, total_loss / max(batches, 1))
            model.train()
    model.eval()
    return model


class AuxTargets:
    """Standardised auxiliary targets, with a per-patient mask.

    Each target is standardised on the training patients that actually have it,
    so the four losses are on one scale and none of them dominates by unit.
    """

    def __init__(self, values: dict, order: tuple, train_ids: list, weight: float):
        self.order = order
        self.weight = float(weight)
        self.values = values
        self.stats = []
        for i, name in enumerate(order):
            column = np.array([values[p][i] for p in train_ids if values.get(p) is not None],
                              dtype=float)
            if column.size < 2:
                raise SystemExit(f"only {column.size} training patients have {name}; "
                                 "refusing to standardise on that")
            self.stats.append((float(column.mean()), float(column.std() or 1.0)))
        self._cache: dict[str, tuple] = {}

    def _row(self, pid: str):
        if pid not in self._cache:
            raw = self.values.get(pid)
            if raw is None:
                self._cache[pid] = ([0.0] * len(self.order), [False] * len(self.order))
            else:
                self._cache[pid] = (
                    [(raw[i] - m) / s for i, (m, s) in enumerate(self.stats)],
                    [True] * len(self.order))
        return self._cache[pid]

    def batch(self, patient_ids, device):
        rows = [self._row(p) for p in patient_ids]
        target = torch.tensor([r[0] for r in rows], dtype=torch.float32, device=device)
        mask = torch.tensor([r[1] for r in rows], dtype=torch.bool, device=device)
        return target, mask


def inference_loader(helper: LoaderHelper, indices: list[int]):
    """A clean pass over the given patients: no augmentation, nothing dropped.

    get_train_dl is built for training -- it applies augmentation, repeats
    patients when the augmentation multiplier is above one, and sets
    drop_last=True. Any of those would corrupt an in-sample readout, so the
    inference pass goes through the plain loader instead.
    """
    return helper._build_loader(
        list(indices),
        batch_size=int(helper.val_batch_size or helper.batch_size),
        shuffle=False,
        drop_last=False,
        augmentation=None,
    )


def predict(model, loader, device, use_amp, mean, std) -> dict[str, float]:
    out: dict[str, float] = {}
    for batch in loader:
        ct = batch["ct"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=bool(use_amp and device.type == "cuda")):
            pred = model(ct).float().view(-1)
        pred = pred.cpu().numpy() * std + mean
        for pid, value in zip(batch["patient_id"], pred):
            out[pid] = float(value)
    return out


def score_block(y, predicted_ratio, cutoff_per_patient, dist, margin, label):
    """Score one decision rule. The score is the negated ratio: lower ratio = more obstructed."""
    score = -np.asarray(predicted_ratio, dtype=float)
    pred = (np.asarray(predicted_ratio) < np.asarray(cutoff_per_patient)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    block = {
        "rule": label,
        "n": int(len(y)),
        "n_abnormal": int(y.sum()),
        "auc": round(float(roc_auc_score(y, score)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "sensitivity": round(float(tp / (tp + fn)) if tp + fn else float("nan"), 4),
        "specificity": round(float(tn / (tn + fp)) if tn + fp else float("nan"), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    bands = {}
    for name, sel in (("borderline", dist < margin), ("clear", dist >= margin)):
        if len(set(y[sel].tolist())) >= 2:
            bands[name] = {"n": int(sel.sum()),
                           "auc": round(float(roc_auc_score(y[sel], score[sel])), 4)}
    block["by_band"] = bands
    return block


def boot_ci(y, score, n=4000, seed=11):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    if not vals:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def calibrate_offset(train_pred, train_ratio, cutoffs):
    """Shift the cutoff so training balanced accuracy peaks; in-sample, so optimistic."""
    y = (np.asarray(train_ratio) < np.asarray(cutoffs)).astype(int)
    best, best_delta = -1.0, 0.0
    for delta in np.arange(-15.0, 15.01, 0.25):
        pred = (np.asarray(train_pred) < np.asarray(cutoffs) + delta).astype(int)
        s = balanced_accuracy_score(y, pred)
        if s > best:
            best, best_delta = s, float(delta)
    return best_delta, round(float(best), 4)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config = Config.from_yaml(str(args.config))
    if int(config.model.num_classes) != 1:
        raise SystemExit(f"{args.config}: regression needs model.num_classes: 1, "
                         f"got {config.model.num_classes}")
    # Same route the classification runner takes: point the config at this
    # cohort rather than passing paths to the helper, which does not accept them.
    config = replace(config, data=replace(config.data,
                                          source_dir=str(args.source_dir),
                                          manifest=str(args.manifest)))
    configure_torch_runtime()

    clinical = load_clinical(args.pft_csv)
    split = json.loads(args.split_json.read_text(encoding="utf-8-sig"))

    helper = LoaderHelper(config)
    train_idx, hold_idx = set_fixed_split(helper, split)
    all_train_ids = [helper.patient_ids[i] for i in train_idx]
    hold_ids = [helper.patient_ids[i] for i in hold_idx]

    position = {pid: i for i, pid in enumerate(helper.patient_ids)}
    train_ids, internal_ids = stratified_carve(
        all_train_ids, clinical, args.internal_val_size, args.internal_val_seed)
    train_idx = [position[p] for p in train_ids]
    internal_idx = [position[p] for p in internal_ids]
    # Only the reduced set is trained on; the carve would be meaningless if the
    # loader still fed the held-out patients back in.
    helper.fold_indices = [(train_idx, hold_idx)]

    # Standardise on the training patients only; the holdout never informs it.
    train_ratio = np.array([ratio_of(clinical, p) for p in train_ids], dtype=float)
    mean, std = float(train_ratio.mean()), float(train_ratio.std() or 1.0)
    print(f"fixed split: train={len(train_ids)} internal_val={len(internal_ids)} "
          f"holdout={len(hold_ids)}; ratio mean={mean:.2f} sd={std:.2f}; "
          f"NO early stopping", flush=True)

    aux = None
    if args.aux_pft_csv is not None:
        order = AUX_PFT_TARGETS
        aux_rows, _ = load_aux_targets(args.aux_pft_csv, order)
        values, reasons = {}, {}
        for pid in train_ids:
            got, why = aux_of(aux_rows, clinical, pid, order)
            values[pid] = got
            if got is None:
                reasons[why] = reasons.get(why, 0) + 1
        covered = sum(1 for v in values.values() if v is not None)
        aux = AuxTargets(values, order, train_ids, args.aux_weight)
        print(f"auxiliary targets {list(order)} weight {args.aux_weight} each: "
              f"{covered}/{len(train_ids)} training patients covered", flush=True)
        for name, (m, s) in zip(order, aux.stats):
            print(f"  {name:<14} mean={m:.3f} sd={s:.3f}", flush=True)
        for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  skipped {n:>3}: {why}", flush=True)
    if args.skip_holdout:
        print("frozen holdout will NOT be scored in this run", flush=True)

    use_amp = bool(config.training.amp)
    internal_curve: list[dict] = []
    eval_hook = None
    if internal_ids:
        internal_ratio = np.array([ratio_of(clinical, p) for p in internal_ids])
        internal_y = (internal_ratio < FIXED_CUTOFF).astype(int)
        internal_gli = np.array([gli_cutoff_of(clinical, p) for p in internal_ids])
        internal_y_gli = (internal_ratio < internal_gli).astype(int)

        def eval_hook(epoch, model, train_loss):  # noqa: F811
            pred_map = predict(model, inference_loader(helper, internal_idx),
                               device, use_amp, mean, std)
            pred = np.array([pred_map[p] for p in internal_ids])
            row = {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 6),
                "internal_mae": round(float(np.abs(pred - internal_ratio).mean()), 4),
                "internal_r": round(float(np.corrcoef(pred, internal_ratio)[0, 1]), 4),
                "internal_auc_fixed70": round(float(roc_auc_score(internal_y, -pred)), 4),
            }
            if len(set(internal_y_gli.tolist())) > 1:
                row["internal_auc_gli"] = round(
                    float(roc_auc_score(internal_y_gli, -pred)), 4)
            internal_curve.append(row)
            print("  [internal] epoch {:3d}  MAE {:.3f}  r {:.4f}  AUC70 {:.4f}"
                  "  AUCgli {}".format(epoch, row["internal_mae"], row["internal_r"],
                                       row["internal_auc_fixed70"],
                                       row.get("internal_auc_gli", "-")), flush=True)

    model = train_regressor(config, helper, clinical, args.seed, args.epochs,
                            device, mean, std, eval_hook=eval_hook,
                            eval_every=args.eval_every, aux=aux)
    train_pred_map = predict(model, inference_loader(helper, train_idx),
                             device, use_amp, mean, std)
    gap = [p for p in train_ids if p not in train_pred_map]
    if gap:
        raise SystemExit(f"no prediction for {len(gap)} training patients: {gap[:5]}")

    if args.skip_holdout:
        summary = {
            "meta": {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": str(args.config),
                "n_training": len(train_ids),
                "n_internal_val": len(internal_ids),
                "internal_val_patient_ids": internal_ids,
                "internal_val_seed": args.internal_val_seed,
                "epochs": args.epochs,
                "seed": args.seed,
                "effective_batch_size": int(helper.batch_size),
                "samples_per_epoch": len(helper.get_train_dl(0, shuffle=False))
                * int(helper.batch_size),
                "image_size": list(config.data.image_size),
                "in_channels": int(config.model.in_channels),
                "holdout_scored": False,
                "why": ("epoch budget and regularisation are being chosen here, so "
                        "the frozen holdout is deliberately left untouched"),
            },
            "training_in_sample": {
                "ratio_mae": round(float(np.abs(
                    np.array([train_pred_map[p] for p in train_ids]) - train_ratio
                ).mean()), 4),
            },
            "internal_validation_curve": internal_curve,
        }
        torch.save({"state_dict": model.state_dict(), "seed": args.seed,
                    "epochs": args.epochs, "ratio_mean": mean, "ratio_std": std},
                   args.out / f"regressor_seed{args.seed}.pth")
        (args.out / f"results_seed{args.seed}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n內部驗證曲線(未碰凍結 200)")
        print("{:>7}{:>12}{:>10}{:>10}{:>11}{:>10}".format(
            "epoch", "訓練loss", "MAE", "r", "AUC70", "AUCgli"))
        print("-" * 60)
        for row in internal_curve:
            print("{:>7}{:>12.5f}{:>10.3f}{:>10.4f}{:>11.4f}{:>10}".format(
                row["epoch"], row["train_loss"], row["internal_mae"], row["internal_r"],
                row["internal_auc_fixed70"], row.get("internal_auc_gli", "-")))
        if internal_curve:
            best = max(internal_curve, key=lambda r: r["internal_auc_fixed70"])
            print(f"\n  AUC70 最佳 epoch {best['epoch']}  ({best['internal_auc_fixed70']:.4f})")
            best_mae = min(internal_curve, key=lambda r: r["internal_mae"])
            print(f"  MAE 最佳 epoch {best_mae['epoch']}  ({best_mae['internal_mae']:.3f})")
        print(f"\n總耗時 {(time.time()-started)/60:.1f} 分  ->  {args.out}")
        return

    hold_pred_map = predict(model, helper.get_test_dl(0, shuffle=False),
                            device, use_amp, mean, std)
    gap = [p for p in hold_ids if p not in hold_pred_map]
    if gap:
        raise SystemExit(f"no prediction for {len(gap)} holdout patients: {gap[:5]}")

    tp_arr = np.array([train_pred_map[p] for p in train_ids])
    hp_arr = np.array([hold_pred_map[p] for p in hold_ids])
    hold_ratio = np.array([ratio_of(clinical, p) for p in hold_ids], dtype=float)

    fixed_train = np.full(len(train_ids), FIXED_CUTOFF)
    fixed_hold = np.full(len(hold_ids), FIXED_CUTOFF)
    gli_train = np.array([gli_cutoff_of(clinical, p) for p in train_ids])
    gli_hold = np.array([gli_cutoff_of(clinical, p) for p in hold_ids])

    results = {"meta": {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": str(args.config),
        "split_json": str(args.split_json),
        "pft_csv": str(args.pft_csv),
        "n_training": len(train_ids),
        "n_holdout": len(hold_ids),
        "epochs": args.epochs,
        "seed": args.seed,
        # Read back off the helper rather than off the config: attention-heavy
        # models take swin_batch_size and leave training.batch_size unused, so
        # the config field is not what the run actually did.
        "effective_batch_size": int(helper.batch_size),
        "effective_eval_batch_size": int(helper.val_batch_size),
        "samples_per_epoch": len(helper.get_train_dl(0, shuffle=False)) * int(helper.batch_size),
        "image_size": list(config.data.image_size),
        "target_spacing": (list(config.data.target_spacing)
                           if config.data.target_spacing else None),
        "in_channels": int(config.model.in_channels),
        "objective": "SmoothL1 on the standardised FEV1/FVC ratio",
        "ratio_standardisation": {"mean": round(mean, 4), "std": round(std, 4),
                                  "fitted_on": "training patients only"},
        "validation_used_during_training": False,
        "holdout_used_for_fitting_or_threshold_selection": False,
        "epoch_basis": f"{args.epochs} passed on the command line, not selected on CV",
    }}

    results["training_in_sample"] = {
        "ratio_mae": round(float(np.abs(tp_arr - train_ratio).mean()), 4),
        "ratio_pearson_r": round(float(np.corrcoef(tp_arr, train_ratio)[0, 1]), 4),
    }
    results["holdout_ratio"] = {
        "mae": round(float(np.abs(hp_arr - hold_ratio).mean()), 4),
        "pearson_r": round(float(np.corrcoef(hp_arr, hold_ratio)[0, 1]), 4),
        "predicted_range": [round(float(hp_arr.min()), 2), round(float(hp_arr.max()), 2)],
        "true_range": [round(float(hold_ratio.min()), 2), round(float(hold_ratio.max()), 2)],
    }

    blocks = {}
    for tag, cut_h, cut_t in (("fixed70", fixed_hold, fixed_train),
                              ("gli", gli_hold, gli_train)):
        y = (hold_ratio < cut_h).astype(int)
        dist = np.abs(hold_ratio - cut_h)
        blocks[f"{tag}_at_label_rule"] = score_block(
            y, hp_arr, cut_h, dist, args.margin,
            "predicted ratio compared with the clinical cutoff; nothing fitted")
        delta, train_bal = calibrate_offset(tp_arr, train_ratio, cut_t)
        blocks[f"{tag}_at_calibrated_offset"] = score_block(
            y, hp_arr, cut_h + delta, dist, args.margin,
            f"cutoff shifted by {delta:+.2f} points, chosen on TRAINING IN-SAMPLE "
            f"predictions (optimistic; balanced accuracy there was {train_bal})")
        blocks[f"{tag}_at_calibrated_offset"]["offset_points"] = delta
        blocks[f"{tag}_auc_95ci"] = boot_ci(y, -hp_arr)
    results["holdout"] = blocks

    results["patients"] = {
        p: {"predicted_ratio": round(float(hold_pred_map[p]), 3),
            "true_ratio": ratio_of(clinical, p),
            "gli_lln": gli_cutoff_of(clinical, p)}
        for p in hold_ids
    }

    torch.save({"state_dict": model.state_dict(), "seed": args.seed,
                "epochs": args.epochs, "ratio_mean": mean, "ratio_std": std},
               args.out / f"regressor_seed{args.seed}.pth")
    (args.out / f"results_seed{args.seed}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n凍結 {len(hold_ids)} 人  比值 MAE {results['holdout_ratio']['mae']}  "
          f"r {results['holdout_ratio']['pearson_r']}")
    print("{:34s}{:>9}{:>10}{:>10}{:>10}{:>12}".format(
        "決策規則", "AUC", "balacc", "敏感度", "特異度", "邊界AUC"))
    print("-" * 85)
    for key in ("fixed70_at_label_rule", "fixed70_at_calibrated_offset",
                "gli_at_label_rule", "gli_at_calibrated_offset"):
        b = blocks[key]
        border = b["by_band"].get("borderline", {}).get("auc", float("nan"))
        print("{:34s}{:>9.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>12.4f}".format(
            key, b["auc"], b["balanced_accuracy"], b["sensitivity"],
            b["specificity"], border))
    print(f"\n總耗時 {(time.time()-started)/60:.1f} 分  ->  {args.out}")


if __name__ == "__main__":
    main()
