#!/usr/bin/env python3
"""Phase B -- train the deployment ensemble on every available patient.

Nothing here is selected: the epoch budget and the decision threshold are read
from the Phase A run, which chose them on training-fold out-of-fold predictions
alone. This script only refits the same recipe on the union of Phase A's
training and holdout sets, so the expected field performance of these weights is
the number Phase A measured on its frozen 200 -- not something this run can
claim for itself. There is no held-out score here and there must not be one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Config  # noqa: E402
from core.runtime import configure_torch_runtime  # noqa: E402
from data.loader import RegressionLoaderHelper as LoaderHelper  # noqa: E402

from train_fixed_holdout_ensemble import predict_member, train_member  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=72)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_of(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_torch_runtime()
    device = torch.device(args.device)

    split = json.loads(args.split_json.read_text(encoding="utf-8-sig"))
    calibration = json.loads(args.calibration_json.read_text(encoding="utf-8-sig"))
    # The calibration file names it "selected_threshold" at the top level and
    # repeats it inside oof_metrics; accept either rather than guess.
    raw_threshold = calibration.get("selected_threshold")
    if raw_threshold is None:
        raw_threshold = calibration.get("oof_metrics", {}).get("threshold")
    if raw_threshold is None:
        raise SystemExit(
            f"no threshold in {args.calibration_json}; keys={sorted(calibration)}"
        )
    threshold = float(raw_threshold)
    all_ids = sorted(
        set(split["training_patient_ids"]) | set(split["validation_patient_ids"])
    )
    print(
        f"deployment fit: {len(all_ids)} patients "
        f"({len(split['training_patient_ids'])} training + "
        f"{len(split['validation_patient_ids'])} former holdout)",
        flush=True,
    )
    print(f"inherited threshold={threshold:.8f}  epochs={args.epochs}", flush=True)

    config = Config.from_yaml(args.config)
    config.data.source_dir = args.source_dir
    config.data.manifest = args.manifest
    helper = LoaderHelper(config)

    index_by_id = {pid: i for i, pid in enumerate(helper.patient_ids)}
    missing = sorted(set(all_ids) - set(index_by_id))
    extra = sorted(set(index_by_id) - set(all_ids))
    if missing or extra:
        raise SystemExit(f"cohort mismatch: missing={missing} extra={extra}")

    indices = [index_by_id[pid] for pid in all_ids]
    class_names = helper.get_class_names()
    abnormal_index = class_names.index("Abnormal")
    labels = np.asarray(
        [int(helper.targets[i]) == abnormal_index for i in indices], dtype=int
    )
    print(
        f"labels: Abnormal={int(labels.sum())} Normal={int((1 - labels).sum())}",
        flush=True,
    )

    # Train on everything; the "test" half only exists because the loader wants a
    # pair. Its predictions are in-sample and are written as a sanity check, never
    # as performance.
    helper.fold_indices = [(indices, indices)]
    helper.k_folds = 1

    seeds = [args.base_seed + i for i in range(args.members)]
    checkpoints = []
    for member_index, seed in enumerate(seeds, start=1):
        checkpoint = args.out / f"deployment_member_{member_index:02d}_seed{seed}.pth"
        insample = args.out / f"deployment_member_{member_index:02d}_insample.json"
        if checkpoint.exists() and insample.exists():
            print(f"member {member_index}/{args.members}: reusing seed={seed}", flush=True)
            checkpoints.append(checkpoint.name)
            continue
        print(
            f"member {member_index}/{args.members}: seed={seed} "
            f"train={len(indices)} epochs={args.epochs}",
            flush=True,
        )
        model = train_member(config, helper, seed, args.epochs, device)
        raw = predict_member(model, helper, device, bool(config.training.amp))
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": seed,
                "epochs": args.epochs,
                "training_patient_ids": all_ids,
                "evaluation_patient_ids": [],
                "phase": "deployment_full_cohort",
                "threshold": threshold,
                "class_names": class_names,
                "model_name": config.model.name,
                "image_size": list(config.data.image_size),
            },
            checkpoint,
        )
        insample.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        checkpoints.append(checkpoint.name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "weights for deployment at a new hospital",
        "expected_performance_source": str(args.split_json),
        "performance_note": (
            "These weights were fitted on every available patient, so no honest "
            "score can be computed from them. Quote the Phase A frozen-200 "
            "numbers instead; that run used the same recipe on a subset."
        ),
        "model_name": config.model.name,
        "members": args.members,
        "seeds": seeds,
        "epochs": args.epochs,
        "decision_threshold": threshold,
        "threshold_source": str(args.calibration_json),
        "vote_rule": f"Abnormal if >= {args.members // 2 + 1} of {args.members} members exceed the threshold",
        "class_names": class_names,
        "n_training_patients": len(all_ids),
        "n_abnormal": int(labels.sum()),
        "n_normal": int((1 - labels).sum()),
        "training_patient_ids_sha256": sha256_of(all_ids),
        "checkpoints": checkpoints,
        "preprocessing": {
            "image_size": list(config.data.image_size),
            "intensity_window_hu": list(config.data.intensity_window),
            "input_normalization": config.data.input_normalization,
            "label_rule": "Abnormal if FEV1/FVC < 70% (strict)",
        },
    }
    (args.out / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / "deployment_training_patient_ids.json").write_text(
        json.dumps(all_ids, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    print("DEPLOYMENT_TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
