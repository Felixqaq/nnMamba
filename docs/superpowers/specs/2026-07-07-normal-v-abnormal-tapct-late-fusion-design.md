# Normal vs Abnormal — TAP-CT Late Fusion (regression pipeline)

**Date:** 2026-07-07
**Goal:** Run Normal-vs-Abnormal binary classification using the regression pipeline's
latest fusion model (`hybrid_mamba_tapct_fusion`) instead of the standalone
`classification/` pipeline.

## Decision

Add a new classification `target_mode = normal_v_abnormal` inside `regression/`,
reusing the existing TAP-CT late-fusion machinery (the same path `oi_emphysema`
already uses end-to-end). We do **not** port the fusion model into `classification/`,
because that would require rebuilding dual-input (CT volume + patient embedding)
loading/training that already works in `regression/`.

## Model

- `hybrid_mamba_tapct_fusion`, `num_classes: 2`.
- CT volume → Hybrid Mamba-Attention encoder; concatenated (late fusion) with the
  frozen TAP-CT-S-3D patient-level embedding (1152-d).
- Hyperparameters cloned from `config.oi.emphysema.tapct_late_fusion.augmentationX5.yaml`
  (a working binary classifier on a similar small cohort).

## Data & labels

- **Source:** `../classification/datasets/normal_v_abnormal_54/` with `Normal/` (21)
  and `Abnormal/` (33) subfolders of symlinked NIfTIs. All 54 patients confirmed.
- **Label = folder name.** No threshold logic (unlike OI/GOLD/Angle) and no new JSON
  label source: reuse the `source_group = ct_path.parent.name` that `manifest.py`
  already records per case.
- **Class order:** index 0 = `Abnormal` (disease-positive), index 1 = `Normal` —
  matching the regression sibling binary modes (`oi_emphysema`, `angle_binary_extreme`)
  so the evaluator reports sensitivity/specificity for the disease-positive class.
  (This intentionally differs from the `classification/` pipeline's Normal=0/Abnormal=1.)
- **Angle-label bypass:** the current non-OI branch in `build_angle_manifest`
  (`regression/data/manifest.py:487-492`) skips any patient absent from the angle
  `labels_json`. `normal_v_abnormal` must not depend on angle labels, so it gets an
  OI-style bypass: `angle = label_map.get(pid, nan)`, `target = float(class_index)`,
  never skip on missing angle.
- **Embeddings:** reuse `regression/embeddings/tapct_s_3d/features.npz`. Its 66 case
  files are a superset of the 54 labeled patients — **no new extraction needed.**

## Code changes

New `target_mode` requires updating the five duplicated classification registries
(`task_type == target_mode`, confirmed at `core/trainer.py:160`), plus the Literal and
the num-classes default:

1. `core/config.py` — `TargetMode` Literal **and** `CLASSIFICATION_TARGET_MODES`;
   add `from_yaml` branch `normal_v_abnormal → default_num_classes = 2`.
2. `data/loader.py` — `CLASSIFICATION_TARGET_MODES`.
3. `data/dataset.py` — `CLASSIFICATION_TARGET_MODES`.
4. `core/evaluator.py` — `CLASSIFICATION_TASK_TYPES` (omitting this routes 2-class
   logits into regression metrics and crashes).
5. `core/visualizer.py` — `CLASSIFICATION_TASK_TYPES`.
6. `data/manifest.py` — add `NORMAL_V_ABNORMAL_NAMES = ["Abnormal", "Normal"]`, a
   `class_names` branch, and folder-name label handling in the loop (bypassing the
   angle-label requirement; `class_index/class_label` from `source_group`).

New config file: `regression/config.normal_v_abnormal.tapct_late_fusion.yaml`
(`target_mode: normal_v_abnormal`, `source_dir` → the 54-case folders,
`tapct_features` → `./embeddings/tapct_s_3d/features.npz`, new `manifest` path, no
`oi_json`/`oi_threshold`/`gold_*`). Augmentation defaults to the same 5x
`balance_then_augment` scheme as OI emphysema (33:21 imbalance, 54-case cohort).

## Verification

1. Build manifest → assert 54 records, 33 Abnormal / 21 Normal, every patient has an
   embedding.
2. Smoke test: 1 fold, 1–2 epochs, confirm forward/loss/eval run without crashing.
3. Full 5-fold CV → report per-fold and mean AUC / Accuracy; write confusion matrix
   and training curves under `regression/figures/`.
