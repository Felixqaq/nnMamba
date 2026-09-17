# nnMamba cross-window distillation pilot

Adaptation of https://arxiv.org/html/2605.12562v1, not a reproduction of its AUC.

The new `networks/window_mamba_adapter.py` subclasses the unchanged classification
nnMamba and exposes its 448-dimensional pooled features without changing the
classifier, parameters, or checkpoint keys. Both teacher and
student use this backbone. Raw feature MSE matches the paper's objective; the
feature size need not be 2048 when both networks share the same representation.

`--backbone hybrid_mamba_attention` selects the current COPD pipeline backbone
(352 pooled features, base_channels=32, depths=3/3/3, one attention layer).
Its existing two-logit head is used through logit[1]-logit[0], equivalent to binary
softmax. This pilot defines class 1 as COPD; do not load old checkpoints whose
class ordering is reversed. Both arms are initialized and trained from scratch.
No existing regression files or deployment weights are modified.

This feature is opt-in through the new `train_window_distillation.py` entry point.
All implementation files are additions; existing model, training, and configuration
files remain unchanged by this task.

## Protocol

1. Supply one original HU NIfTI per patient, with labels 0=non-COPD, 1=COPD.
   CSV columns: `patient_id,path,label,split`; split must be train, val, or test.
   Relative paths resolve from the CSV directory. Reuse the cohort's existing
   patient split. Do not use augmented scans as additional patients, cherry-picked
   easy test cohorts, or infer COPD labels from arbitrary folder names.
2. Generate lung (level -600, width 1500) and mediastinal (20, 350) windows
   **before** resizing. Reorient to RAS and resize the whole volume to
   112x136x112. Fit per-window mean/std on training voxels only. This is a
   whole-volume adaptation, not the paper's 32-slice protocol.
3. Train both windows from identical initialization. Select the teacher by
   validation AUC, never by test results. Teacher is frozen including BatchNorm.
4. Fork the student's selected checkpoint into a control and a distilled arm.
   Reset RNG and optimizer identically; use equal additional epochs. Control uses
   BCE; distillation uses 0.5 BCE + 0.5 raw-feature MSE. No augmentation in this
   first pilot, ensuring perfectly aligned views and comparable training exposure.
5. Select each checkpoint/threshold on validation only; evaluate the held-out test
   set after all stages. Save paired predictions, metrics, and a paired bootstrap
   95% interval for delta AUC. This single-seed pilot cannot establish equivalence
   or robust superiority. Comparison to historical nnMamba requires the same
   cohort, split and preprocessing; the matched control is the primary comparator.

## Run from WSL

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba
cd /mnt/d/Felix/Hospital/nnMamba/classification
python scripts/test_window_distillation.py
python train_window_distillation.py \
  --manifest experiments/window_distillation/patients.csv \
  --output ../weights/window_distillation/pilot01 \
  --hu-confirmed --epochs 10 --batch-size 2 --minutes 180
```

Use a new output directory per run (overwrite is rejected). Data preparation is
included in the budget. Deadline checks occur between scans/batches; an individual
operation can overrun the deadline. Incomplete stages preserve checkpoints and
write `status=incomplete`, never a completed efficacy claim. A 3-hour budget is
not a convergence guarantee. The backbone currently runs fp32 for custom Mamba
kernel compatibility; measure actual time/memory before a full cohort experiment.

Outputs: cached paired views, four stage checkpoints, `results.json`, and
`test_predictions.csv` under the gitignored weights directory. No patient data
or example clinical results are committed. The test helper uses synthetic HU and
a tiny CPU model solely for implementation validation.
