# Five-window SE-ResNet50 experiment

New opt-in implementation of arXiv:2605.12562v1 on the existing 777-patient cohort.
No existing training code, configs, datasets or deployment checkpoints are changed.

## Protocol and limits

- Fixed train/val/test split: 461/116/200; retains official200, excludes stale
  the head/neck scan named in the cohort decisions file. Labels remain the source FEV1/FVC<70 rule, including
  pre-bronchodilator measurements when post measurements were unavailable.
- Read original DICOM using the previously audited series UID, convert to HU and
  LPS orientation, reverse z to apex-to-base. Original arrays are never overwritten.
- Paper sampling is underspecified: use 32 unique uniformly spaced slices within
  40%-90% of the original axial stack; expand to 10%-90%, then whole stack only
  when fewer than 32 unique slices remain. Record indices and source metadata per
  patient. Resize in-plane to 512x512, then apply each window. This interpretation
  and the local cohort mean this is not a bit-exact replication.
- Five windows (level,width): lung(-600,1500), mediastinal(20,350),
  HRCT(-600,2000), zero(0,1500), bone(250,1000). Train-only global mean/std.
- MONAI 3D SEResNet50, 2048-dimensional pooled features, single binary logit.
  Adam 0.001, cosine schedule 40 epochs, validation-BCE early stopping patience10.
  No data augmentation or pretrained external weights.
- RTX3060Ti: fp16 AMP and microbatch1, accumulated to effective batch4.
  BatchNorm remains microbatch1, **not equivalent** to paper batch4 BatchNorm.
- Five supervised models. Teacher chosen by validation AUC of each selected
  checkpoint. Four students continue from their own pretrained checkpoints,
  with 0.5 BCE + 0.5 raw-feature MSE. Teacher eval features are computed once
  on training patients and cached losslessly (no stochastic augmentation).
- Four continuation controls start from the same student weights and receive
  the same epochs as their distilled counterpart, resetting RNG and optimizer.
  Their cosine horizon remains 40; checkpoint selection uses validation BCE.
- Three logistic regression stacks: initial supervised, distilled and matched
  continued-training models. All fit only on validation probabilities (C=1,
  max_iter=1000). Thresholds use validation Youden index; test is evaluated after
  all training. Report paired delta-AUC bootstrap intervals and each window.

## Run / resume

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba
cd /mnt/d/Felix/Hospital/nnMamba/classification
python -m experiments.full_window_distillation.test_full
python -u -m experiments.full_window_distillation.run \
  --manifest ../weights/window_distillation/cohort777/patients.csv \
  --source-summary /home/felix/Research/nnMamba/classification/datasets/normal_v_abnormal_fev1fvc70/build_summary.json \
  --output ../weights/window_distillation/seresnet50_full777 \
  --hours 23
```

Run the same command to resume. Finished stages are reused; latest epoch stores
optimizer, scheduler, AMP scaler and Torch/CUDA RNG states. An interrupted partial
epoch is replayed from the previous checkpoint. Each invocation's --hours budget
resets, so the supervising agent must preserve the user's original 24h deadline
and only pass the remaining budget when recovering. Deadline checks occur between
operations and may overrun by one operation. Original source/config fingerprints
must match on resume. Completion is only recorded after all 13 stages and stacks.

Status: `status.json`; stage curves: `<stage>/progress.json`; outputs:
`results.json`, `test_predictions.csv`, per-stage `best.pt`/`last.pt`.
All artifacts are under gitignored weights. Synthetic tests have explicitly
separate output folders and are not evidence of clinical efficacy.
