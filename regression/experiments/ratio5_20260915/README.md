# Expanded CT + LAA two-stage ratio ensemble

Authorized split: preserve the original official200 patient IDs; train five fresh
models on every other eligible case. Expected 900 total / 700 training / 200 test,
subject to conversion and preprocessing checks. Never include Test or non_PFT.

`run.py` snapshots the 890 successful CTs by hard link into a new WSL output tree,
adds the eligible extra deliveries, verifies unique IDs and labels, preserves
existing density arrays, and computes only missing lung masks and LAA arrays.
New masks use TotalSegmentator fast. Mask and CT grids must match. The original
dataset, density arrays and old model weights are read-only inputs.

Training uses the existing hybrid Mamba ratio pipeline: 112x136x112 CT + native
LAA density, SmoothL1 on training-standardized FEV1/FVC, 80 fixed epochs, five
seeds 72–76, no early stopping. Data-loader workers reduced to four for memory
headroom; architecture and optimization match the existing two-channel config.
Training runs use --skip-holdout. The 200 patients are scored only after all five
models finish. Fixed70 is the primary rule; missing GLI is reported, never filled
with invented cutoffs. Clinical ratios come from reconciled build_summary rows.

Report single seed72, first three and all five; mean predicted ratio and majority
vote are separate rules. Vote fractions and FEV1/FVC percentages are not calibrated
COPD probabilities. Historical repeated use of official200 limits claims of fresh
test-set independence. Do not label full-cohort training or hospital deployment as
completed: this run preserves the 200-case test set and prepares evaluated weights.

Artifacts in WSL:
/home/felix/Research/nnMamba/regression/outputs/ratio5_expanded_20260915

Logs, status and final report on Windows:
D:/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260915

Start via launch.sh with activated conda nnMamba. Completed seeds are reused on
resume; interrupted seeds restart, because the inherited trainer saves at the end.
Use a persistent exec session, not a short-lived nohup WSL invocation.

Validation: test_pipeline.py passed a real 12-patient CT+LAA one-epoch train and
score test; none of those 12 is in official200. Smoke scores are not performance
estimates. Initial conversion ran while a CSV whitespace fix was prepared;
continue_after_conversion.py waits for that original runner to exit before loading
the corrected runner, so it never starts two training pipelines simultaneously.

Completion verification (2026-09-16): the separate physician-review record removes
12 cases from scoring only. `split.json` still reserves the original 200 cases;
`split_scoring.json` evaluates 188, with the same 700 training patients. The
existing `ensemble3.json` and `ensemble5.json` describe that reviewed subset.
`finalize_evaluation.py` verifies all five 80-epoch checkpoints and separately
scores the original holdout into `ensemble3_official200.json` and
`ensemble5_official200.json`. It does not retrain, select checkpoints, or tune
thresholds. `report.py` labels each cohort explicitly and keeps the two results
separate. Do not compare a reviewed-subset score directly with historical scores
on all 200 patients.
