# Hybrid-Mamba five-window distillation

Additive experiment using the existing HybridMambaAttentionRegressor with depths
(3,3,3), base channels 32, hidden head 256 and a single binary logit. The pooled
352-dimensional representation is distilled; no auxiliary emphysema loss is used.
Existing regression and SE-ResNet training code is not modified.

Same audited 777-patient split (461/116/200), 32x512x512 raw HU sampling, five
windows, training-only normalization, Adam 0.001, cosine 40 epochs, validation
BCE patience 10, AMP microbatch 1 and accumulation 4 as the full SE-ResNet run.
Hybrid uses GroupNorm, Mamba and attention; this is a backbone substitution,
not an exact reproduction of the paper. Feature MSE scale differs with backbone.

Five supervised models select a teacher by validation AUC; four students use
0.5 BCE + 0.5 feature MSE; four controls continue for identical epoch counts.
Three validation-fitted logistic regression ensembles are evaluated on the fixed
test set only after training. Report paired bootstrap AUC intervals.

The completed SE-ResNet cache is reused by hard links after checking source and
manifest hashes, shape, sampling and windows. Cached arrays are read-only inputs.
No SE-ResNet weights are reused. Separate artifacts: weights/window_distillation/hybrid_full777.

Run from classification in conda nnMamba using run_overnight.sh; --hours is per
invocation, so resume only with time remaining before the original 24-hour deadline.
Generate results with python -m experiments.full_hybrid_window_distillation.report RUN.

Validation: full-size real hybrid GPU forward/backward benchmark, and synthetic
13-stage workflow test including completed-run idempotence and matched epochs.

Recovery: initial FP16 run failed before epoch 1 completed with non-finite loss. Preserved hybrid_full777. New run hybrid_full777_fp32 disables autocast and GradScaler, preserving all other settings and the original absolute deadline. FP32 benchmark: 1636 MiB, 0.061 s/patient.
