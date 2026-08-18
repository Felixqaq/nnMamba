# Per-patient predictions and hard-vote result

- Generated: 2026-08-04T11:39:46
- Cohort: 61 patients
- Vote: hard majority over 3 models (image, fusion, quant) — every patient now has a decision; no more `pending`

> **Caveat.** The `image` / `fusion` per-patient predictions were saved at each fold's best epoch, and that epoch was selected on the test fold itself (`trainer.py` only calls `save_predictions` when the fold score improves), so those two rows are an optimistic upper bound. `quant` is the last epoch of a fixed 100-epoch budget with no test-fold selection, so it carries no such bias — its pooled out-of-fold accuracy 0.754 matches the 5-fold mean in `result.md`. The `vote` row contains `image` / `fusion` and therefore inherits their optimism.

## Classes

- Positive: `Abnormal/emphysema-like (AC <=131 deg)` (shown as **Abnormal**)
- Negative: `Normal-like (AC >=152 deg)` (shown as **Normal-like**)

## Sources

- `image`: `figures/RQ2b_angle_binary_extreme/hybrid_mamba_attention_rq2b_angle_binary_extreme_image_2026-07-14_17:03:26`
- `fusion`: `figures/RQ2b_angle_binary_extreme/hybrid_mamba_tapct_fusion_rq2b_angle_binary_extreme_fusion_2026-07-14_17:24:43`
- `quant`: `models/rq_quant/param_aeropath__rq2b_per_patient.json (export_quant_per_patient.py, param_aeropath, seed=42, 5-fold, 100 epochs, early stopping off)`

`quant` = 與 result.md 中 RQ2b 極端二分類的「量化(12 特徵+FCNN)」那一列 (Acc 0.755±0.087) 完全相同的設定(`param_aeropath` 特徵、seed=42、5-fold stratified、100 epochs、early stopping 關閉、class-weighted CE、per-fold StandardScaler、argmax)。

> Note: `fold` is the deep models' fold index; `quant fold` is the quantitative model's own fold index. Both are seed=42 5-fold stratified, but the cohort ordering differs so the groupings differ; per-patient voting does not depend on the fold index.

## Summary

| model | n | Accuracy | Sensitivity (Abnormal) | Specificity (Normal-like) | TP | FN | TN | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image | 61 | 0.8852 | 0.8571 | 0.8936 | 12 | 2 | 42 | 5 |
| fusion | 61 | 0.8689 | 0.8571 | 0.8723 | 12 | 2 | 41 | 6 |
| quant | 61 | 0.7541 | 0.7857 | 0.7447 | 11 | 3 | 35 | 12 |
| **vote** | 61 | 0.9016 | 0.9286 | 0.8936 | 13 | 1 | 42 | 5 |

With the third model in place the vote covers all 61 patients, on the same cohort as the three single-model rows, so the four rows are directly comparable.
Vote structure: unanimous 3-0 on 45 patients (41 correct, 91.1%); split 2-1 on 16 patients (14 correct, 87.5%).

### Previously pending (7) — now broken by `quant`

These are exactly the cases where `image` and `fusion` contradicted each other. The `quant` vote lands **5/7** of them on the correct side (71.4%).

| patient_id | true | GOLD | image | fusion | quant | vote | correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1604378 | Normal-like | GOLD 1 (Mild) | Normal-like | Abnormal | Normal-like | Normal-like | yes |
| 2500824 | Normal-like | GOLD 1 (Mild) | Abnormal | Normal-like | Normal-like | Normal-like | yes |
| 4710629 | Normal-like | GOLD 2 (Moderate) | Normal-like | Abnormal | Normal-like | Normal-like | yes |
| C905524 | Normal-like | GOLD 3 (Severe) | Abnormal | Normal-like | Abnormal | Abnormal | no |
| D132855 | Abnormal | GOLD 2 (Moderate) | Normal-like | Abnormal | Abnormal | Abnormal | yes |
| E353272 | Abnormal | GOLD 4 (Very Severe) | Abnormal | Normal-like | Abnormal | Abnormal | yes |
| E771850 | Normal-like | GOLD 3 (Severe) | Normal-like | Abnormal | Abnormal | Abnormal | no |

### Vote errors (6)

| patient_id | true | GOLD | image | fusion | quant | vote | margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3647457 | Normal-like | GOLD 3 (Severe) | Abnormal | Abnormal | Abnormal | Abnormal | 3-0 |
| 8009284 | Normal-like | GOLD 3 (Severe) | Abnormal | Abnormal | Abnormal | Abnormal | 3-0 |
| 8404129 | Abnormal | GOLD 3 (Severe) | Normal-like | Normal-like | Normal-like | Normal-like | 3-0 |
| 9529629 | Normal-like | GOLD 2 (Moderate) | Abnormal | Abnormal | Abnormal | Abnormal | 3-0 |
| C905524 | Normal-like | GOLD 3 (Severe) | Abnormal | Normal-like | Abnormal | Abnormal | 2-1 |
| E771850 | Normal-like | GOLD 3 (Severe) | Normal-like | Abnormal | Abnormal | Abnormal | 2-1 |

## Per-patient table

| patient_id | fold | quant fold | true | GOLD | image pred | image p(pos) | fusion pred | fusion p(pos) | quant pred | quant p(pos) | vote | margin | vote correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0781915 | 1 | 3 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.363 | Normal-like | 0.004 | Normal-like | 3-0 | yes |
| 1261736 | 4 | 4 | Abnormal | GOLD 4 (Very Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.883 | Abnormal | 3-0 | yes |
| 1596038 | 1 | 3 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.345 | Normal-like | 0.452 | Normal-like | 3-0 | yes |
| 1604378 | 1 | 4 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Abnormal ✗ | 0.521 | Normal-like | 0.151 | Normal-like | 2-1 | yes |
| 1663485 | 5 | 2 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.001 | Normal-like | 0.000 | Normal-like | 0.059 | Normal-like | 3-0 | yes |
| 1687031 | 4 | 3 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.765 | Abnormal | 3-0 | yes |
| 1746380 | 1 | 5 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.002 | Normal-like | 0.415 | Normal-like | 0.131 | Normal-like | 3-0 | yes |
| 1800944 | 1 | 1 | Abnormal | GOLD 3 (Severe) | Abnormal | 0.848 | Abnormal | 0.584 | Normal-like ✗ | 0.203 | Abnormal | 2-1 | yes |
| 1814107 | 5 | 3 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.037 | Normal-like | 3-0 | yes |
| 2094528 | 1 | 3 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.289 | Normal-like | 0.021 | Normal-like | 3-0 | yes |
| 2221276 | 5 | 5 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.010 | Normal-like | 0.000 | Normal-like | 0.021 | Normal-like | 3-0 | yes |
| 2256243 | 3 | 2 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.001 | Normal-like | 0.217 | Abnormal ✗ | 0.503 | Normal-like | 2-1 | yes |
| 2291134 | 4 | 5 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.046 | Normal-like | 3-0 | yes |
| 2500824 | 2 | 3 | Normal-like | GOLD 1 (Mild) | Abnormal ✗ | 1.000 | Normal-like | 0.000 | Normal-like | 0.048 | Normal-like | 2-1 | yes |
| 2588424 | 5 | 1 | Abnormal | GOLD 4 (Very Severe) | Abnormal | 0.585 | Abnormal | 0.998 | Abnormal | 0.924 | Abnormal | 3-0 | yes |
| 2860903 | 3 | 4 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.036 | Normal-like | 0.059 | Normal-like | 3-0 | yes |
| 2991621 | 2 | 3 | Abnormal | GOLD 4 (Very Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.914 | Abnormal | 3-0 | yes |
| 3097765 | 4 | 2 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.048 | Normal-like | 3-0 | yes |
| 3635301 | 2 | 2 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.058 | Normal-like | 3-0 | yes |
| 3647457 | 2 | 2 | Normal-like | GOLD 3 (Severe) | Abnormal ✗ | 1.000 | Abnormal ✗ | 1.000 | Abnormal ✗ | 0.616 | Abnormal | 3-0 | no |
| 4205212 | 2 | 3 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.011 | Normal-like | 3-0 | yes |
| 4230847 | 4 | 1 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.087 | Normal-like | 3-0 | yes |
| 4302294 | 3 | 1 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.001 | Normal-like | 0.029 | Normal-like | 0.038 | Normal-like | 3-0 | yes |
| 4372708 | 3 | 3 | Abnormal | GOLD 2 (Moderate) | Abnormal | 0.829 | Abnormal | 0.942 | Abnormal | 0.814 | Abnormal | 3-0 | yes |
| 4710629 | 1 | 4 | Normal-like | GOLD 2 (Moderate) | Normal-like | 0.000 | Abnormal ✗ | 0.528 | Normal-like | 0.182 | Normal-like | 2-1 | yes |
| 4996166 | 4 | 5 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.080 | Normal-like | 3-0 | yes |
| 5046455 | 1 | 3 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.323 | Normal-like | 0.074 | Normal-like | 3-0 | yes |
| 5390303 | 4 | 1 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.088 | Normal-like | 3-0 | yes |
| 5925853 | 4 | 2 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Abnormal ✗ | 0.680 | Normal-like | 2-1 | yes |
| 6212308 | 3 | 1 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.001 | Normal-like | 0.241 | Normal-like | 0.043 | Normal-like | 3-0 | yes |
| 6312603 | 3 | 1 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.082 | Normal-like | 0.166 | Normal-like | 3-0 | yes |
| 6757504 | 5 | 3 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.002 | Normal-like | 0.000 | Normal-like | 0.027 | Normal-like | 3-0 | yes |
| 6858508 | 2 | 5 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.046 | Normal-like | 3-0 | yes |
| 6887256 | 1 | 2 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 0.551 | Abnormal | 0.833 | Abnormal | 3-0 | yes |
| 7871759 | 4 | 4 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.053 | Normal-like | 3-0 | yes |
| 8009284 | 3 | 4 | Normal-like | GOLD 3 (Severe) | Abnormal ✗ | 0.998 | Abnormal ✗ | 0.814 | Abnormal ✗ | 0.948 | Abnormal | 3-0 | no |
| 8126939 | 2 | 5 | Normal-like | GOLD 2 (Moderate) | Normal-like | 0.000 | Normal-like | 0.108 | Abnormal ✗ | 0.838 | Normal-like | 2-1 | yes |
| 8244460 | 5 | 2 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.117 | Normal-like | 3-0 | yes |
| 8332556 | 5 | 2 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.112 | Normal-like | 0.000 | Normal-like | 0.159 | Normal-like | 3-0 | yes |
| 8404129 | 1 | 5 | Abnormal | GOLD 3 (Severe) | Normal-like ✗ | 0.088 | Normal-like ✗ | 0.475 | Normal-like ✗ | 0.035 | Normal-like | 3-0 | no |
| 9075311 | 4 | 1 | Normal-like | GOLD 2 (Moderate) | Normal-like | 0.000 | Normal-like | 0.474 | Normal-like | 0.274 | Normal-like | 3-0 | yes |
| 9529629 | 2 | 4 | Normal-like | GOLD 2 (Moderate) | Abnormal ✗ | 1.000 | Abnormal ✗ | 0.966 | Abnormal ✗ | 0.944 | Abnormal | 3-0 | no |
| A267542 | 5 | 5 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.038 | Normal-like | 3-0 | yes |
| A613117 | 1 | 4 | Normal-like | GOLD 2 (Moderate) | Normal-like | 0.000 | Normal-like | 0.440 | Normal-like | 0.468 | Normal-like | 3-0 | yes |
| A754735 | 3 | 5 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.024 | Normal-like | 0.051 | Normal-like | 3-0 | yes |
| A762364 | 5 | 1 | Normal-like | GOLD 2 (Moderate) | Normal-like | 0.003 | Normal-like | 0.000 | Abnormal ✗ | 0.836 | Normal-like | 2-1 | yes |
| B213449 | 5 | 2 | Abnormal | GOLD 4 (Very Severe) | Abnormal | 0.992 | Abnormal | 1.000 | Normal-like ✗ | 0.433 | Abnormal | 2-1 | yes |
| C041635 | 4 | 2 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.197 | Normal-like | 3-0 | yes |
| C081146 | 3 | 2 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.028 | Normal-like | 0.040 | Normal-like | 3-0 | yes |
| C435832 | 2 | 5 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.748 | Abnormal | 3-0 | yes |
| C543831 | 2 | 4 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Normal-like | 0.047 | Normal-like | 3-0 | yes |
| C586742 | 1 | 1 | Normal-like | GOLD 3 (Severe) | Normal-like | 0.013 | Normal-like | 0.461 | Abnormal ✗ | 0.686 | Normal-like | 2-1 | yes |
| C905524 | 5 | 5 | Normal-like | GOLD 3 (Severe) | Abnormal ✗ | 0.988 | Normal-like | 0.005 | Abnormal ✗ | 0.951 | Abnormal | 2-1 | no |
| D132855 | 3 | 4 | Abnormal | GOLD 2 (Moderate) | Normal-like ✗ | 0.002 | Abnormal | 0.793 | Abnormal | 0.846 | Abnormal | 2-1 | yes |
| D550510 | 2 | 3 | Normal-like | GOLD 2 (Moderate) | Normal-like | 0.000 | Normal-like | 0.000 | Abnormal ✗ | 0.862 | Normal-like | 2-1 | yes |
| E353272 | 4 | 1 | Abnormal | GOLD 4 (Very Severe) | Abnormal | 0.876 | Normal-like ✗ | 0.000 | Abnormal | 0.890 | Abnormal | 2-1 | yes |
| E558113 | 2 | 5 | Abnormal | GOLD 4 (Very Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.934 | Abnormal | 3-0 | yes |
| E647833 | 3 | 4 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 0.643 | Abnormal | 0.642 | Abnormal | 3-0 | yes |
| E717248 | 1 | 1 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.339 | Normal-like | 0.213 | Normal-like | 3-0 | yes |
| E771850 | 3 | 1 | Normal-like | GOLD 3 (Severe) | Normal-like | 0.021 | Abnormal ✗ | 0.569 | Abnormal ✗ | 0.765 | Abnormal | 2-1 | no |
| E797258 | 5 | 4 | Normal-like | GOLD 1 (Mild) | Normal-like | 0.000 | Normal-like | 0.000 | Abnormal ✗ | 0.757 | Normal-like | 2-1 | yes |
