# Per-patient predictions and hard-vote result

- Generated: 2026-08-04T11:39:46
- Cohort: 66 patients
- Vote: hard majority over 3 models (image, fusion, quant) — every patient now has a decision; no more `pending`

> **Caveat.** The `image` / `fusion` per-patient predictions were saved at each fold's best epoch, and that epoch was selected on the test fold itself (`trainer.py` only calls `save_predictions` when the fold score improves), so those two rows are an optimistic upper bound. `quant` is the last epoch of a fixed 100-epoch budget with no test-fold selection, so it carries no such bias — its pooled out-of-fold accuracy 0.803 matches the 5-fold mean in `result.md`. The `vote` row contains `image` / `fusion` and therefore inherits their optimism.

## Classes

- Positive: `Significant emphysema (OI >= 3)` (shown as **Significant emphysema**)
- Negative: `No significant emphysema (OI < 3)` (shown as **No significant emphysema**)

## Sources

- `image`: `figures/RQ3_oi_emphysema/hybrid_mamba_attention_rq3_oi_emphysema_image_2026-07-14_18:04:50`
- `fusion`: `figures/RQ3_oi_emphysema/hybrid_mamba_tapct_fusion_rq3_oi_emphysema_fusion_2026-07-14_19:04:41`
- `quant`: `models/rq_quant/param_aeropath__rq3_per_patient.json (export_quant_per_patient.py, param_aeropath, seed=42, 5-fold, 100 epochs, early stopping off)`

`quant` = 與 result.md 中 RQ3 OI 氣腫的「量化(12 特徵+FCNN)」那一列 (Acc 0.804±0.054) 完全相同的設定(`param_aeropath` 特徵、seed=42、5-fold stratified、100 epochs、early stopping 關閉、class-weighted CE、per-fold StandardScaler、argmax)。

> Note: `fold` is the deep models' fold index; `quant fold` is the quantitative model's own fold index. Both are seed=42 5-fold stratified, but the cohort ordering differs so the groupings differ; per-patient voting does not depend on the fold index.

## Summary

| model | n | Accuracy | Sensitivity (Significant emphysema) | Specificity (No significant emphysema) | TP | FN | TN | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image | 66 | 0.8030 | 0.7647 | 0.8438 | 26 | 8 | 27 | 5 |
| fusion | 66 | 0.8333 | 0.8529 | 0.8125 | 29 | 5 | 26 | 6 |
| quant | 66 | 0.8030 | 0.7353 | 0.8750 | 25 | 9 | 28 | 4 |
| **vote** | 66 | 0.8333 | 0.7647 | 0.9062 | 26 | 8 | 29 | 3 |

With the third model in place the vote covers all 66 patients, on the same cohort as the three single-model rows, so the four rows are directly comparable.
Vote structure: unanimous 3-0 on 44 patients (42 correct, 95.5%); split 2-1 on 22 patients (13 correct, 59.1%).

### Previously pending (16) — now broken by `quant`

These are exactly the cases where `image` and `fusion` contradicted each other. The `quant` vote lands **9/16** of them on the correct side (56.2%).

| patient_id | true | GOLD | image | fusion | quant | vote | correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1746380 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | Significant emphysema | No significant emphysema | No significant emphysema | yes |
| 2860903 | No significant emphysema | GOLD 1 (Mild) | Significant emphysema | No significant emphysema | No significant emphysema | No significant emphysema | yes |
| 3097765 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | Significant emphysema | No significant emphysema | No significant emphysema | yes |
| 4302294 | Significant emphysema | GOLD 1 (Mild) | Significant emphysema | No significant emphysema | No significant emphysema | No significant emphysema | no |
| 4710629 | No significant emphysema | GOLD 2 (Moderate) | No significant emphysema | Significant emphysema | Significant emphysema | Significant emphysema | no |
| 4796667 | Significant emphysema | GOLD 2 (Moderate) | No significant emphysema | Significant emphysema | No significant emphysema | No significant emphysema | no |
| 4996166 | Significant emphysema | GOLD 1 (Mild) | Significant emphysema | No significant emphysema | No significant emphysema | No significant emphysema | no |
| 6858508 | No significant emphysema | GOLD 1 (Mild) | Significant emphysema | No significant emphysema | No significant emphysema | No significant emphysema | yes |
| 7871759 | No significant emphysema | GOLD 1 (Mild) | Significant emphysema | No significant emphysema | No significant emphysema | No significant emphysema | yes |
| 8404129 | Significant emphysema | GOLD 3 (Severe) | No significant emphysema | Significant emphysema | No significant emphysema | No significant emphysema | no |
| 8704416 | Significant emphysema | GOLD 3 (Severe) | No significant emphysema | Significant emphysema | Significant emphysema | Significant emphysema | yes |
| 9075311 | Significant emphysema | GOLD 2 (Moderate) | No significant emphysema | Significant emphysema | No significant emphysema | No significant emphysema | no |
| A613117 | Significant emphysema | GOLD 2 (Moderate) | No significant emphysema | Significant emphysema | Significant emphysema | Significant emphysema | yes |
| C041635 | Significant emphysema | GOLD 1 (Mild) | Significant emphysema | No significant emphysema | No significant emphysema | No significant emphysema | no |
| D132855 | Significant emphysema | GOLD 2 (Moderate) | No significant emphysema | Significant emphysema | Significant emphysema | Significant emphysema | yes |
| E717248 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | Significant emphysema | No significant emphysema | No significant emphysema | yes |

### Vote errors (11)

| patient_id | true | GOLD | image | fusion | quant | vote | margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2221276 | Significant emphysema | GOLD 1 (Mild) | No significant emphysema | No significant emphysema | Significant emphysema | No significant emphysema | 2-1 |
| 2256243 | No significant emphysema | GOLD 1 (Mild) | Significant emphysema | Significant emphysema | Significant emphysema | Significant emphysema | 3-0 |
| 2500824 | No significant emphysema | GOLD 1 (Mild) | Significant emphysema | Significant emphysema | No significant emphysema | Significant emphysema | 2-1 |
| 4302294 | Significant emphysema | GOLD 1 (Mild) | Significant emphysema | No significant emphysema | No significant emphysema | No significant emphysema | 2-1 |
| 4710629 | No significant emphysema | GOLD 2 (Moderate) | No significant emphysema | Significant emphysema | Significant emphysema | Significant emphysema | 2-1 |
| 4796667 | Significant emphysema | GOLD 2 (Moderate) | No significant emphysema | Significant emphysema | No significant emphysema | No significant emphysema | 2-1 |
| 4996166 | Significant emphysema | GOLD 1 (Mild) | Significant emphysema | No significant emphysema | No significant emphysema | No significant emphysema | 2-1 |
| 5630846 | Significant emphysema | GOLD 3 (Severe) | No significant emphysema | No significant emphysema | No significant emphysema | No significant emphysema | 3-0 |
| 8404129 | Significant emphysema | GOLD 3 (Severe) | No significant emphysema | Significant emphysema | No significant emphysema | No significant emphysema | 2-1 |
| 9075311 | Significant emphysema | GOLD 2 (Moderate) | No significant emphysema | Significant emphysema | No significant emphysema | No significant emphysema | 2-1 |
| C041635 | Significant emphysema | GOLD 1 (Mild) | Significant emphysema | No significant emphysema | No significant emphysema | No significant emphysema | 2-1 |

## Per-patient table

| patient_id | fold | quant fold | true | GOLD | image pred | image p(pos) | fusion pred | fusion p(pos) | quant pred | quant p(pos) | vote | margin | vote correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0781915 | 5 | 5 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.002 | No significant emphysema | 0.041 | No significant emphysema | 3-0 | yes |
| 1261736 | 3 | 1 | Significant emphysema | GOLD 4 (Very Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.974 | Significant emphysema | 3-0 | yes |
| 1596038 | 2 | 5 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.146 | No significant emphysema | 0.392 | No significant emphysema | 3-0 | yes |
| 1604378 | 2 | 4 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.496 | No significant emphysema | 0.493 | No significant emphysema | 3-0 | yes |
| 1663485 | 4 | 3 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.000 | No significant emphysema | 0.309 | No significant emphysema | 3-0 | yes |
| 1687031 | 3 | 4 | Significant emphysema | GOLD 3 (Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.973 | Significant emphysema | 3-0 | yes |
| 1746380 | 3 | 3 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | Significant emphysema ✗ | 1.000 | No significant emphysema | 0.393 | No significant emphysema | 2-1 | yes |
| 1800944 | 4 | 1 | Significant emphysema | GOLD 3 (Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | No significant emphysema ✗ | 0.442 | Significant emphysema | 2-1 | yes |
| 1814107 | 5 | 5 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.136 | No significant emphysema | 0.185 | No significant emphysema | 3-0 | yes |
| 2094528 | 2 | 4 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.000 | No significant emphysema | 0.054 | No significant emphysema | 3-0 | yes |
| 2221276 | 4 | 2 | Significant emphysema | GOLD 1 (Mild) | No significant emphysema ✗ | 0.000 | No significant emphysema ✗ | 0.000 | Significant emphysema | 0.666 | No significant emphysema | 2-1 | no |
| 2256243 | 5 | 4 | No significant emphysema | GOLD 1 (Mild) | Significant emphysema ✗ | 1.000 | Significant emphysema ✗ | 0.985 | Significant emphysema ✗ | 0.832 | Significant emphysema | 3-0 | no |
| 2291134 | 4 | 2 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.001 | No significant emphysema | 0.000 | No significant emphysema | 0.051 | No significant emphysema | 3-0 | yes |
| 2500824 | 3 | 3 | No significant emphysema | GOLD 1 (Mild) | Significant emphysema ✗ | 1.000 | Significant emphysema ✗ | 0.998 | No significant emphysema | 0.220 | Significant emphysema | 2-1 | no |
| 2588424 | 2 | 4 | Significant emphysema | GOLD 4 (Very Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.915 | Significant emphysema | 3-0 | yes |
| 2860903 | 1 | 2 | No significant emphysema | GOLD 1 (Mild) | Significant emphysema ✗ | 1.000 | No significant emphysema | 0.066 | No significant emphysema | 0.080 | No significant emphysema | 2-1 | yes |
| 2991621 | 4 | 3 | Significant emphysema | GOLD 4 (Very Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.896 | Significant emphysema | 3-0 | yes |
| 3097765 | 1 | 2 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.005 | Significant emphysema ✗ | 0.996 | No significant emphysema | 0.018 | No significant emphysema | 2-1 | yes |
| 3635301 | 1 | 5 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.072 | No significant emphysema | 0.000 | No significant emphysema | 0.210 | No significant emphysema | 3-0 | yes |
| 3647457 | 4 | 2 | Significant emphysema | GOLD 3 (Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.987 | Significant emphysema | 3-0 | yes |
| 4204917 | 4 | 4 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.006 | No significant emphysema | 0.037 | No significant emphysema | 3-0 | yes |
| 4205212 | 4 | 2 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.000 | No significant emphysema | 0.036 | No significant emphysema | 3-0 | yes |
| 4230847 | 2 | 3 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.013 | No significant emphysema | 0.145 | No significant emphysema | 3-0 | yes |
| 4302294 | 5 | 2 | Significant emphysema | GOLD 1 (Mild) | Significant emphysema | 0.797 | No significant emphysema ✗ | 0.267 | No significant emphysema ✗ | 0.055 | No significant emphysema | 2-1 | no |
| 4372708 | 2 | 4 | Significant emphysema | GOLD 2 (Moderate) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.965 | Significant emphysema | 3-0 | yes |
| 4710629 | 1 | 1 | No significant emphysema | GOLD 2 (Moderate) | No significant emphysema | 0.034 | Significant emphysema ✗ | 1.000 | Significant emphysema ✗ | 0.960 | Significant emphysema | 2-1 | no |
| 4796667 | 5 | 5 | Significant emphysema | GOLD 2 (Moderate) | No significant emphysema ✗ | 0.001 | Significant emphysema | 0.969 | No significant emphysema ✗ | 0.070 | No significant emphysema | 2-1 | no |
| 4996166 | 2 | 5 | Significant emphysema | GOLD 1 (Mild) | Significant emphysema | 0.501 | No significant emphysema ✗ | 0.409 | No significant emphysema ✗ | 0.062 | No significant emphysema | 2-1 | no |
| 5046455 | 1 | 2 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.000 | No significant emphysema | 0.122 | No significant emphysema | 3-0 | yes |
| 5127217 | 1 | 5 | Significant emphysema | GOLD 2 (Moderate) | Significant emphysema | 1.000 | Significant emphysema | 0.608 | Significant emphysema | 0.619 | Significant emphysema | 3-0 | yes |
| 5390303 | 5 | 1 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.032 | No significant emphysema | 0.123 | No significant emphysema | 3-0 | yes |
| 5630846 | 1 | 2 | Significant emphysema | GOLD 3 (Severe) | No significant emphysema ✗ | 0.000 | No significant emphysema ✗ | 0.000 | No significant emphysema ✗ | 0.067 | No significant emphysema | 3-0 | no |
| 5925853 | 1 | 5 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.000 | Significant emphysema ✗ | 0.841 | No significant emphysema | 2-1 | yes |
| 6212308 | 1 | 1 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.004 | No significant emphysema | 0.167 | No significant emphysema | 3-0 | yes |
| 6312603 | 4 | 3 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.001 | No significant emphysema | 0.000 | No significant emphysema | 0.122 | No significant emphysema | 3-0 | yes |
| 6757504 | 4 | 3 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.474 | No significant emphysema | 0.000 | No significant emphysema | 0.179 | No significant emphysema | 3-0 | yes |
| 6858508 | 2 | 1 | No significant emphysema | GOLD 1 (Mild) | Significant emphysema ✗ | 1.000 | No significant emphysema | 0.042 | No significant emphysema | 0.150 | No significant emphysema | 2-1 | yes |
| 6887256 | 3 | 1 | Significant emphysema | GOLD 3 (Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.875 | Significant emphysema | 3-0 | yes |
| 7871759 | 3 | 2 | No significant emphysema | GOLD 1 (Mild) | Significant emphysema ✗ | 1.000 | No significant emphysema | 0.000 | No significant emphysema | 0.340 | No significant emphysema | 2-1 | yes |
| 8009284 | 2 | 3 | Significant emphysema | GOLD 3 (Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.733 | Significant emphysema | 3-0 | yes |
| 8126939 | 3 | 4 | Significant emphysema | GOLD 2 (Moderate) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.949 | Significant emphysema | 3-0 | yes |
| 8244460 | 5 | 2 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.024 | No significant emphysema | 0.149 | No significant emphysema | 3-0 | yes |
| 8332556 | 3 | 4 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.002 | No significant emphysema | 0.000 | No significant emphysema | 0.284 | No significant emphysema | 3-0 | yes |
| 8404129 | 1 | 4 | Significant emphysema | GOLD 3 (Severe) | No significant emphysema ✗ | 0.015 | Significant emphysema | 1.000 | No significant emphysema ✗ | 0.156 | No significant emphysema | 2-1 | no |
| 8704416 | 2 | 1 | Significant emphysema | GOLD 3 (Severe) | No significant emphysema ✗ | 0.000 | Significant emphysema | 0.907 | Significant emphysema | 0.569 | Significant emphysema | 2-1 | yes |
| 9075311 | 2 | 3 | Significant emphysema | GOLD 2 (Moderate) | No significant emphysema ✗ | 0.000 | Significant emphysema | 1.000 | No significant emphysema ✗ | 0.274 | No significant emphysema | 2-1 | no |
| 9529629 | 2 | 4 | Significant emphysema | GOLD 2 (Moderate) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.961 | Significant emphysema | 3-0 | yes |
| A267542 | 3 | 1 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.000 | Significant emphysema ✗ | 0.616 | No significant emphysema | 2-1 | yes |
| A613117 | 3 | 2 | Significant emphysema | GOLD 2 (Moderate) | No significant emphysema ✗ | 0.000 | Significant emphysema | 0.998 | Significant emphysema | 0.789 | Significant emphysema | 2-1 | yes |
| A754735 | 5 | 1 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.001 | No significant emphysema | 0.061 | No significant emphysema | 0.206 | No significant emphysema | 3-0 | yes |
| A762364 | 1 | 1 | Significant emphysema | GOLD 2 (Moderate) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.883 | Significant emphysema | 3-0 | yes |
| B213449 | 1 | 5 | Significant emphysema | GOLD 4 (Very Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.944 | Significant emphysema | 3-0 | yes |
| C041635 | 5 | 1 | Significant emphysema | GOLD 1 (Mild) | Significant emphysema | 1.000 | No significant emphysema ✗ | 0.313 | No significant emphysema ✗ | 0.333 | No significant emphysema | 2-1 | no |
| C081146 | 3 | 1 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.000 | No significant emphysema | 0.156 | No significant emphysema | 3-0 | yes |
| C435832 | 1 | 1 | Significant emphysema | GOLD 3 (Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.827 | Significant emphysema | 3-0 | yes |
| C543831 | 5 | 3 | Significant emphysema | GOLD 1 (Mild) | Significant emphysema | 1.000 | Significant emphysema | 0.975 | No significant emphysema ✗ | 0.166 | Significant emphysema | 2-1 | yes |
| C586742 | 4 | 4 | Significant emphysema | GOLD 3 (Severe) | Significant emphysema | 0.648 | Significant emphysema | 1.000 | Significant emphysema | 0.874 | Significant emphysema | 3-0 | yes |
| C905524 | 5 | 2 | Significant emphysema | GOLD 3 (Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.995 | Significant emphysema | 3-0 | yes |
| D132855 | 4 | 3 | Significant emphysema | GOLD 2 (Moderate) | No significant emphysema ✗ | 0.142 | Significant emphysema | 1.000 | Significant emphysema | 0.867 | Significant emphysema | 2-1 | yes |
| D550510 | 3 | 5 | Significant emphysema | GOLD 2 (Moderate) | Significant emphysema | 1.000 | Significant emphysema | 0.966 | Significant emphysema | 0.998 | Significant emphysema | 3-0 | yes |
| E353272 | 4 | 5 | Significant emphysema | GOLD 4 (Very Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.931 | Significant emphysema | 3-0 | yes |
| E558113 | 1 | 5 | Significant emphysema | GOLD 4 (Very Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.988 | Significant emphysema | 3-0 | yes |
| E647833 | 3 | 3 | Significant emphysema | GOLD 3 (Severe) | Significant emphysema | 1.000 | Significant emphysema | 1.000 | Significant emphysema | 0.985 | Significant emphysema | 3-0 | yes |
| E717248 | 5 | 5 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | Significant emphysema ✗ | 0.866 | No significant emphysema | 0.080 | No significant emphysema | 2-1 | yes |
| E771850 | 5 | 3 | Significant emphysema | GOLD 3 (Severe) | Significant emphysema | 1.000 | Significant emphysema | 0.987 | Significant emphysema | 0.850 | Significant emphysema | 3-0 | yes |
| E797258 | 2 | 4 | No significant emphysema | GOLD 1 (Mild) | No significant emphysema | 0.000 | No significant emphysema | 0.015 | No significant emphysema | 0.345 | No significant emphysema | 3-0 | yes |
