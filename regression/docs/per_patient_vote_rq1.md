# Per-patient predictions and hard-vote result

- Generated: 2026-07-28T13:32:05
- Cohort: 66 patients
- Vote: hard majority over 3 models (image, fusion, quant) — 三票取多數,每位病人皆有結論,不再有 pending

> **Caveat.** `image` / `fusion` 的 per-patient 預測存在每折的 best epoch,而該 epoch 是在 test fold 上挑的(`trainer.py` 只在 fold 分數改善時呼叫 `save_predictions`),因此這兩列是樂觀上界。`quant` 為固定 100 epoch 的最後一個 epoch、沒有用 test fold 選停點,無此偏差 —— 它的 pooled out-of-fold accuracy 0.848 與 `result.md` 的 5-fold 平均一致。`vote` 列因為含有 image / fusion,同樣繼承了這份樂觀性。

## Sources

- `image`: `figures/RQ1_normal_v_abnormal/hybrid_mamba_attention_rq1_normal_v_abnormal_image_2026-07-14_15:15:09`
- `fusion`: `figures/RQ1_normal_v_abnormal/hybrid_mamba_tapct_fusion_rq1_normal_v_abnormal_fusion_2026-07-14_15:54:16`
- `quant`: `models/rq_quant/param_aeropath__rq1_per_patient.json (export_quant_per_patient_rq1.py, param_aeropath, seed=42, 5-fold, 100 epochs, early stopping off)`

`quant` = result.md 中「量化(12 特徵+FCNN)」那一列的同一組設定(`param_aeropath` 特徵、seed=42、5-fold stratified、100 epochs、early stopping 關閉、class-weighted CE、per-fold StandardScaler、argmax)。

> 註:`fold` 欄是深度模型的折號,`quant_fold` 是量化模型自己的折號。兩者都是 seed=42 的 5-fold stratified,但世代排序不同故分組不同;per-patient 投票與折號無關,不受影響。

## Summary

| model | n | Accuracy | Sensitivity (Abnormal) | Specificity (Normal) | TP | FN | TN | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image | 66 | 0.8788 | 0.8485 | 0.9091 | 28 | 5 | 30 | 3 |
| fusion | 66 | 0.8636 | 0.8485 | 0.8788 | 28 | 5 | 29 | 4 |
| quant | 66 | 0.8485 | 0.7879 | 0.9091 | 26 | 7 | 30 | 3 |
| **vote** | 66 | 0.8939 | 0.8485 | 0.9394 | 28 | 5 | 31 | 2 |

補上第三個模型後,vote 涵蓋全部 66 位病人,與三個單模型列在同一世代上,可直接比較。
投票結構:一致 3-0 共 50 位(正確 48,96.0%);分歧 2-1 共 16 位(正確 11,68.8%)—— 模型一致時幾乎必對,錯誤集中在分歧的少數個案。

### 先前 pending 的 11 位 —— 現由 `quant` 打破平手

這些正是 image / fusion 互相矛盾的個案。quant 的一票讓其中 **7/11** 投對(63.6%)。

| patient_id | true | GOLD | image | fusion | quant | vote | correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1746380 | Normal | GOLD 1 (Mild) | Normal | Abnormal | Normal | Normal | yes |
| 2500824 | Normal | GOLD 1 (Mild) | Abnormal | Normal | Normal | Normal | yes |
| 4796667 | Abnormal | GOLD 2 (Moderate) | Abnormal | Normal | Normal | Normal | no |
| 5127217 | Abnormal | GOLD 2 (Moderate) | Abnormal | Normal | Normal | Normal | no |
| 6212308 | Normal | GOLD 1 (Mild) | Normal | Abnormal | Normal | Normal | yes |
| 9075311 | Abnormal | GOLD 2 (Moderate) | Normal | Abnormal | Abnormal | Abnormal | yes |
| A613117 | Abnormal | GOLD 2 (Moderate) | Abnormal | Normal | Abnormal | Abnormal | yes |
| C041635 | Abnormal | GOLD 1 (Mild) | Abnormal | Normal | Normal | Normal | no |
| C586742 | Abnormal | GOLD 3 (Severe) | Normal | Abnormal | Abnormal | Abnormal | yes |
| C905524 | Abnormal | GOLD 3 (Severe) | Normal | Abnormal | Abnormal | Abnormal | yes |
| E797258 | Abnormal | GOLD 1 (Mild) | Normal | Abnormal | Normal | Normal | no |

### Vote 判錯的 7 位

| patient_id | true | GOLD | image | fusion | quant | vote | margin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2256243 | Normal | GOLD 1 (Mild) | Abnormal | Abnormal | Abnormal | Abnormal | 3-0 |
| 4796667 | Abnormal | GOLD 2 (Moderate) | Abnormal | Normal | Normal | Normal | 2-1 |
| 4996166 | Normal | GOLD 1 (Mild) | Abnormal | Abnormal | Normal | Abnormal | 2-1 |
| 5127217 | Abnormal | GOLD 2 (Moderate) | Abnormal | Normal | Normal | Normal | 2-1 |
| 5630846 | Abnormal | GOLD 3 (Severe) | Normal | Normal | Normal | Normal | 3-0 |
| C041635 | Abnormal | GOLD 1 (Mild) | Abnormal | Normal | Normal | Normal | 2-1 |
| E797258 | Abnormal | GOLD 1 (Mild) | Normal | Abnormal | Normal | Normal | 2-1 |

## Per-patient table

| patient_id | fold | quant fold | true | GOLD | image pred | image p(Abn) | fusion pred | fusion p(Abn) | quant pred | quant p(Abn) | vote | margin | vote correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0781915 | 2 | 4 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.004 | Normal | 3-0 | yes |
| 1261736 | 5 | 2 | Abnormal | GOLD 4 (Very Severe) | Abnormal | 1.000 | Abnormal | 0.999 | Abnormal | 0.985 | Abnormal | 3-0 | yes |
| 1596038 | 2 | 5 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.154 | Normal | 3-0 | yes |
| 1604378 | 1 | 5 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.002 | Normal | 0.200 | Normal | 3-0 | yes |
| 1663485 | 4 | 4 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.002 | Normal | 0.166 | Normal | 3-0 | yes |
| 1687031 | 3 | 2 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.978 | Abnormal | 3-0 | yes |
| 1746380 | 1 | 3 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Abnormal ✗ | 0.999 | Normal | 0.137 | Normal | 2-1 | yes |
| 1800944 | 4 | 1 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Normal ✗ | 0.213 | Abnormal | 2-1 | yes |
| 1814107 | 4 | 3 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.217 | Normal | 3-0 | yes |
| 2094528 | 3 | 4 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.003 | Normal | 3-0 | yes |
| 2221276 | 2 | 3 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Abnormal ✗ | 0.669 | Normal | 2-1 | yes |
| 2256243 | 4 | 4 | Normal | GOLD 1 (Mild) | Abnormal ✗ | 1.000 | Abnormal ✗ | 1.000 | Abnormal ✗ | 0.845 | Abnormal | 3-0 | no |
| 2291134 | 5 | 1 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.017 | Normal | 0.056 | Normal | 3-0 | yes |
| 2500824 | 5 | 2 | Normal | GOLD 1 (Mild) | Abnormal ✗ | 1.000 | Normal | 0.032 | Normal | 0.127 | Normal | 2-1 | yes |
| 2588424 | 3 | 4 | Abnormal | GOLD 4 (Very Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.954 | Abnormal | 3-0 | yes |
| 2860903 | 2 | 2 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.091 | Normal | 3-0 | yes |
| 2991621 | 2 | 1 | Abnormal | GOLD 4 (Very Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.933 | Abnormal | 3-0 | yes |
| 3097765 | 1 | 3 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.208 | Normal | 3-0 | yes |
| 3635301 | 3 | 4 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.297 | Normal | 0.318 | Normal | 3-0 | yes |
| 3647457 | 2 | 4 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.977 | Abnormal | 3-0 | yes |
| 4204917 | 5 | 2 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.099 | Normal | 0.047 | Normal | 3-0 | yes |
| 4205212 | 4 | 3 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.194 | Normal | 3-0 | yes |
| 4230847 | 1 | 5 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.154 | Normal | 0.064 | Normal | 3-0 | yes |
| 4302294 | 3 | 1 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.083 | Normal | 3-0 | yes |
| 4372708 | 3 | 3 | Abnormal | GOLD 2 (Moderate) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.955 | Abnormal | 3-0 | yes |
| 4710629 | 4 | 2 | Abnormal | GOLD 2 (Moderate) | Abnormal | 0.662 | Abnormal | 1.000 | Abnormal | 0.891 | Abnormal | 3-0 | yes |
| 4796667 | 2 | 4 | Abnormal | GOLD 2 (Moderate) | Abnormal | 0.994 | Normal ✗ | 0.000 | Normal ✗ | 0.021 | Normal | 2-1 | no |
| 4996166 | 4 | 2 | Normal | GOLD 1 (Mild) | Abnormal ✗ | 1.000 | Abnormal ✗ | 0.884 | Normal | 0.123 | Abnormal | 2-1 | no |
| 5046455 | 2 | 1 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.107 | Normal | 3-0 | yes |
| 5127217 | 1 | 5 | Abnormal | GOLD 2 (Moderate) | Abnormal | 1.000 | Normal ✗ | 0.015 | Normal ✗ | 0.073 | Normal | 2-1 | no |
| 5390303 | 1 | 3 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.059 | Normal | 3-0 | yes |
| 5630846 | 1 | 5 | Abnormal | GOLD 3 (Severe) | Normal ✗ | 0.000 | Normal ✗ | 0.000 | Normal ✗ | 0.040 | Normal | 3-0 | no |
| 5925853 | 5 | 2 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.336 | Abnormal ✗ | 0.944 | Normal | 2-1 | yes |
| 6212308 | 1 | 1 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Abnormal ✗ | 0.912 | Normal | 0.118 | Normal | 2-1 | yes |
| 6312603 | 1 | 2 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.004 | Normal | 0.124 | Normal | 3-0 | yes |
| 6757504 | 4 | 3 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.377 | Normal | 3-0 | yes |
| 6858508 | 4 | 2 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.176 | Normal | 3-0 | yes |
| 6887256 | 3 | 2 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.941 | Abnormal | 3-0 | yes |
| 7871759 | 2 | 4 | Normal | GOLD 1 (Mild) | Normal | 0.002 | Normal | 0.000 | Normal | 0.462 | Normal | 3-0 | yes |
| 8009284 | 1 | 1 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.841 | Abnormal | 3-0 | yes |
| 8126939 | 2 | 3 | Abnormal | GOLD 2 (Moderate) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.978 | Abnormal | 3-0 | yes |
| 8244460 | 3 | 1 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.153 | Normal | 3-0 | yes |
| 8332556 | 5 | 1 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.236 | Normal | 0.070 | Normal | 3-0 | yes |
| 8404129 | 2 | 5 | Abnormal | GOLD 3 (Severe) | Abnormal | 0.999 | Abnormal | 1.000 | Abnormal | 0.689 | Abnormal | 3-0 | yes |
| 8704416 | 4 | 4 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 0.989 | Abnormal | 0.796 | Abnormal | 3-0 | yes |
| 9075311 | 1 | 1 | Abnormal | GOLD 2 (Moderate) | Normal ✗ | 0.000 | Abnormal | 1.000 | Abnormal | 0.850 | Abnormal | 2-1 | yes |
| 9529629 | 1 | 3 | Abnormal | GOLD 2 (Moderate) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.854 | Abnormal | 3-0 | yes |
| A267542 | 3 | 5 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.000 | Normal | 0.244 | Normal | 3-0 | yes |
| A613117 | 5 | 4 | Abnormal | GOLD 2 (Moderate) | Abnormal | 1.000 | Normal ✗ | 0.241 | Abnormal | 0.890 | Abnormal | 2-1 | yes |
| A754735 | 3 | 5 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.006 | Normal | 0.115 | Normal | 3-0 | yes |
| A762364 | 1 | 2 | Abnormal | GOLD 2 (Moderate) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.922 | Abnormal | 3-0 | yes |
| B213449 | 5 | 1 | Abnormal | GOLD 4 (Very Severe) | Abnormal | 1.000 | Abnormal | 0.993 | Abnormal | 0.897 | Abnormal | 3-0 | yes |
| C041635 | 4 | 5 | Abnormal | GOLD 1 (Mild) | Abnormal | 0.525 | Normal ✗ | 0.006 | Normal ✗ | 0.218 | Normal | 2-1 | no |
| C081146 | 5 | 1 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.010 | Normal | 0.125 | Normal | 3-0 | yes |
| C435832 | 5 | 1 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 0.968 | Abnormal | 0.914 | Abnormal | 3-0 | yes |
| C543831 | 4 | 1 | Abnormal | GOLD 1 (Mild) | Abnormal | 1.000 | Abnormal | 0.988 | Normal ✗ | 0.090 | Abnormal | 2-1 | yes |
| C586742 | 3 | 4 | Abnormal | GOLD 3 (Severe) | Normal ✗ | 0.000 | Abnormal | 1.000 | Abnormal | 0.930 | Abnormal | 2-1 | yes |
| C905524 | 4 | 4 | Abnormal | GOLD 3 (Severe) | Normal ✗ | 0.000 | Abnormal | 1.000 | Abnormal | 0.985 | Abnormal | 2-1 | yes |
| D132855 | 1 | 2 | Abnormal | GOLD 2 (Moderate) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.963 | Abnormal | 3-0 | yes |
| D550510 | 3 | 3 | Abnormal | GOLD 2 (Moderate) | Abnormal | 1.000 | Abnormal | 0.853 | Abnormal | 0.999 | Abnormal | 3-0 | yes |
| E353272 | 5 | 5 | Abnormal | GOLD 4 (Very Severe) | Abnormal | 1.000 | Abnormal | 0.993 | Abnormal | 0.905 | Abnormal | 3-0 | yes |
| E558113 | 2 | 3 | Abnormal | GOLD 4 (Very Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.905 | Abnormal | 3-0 | yes |
| E647833 | 2 | 3 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 1.000 | Abnormal | 0.997 | Abnormal | 3-0 | yes |
| E717248 | 5 | 5 | Normal | GOLD 1 (Mild) | Normal | 0.000 | Normal | 0.085 | Normal | 0.061 | Normal | 3-0 | yes |
| E771850 | 3 | 5 | Abnormal | GOLD 3 (Severe) | Abnormal | 1.000 | Abnormal | 0.973 | Abnormal | 0.678 | Abnormal | 3-0 | yes |
| E797258 | 5 | 5 | Abnormal | GOLD 1 (Mild) | Normal ✗ | 0.000 | Abnormal | 0.610 | Normal ✗ | 0.410 | Normal | 2-1 | no |
