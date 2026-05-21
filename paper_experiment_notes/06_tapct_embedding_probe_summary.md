# TAP-CT Embedding Probe Summary

這份表格把 `regression/figures/TAPCT_embedding_probes/*/results.json` 裡每個 target/model 組合攤平成論文可用的比較表。

| run | target | model | n | mean_accuracy | std_accuracy | mean_macro_f1 | std_macro_f1 | mean_bal_acc | mean_mae | mean_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [2026-05-13_13-59-34](runs/tapct_embedding_probes/2026-05-13_13-59-34.md) | angle_3class | angle_ridge_threshold | 66 | 0.66703 | 0.09019 | 0.45669 | 0.1128 | 0.5163 |  |  |
| [tapct_b_3d_2026-05-13_14-06-00](runs/tapct_embedding_probes/tapct_b_3d_2026-05-13_14-06-00.md) | angle_3class | angle_ridge_threshold | 66 | 0.62088 | 0.15424 | 0.43972 | 0.14526 | 0.42 |  |  |
| [tapct_b_3d_yaml](runs/tapct_embedding_probes/tapct_b_3d_yaml.md) | angle_3class | angle_ridge_threshold | 66 | 0.62088 | 0.15424 | 0.43972 | 0.14526 | 0.42 |  |  |
| [tapct_s_2_5d_yaml](runs/tapct_embedding_probes/tapct_s_2_5d_yaml.md) | angle_3class | angle_ridge_threshold | 66 | 0.6967 | 0.06936 | 0.46137 | 0.09487 | 0.53407 |  |  |
| [2026-05-13_13-59-34](runs/tapct_embedding_probes/2026-05-13_13-59-34.md) | angle_3class | linear_svm | 66 | 0.63627 | 0.08944 | 0.44722 | 0.06581 | 0.48519 |  |  |
| [tapct_b_3d_2026-05-13_14-06-00](runs/tapct_embedding_probes/tapct_b_3d_2026-05-13_14-06-00.md) | angle_3class | linear_svm | 66 | 0.63626 | 0.11284 | 0.52015 | 0.09051 | 0.6363 |  |  |
| [tapct_b_3d_yaml](runs/tapct_embedding_probes/tapct_b_3d_yaml.md) | angle_3class | linear_svm | 66 | 0.63626 | 0.11284 | 0.52015 | 0.09051 | 0.6363 |  |  |
| [tapct_s_2_5d_yaml](runs/tapct_embedding_probes/tapct_s_2_5d_yaml.md) | angle_3class | linear_svm | 66 | 0.6989 | 0.08725 | 0.52996 | 0.11937 | 0.58963 |  |  |
| [2026-05-13_13-59-34](runs/tapct_embedding_probes/2026-05-13_13-59-34.md) | angle_3class | logistic | 66 | 0.76044 | 0.11172 | 0.48702 | 0.12888 | 0.51259 |  |  |
| [tapct_b_3d_2026-05-13_14-06-00](runs/tapct_embedding_probes/tapct_b_3d_2026-05-13_14-06-00.md) | angle_3class | logistic | 66 | 0.78901 | 0.05078 | 0.4776 | 0.10829 | 0.51185 |  |  |
| [tapct_b_3d_yaml](runs/tapct_embedding_probes/tapct_b_3d_yaml.md) | angle_3class | logistic | 66 | 0.78901 | 0.05078 | 0.4776 | 0.10829 | 0.51185 |  |  |
| [tapct_s_2_5d_yaml](runs/tapct_embedding_probes/tapct_s_2_5d_yaml.md) | angle_3class | logistic | 66 | 0.75824 | 0.05338 | 0.48673 | 0.11553 | 0.51259 |  |  |
| [2026-05-13_13-59-34](runs/tapct_embedding_probes/2026-05-13_13-59-34.md) | angle_3class | ordinal_logistic | 66 | 0.77473 | 0.08887 | 0.49966 | 0.12429 | 0.51926 |  |  |
| [tapct_b_3d_2026-05-13_14-06-00](runs/tapct_embedding_probes/tapct_b_3d_2026-05-13_14-06-00.md) | angle_3class | ordinal_logistic | 66 | 0.75824 | 0.05338 | 0.47235 | 0.10205 | 0.49704 |  |  |
| [tapct_b_3d_yaml](runs/tapct_embedding_probes/tapct_b_3d_yaml.md) | angle_3class | ordinal_logistic | 66 | 0.75824 | 0.05338 | 0.47235 | 0.10205 | 0.49704 |  |  |
| [tapct_s_2_5d_yaml](runs/tapct_embedding_probes/tapct_s_2_5d_yaml.md) | angle_3class | ordinal_logistic | 66 | 0.75824 | 0.08708 | 0.48834 | 0.11796 | 0.51111 |  |  |
| [2026-05-13_13-59-34](runs/tapct_embedding_probes/2026-05-13_13-59-34.md) | angle_3class | ridge_classifier | 66 | 0.79011 | 0.09314 | 0.50352 | 0.126 | 0.52667 |  |  |
| [tapct_b_3d_2026-05-13_14-06-00](runs/tapct_embedding_probes/tapct_b_3d_2026-05-13_14-06-00.md) | angle_3class | ridge_classifier | 66 | 0.75824 | 0.02198 | 0.44199 | 0.08684 | 0.47111 |  |  |
| [tapct_b_3d_yaml](runs/tapct_embedding_probes/tapct_b_3d_yaml.md) | angle_3class | ridge_classifier | 66 | 0.75824 | 0.02198 | 0.44199 | 0.08684 | 0.47111 |  |  |
| [tapct_s_2_5d_yaml](runs/tapct_embedding_probes/tapct_s_2_5d_yaml.md) | angle_3class | ridge_classifier | 66 | 0.77363 | 0.09417 | 0.48344 | 0.12529 | 0.51852 |  |  |
| [2026-05-13_13-59-34](runs/tapct_embedding_probes/2026-05-13_13-59-34.md) | angle_binary_extreme | linear_svm | 61 | 0.75385 | 0.10569 | 0.7027 | 0.10065 | 0.76889 |  |  |
| [tapct_b_3d_2026-05-13_14-06-00](runs/tapct_embedding_probes/tapct_b_3d_2026-05-13_14-06-00.md) | angle_binary_extreme | linear_svm | 61 | 0.7359 | 0.14524 | 0.71539 | 0.14106 | 0.82889 |  |  |
| [tapct_b_3d_yaml](runs/tapct_embedding_probes/tapct_b_3d_yaml.md) | angle_binary_extreme | linear_svm | 61 | 0.7359 | 0.14524 | 0.71539 | 0.14106 | 0.82889 |  |  |
| [tapct_s_2_5d_yaml](runs/tapct_embedding_probes/tapct_s_2_5d_yaml.md) | angle_binary_extreme | linear_svm | 61 | 0.83462 | 0.09249 | 0.8064 | 0.07969 | 0.86778 |  |  |
| [2026-05-13_13-59-34](runs/tapct_embedding_probes/2026-05-13_13-59-34.md) | angle_binary_extreme | logistic | 61 | 0.87051 | 0.0799 | 0.77975 | 0.18837 | 0.8 |  |  |
| [tapct_b_3d_2026-05-13_14-06-00](runs/tapct_embedding_probes/tapct_b_3d_2026-05-13_14-06-00.md) | angle_binary_extreme | logistic | 61 | 0.85385 | 0.05638 | 0.73893 | 0.16477 | 0.76778 |  |  |
| [tapct_b_3d_yaml](runs/tapct_embedding_probes/tapct_b_3d_yaml.md) | angle_binary_extreme | logistic | 61 | 0.85385 | 0.05638 | 0.73893 | 0.16477 | 0.76778 |  |  |
| [tapct_s_2_5d_yaml](runs/tapct_embedding_probes/tapct_s_2_5d_yaml.md) | angle_binary_extreme | logistic | 61 | 0.83718 | 0.07053 | 0.72582 | 0.15253 | 0.73889 |  |  |
| [2026-05-13_13-59-34](runs/tapct_embedding_probes/2026-05-13_13-59-34.md) | angle_binary_extreme | ridge_classifier | 61 | 0.88718 | 0.07909 | 0.79708 | 0.19447 | 0.81111 |  |  |
| [autoplot_smoke](runs/tapct_embedding_probes/autoplot_smoke.md) | angle_binary_extreme | ridge_classifier | 61 | 0.88718 | 0.07909 | 0.79708 | 0.19447 | 0.81111 |  |  |
| [tapct_b_3d_2026-05-13_14-06-00](runs/tapct_embedding_probes/tapct_b_3d_2026-05-13_14-06-00.md) | angle_binary_extreme | ridge_classifier | 61 | 0.83718 | 0.04686 | 0.70419 | 0.15038 | 0.71778 |  |  |
| [tapct_b_3d_yaml](runs/tapct_embedding_probes/tapct_b_3d_yaml.md) | angle_binary_extreme | ridge_classifier | 61 | 0.83718 | 0.04686 | 0.70419 | 0.15038 | 0.71778 |  |  |
| [tapct_s_2_5d_yaml](runs/tapct_embedding_probes/tapct_s_2_5d_yaml.md) | angle_binary_extreme | ridge_classifier | 61 | 0.85256 | 0.06189 | 0.77977 | 0.0592 | 0.77222 |  |  |
