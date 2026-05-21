# TAP-CT Embedding Probe: tapct_s_2_5d_yaml

- 原始 results.json: [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/results.json](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/results.json)
- features: `/home/felix/Research/nnMamba/regression/embeddings/tapct_s_2_5d/features.npz`
- metadata: `/home/felix/Research/nnMamba/regression/embeddings/tapct_s_2_5d/metadata.csv`
- n_splits: `5`
- seed: `42`
- ridge_alpha: `1.0`

## Probe Summary
| target | model | n | mean_accuracy | mean_macro_f1 | mean_bal_acc | mean_mae | mean_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| angle_3class | logistic | 66 | 0.75824 | 0.48673 | 0.51259 |  |  |
| angle_3class | linear_svm | 66 | 0.6989 | 0.52996 | 0.58963 |  |  |
| angle_3class | ridge_classifier | 66 | 0.77363 | 0.48344 | 0.51852 |  |  |
| angle_3class | ordinal_logistic | 66 | 0.75824 | 0.48834 | 0.51111 |  |  |
| angle_3class | angle_ridge_threshold | 66 | 0.6967 | 0.46137 | 0.53407 |  |  |
| angle_binary_extreme | logistic | 61 | 0.83718 | 0.72582 | 0.73889 |  |  |
| angle_binary_extreme | linear_svm | 61 | 0.83462 | 0.8064 | 0.86778 |  |  |
| angle_binary_extreme | ridge_classifier | 61 | 0.85256 | 0.77977 | 0.77222 |  |  |

## Related Files
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_angle_ridge_threshold_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_angle_ridge_threshold_class_recall.png) — 82.6 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_angle_ridge_threshold_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_angle_ridge_threshold_confusion_matrix.png) — 112.1 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_linear_svm_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_linear_svm_class_recall.png) — 82.9 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_linear_svm_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_linear_svm_confusion_matrix.png) — 107.0 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_logistic_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_logistic_class_recall.png) — 81.7 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_logistic_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_logistic_confusion_matrix.png) — 110.0 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_metric_comparison.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_metric_comparison.png) — 86.1 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_ordinal_logistic_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_ordinal_logistic_class_recall.png) — 81.5 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_ordinal_logistic_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_ordinal_logistic_confusion_matrix.png) — 108.8 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_ridge_classifier_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_ridge_classifier_class_recall.png) — 81.7 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_ridge_classifier_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_3class_ridge_classifier_confusion_matrix.png) — 110.7 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_linear_svm_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_linear_svm_class_recall.png) — 76.9 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_linear_svm_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_linear_svm_confusion_matrix.png) — 100.5 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_logistic_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_logistic_class_recall.png) — 75.9 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_logistic_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_logistic_confusion_matrix.png) — 100.7 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_metric_comparison.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_metric_comparison.png) — 76.7 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_ridge_classifier_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_ridge_classifier_class_recall.png) — 75.4 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_ridge_classifier_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/angle_binary_extreme_ridge_classifier_confusion_matrix.png) — 99.2 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/predictions.csv](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/predictions.csv) — 101.8 KB
- [regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/probe_metric_overview.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_s_2_5d_yaml/probe_metric_overview.png) — 82.4 KB

## Full results.json
```json
{
  "features": "/home/felix/Research/nnMamba/regression/embeddings/tapct_s_2_5d/features.npz",
  "metadata": "/home/felix/Research/nnMamba/regression/embeddings/tapct_s_2_5d/metadata.csv",
  "n_splits": 5,
  "seed": 42,
  "ridge_alpha": 1.0,
  "results": [
    {
      "target": "angle_3class",
      "model": "logistic",
      "class_names": [
        "Emphysema/Abnormal (<=131 deg)",
        "Intermediate (132-151 deg)",
        "Normal (>=152 deg)"
      ],
      "num_samples": 66,
      "class_counts": {
        "Emphysema/Abnormal (<=131 deg)": 14,
        "Intermediate (132-151 deg)": 5,
        "Normal (>=152 deg)": 47
      },
      "folds": [
        {
          "accuracy": 0.71429,
          "macro_f1": 0.27778,
          "macro_precision": 0.2381,
          "macro_recall": 0.33333,
          "balanced_accuracy": 0.33333,
          "confusion_matrix": [
            [
              0,
              0,
              3
            ],
            [
              0,
              0,
              1
            ],
            [
              0,
              0,
              10
            ]
          ],
          "fold": 1
        },
        {
          "accuracy": 0.69231,
          "macro_f1": 0.47222,
          "macro_precision": 0.45238,
          "macro_recall": 0.55556,
          "balanced_accuracy": 0.55556,
          "confusion_matrix": [
            [
              3,
              0,
              0
            ],
            [
              0,
              0,
              1
            ],
            [
              3,
              0,
              6
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.76923,
          "macro_f1": 0.50292,
          "macro_precision": 0.48889,
          "macro_recall": 0.51852,
          "balanced_accuracy": 0.51852,
          "confusion_matrix": [
            [
              2,
              0,
              1
            ],
            [
              0,
              0,
              1
            ],
            [
              1,
              0,
              8
            ]
          ],
          "fold": 3
        },
        {
          "accuracy": 0.84615,
          "macro_f1": 0.56667,
          "macro_precision": 0.60606,
          "macro_recall": 0.55556,
          "balanced_accuracy": 0.55556,
          "confusion_matrix": [
            [
              2,
              0,
              1
            ],
            [
              0,
              0,
              1
            ],
            [
              0,
              0,
              9
            ]
          ],
          "fold": 4
        },
        {
          "accuracy": 0.76923,
          "macro_f1": 0.61404,
          "macro_precision": 0.62963,
          "macro_recall": 0.6,
          "balanced_accuracy": 0.6,
          "confusion_matrix": [
            [
              2,
              0,
              0
            ],
            [
              0,
              0,
              1
            ],
            [
              0,
              2,
              8
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.75824,
        "std_accuracy": 0.05338,
        "mean_macro_f1": 0.48673,
        "std_macro_f1": 0.11553,
        "mean_macro_precision": 0.48301,
        "std_macro_precision": 0.13971,
        "mean_macro_recall": 0.51259,
        "std_macro_recall": 0.09328,
        "mean_balanced_accuracy": 0.51259,
        "std_balanced_accuracy": 0.09328
      },
      "combined": {
        "accuracy": 0.75758,
        "macro_f1": 0.50113,
        "macro_precision": 0.49874,
        "macro_recall": 0.50507,
        "balanced_accuracy": 0.50507,
        "confusion_matrix": [
          [
            9,
            0,
            5
          ],
          [
            0,
            0,
            5
          ],
          [
            4,
            2,
            41
          ]
        ]
      }
    },
    {
      "target": "angle_3class",
      "model": "linear_svm",
      "class_names": [
        "Emphysema/Abnormal (<=131 deg)",
        "Intermediate (132-151 deg)",
        "Normal (>=152 deg)"
      ],
      "num_samples": 66,
      "class_counts": {
        "Emphysema/Abnormal (<=131 deg)": 14,
        "Intermediate (132-151 deg)": 5,
        "Normal (>=152 deg)": 47
      },
      "folds": [
        {
          "accuracy": 0.57143,
          "macro_f1": 0.41228,
          "macro_precision": 0.59259,
          "macro_recall": 0.34444,
          "balanced_accuracy": 0.34444,
          "confusion_matrix": [
            [
              1,
              1,
              1
            ],
            [
              0,
              0,
              1
            ],
            [
              0,
              3,
              7
            ]
          ],
          "fold": 1
        },
        {
          "accuracy": 0.61538,
          "macro_f1": 0.42222,
          "macro_precision": 0.42063,
          "macro_recall": 0.51852,
          "balanced_accuracy": 0.51852,
          "confusion_matrix": [
            [
              3,
              0,
              0
            ],
            [
              0,
              0,
              1
            ],
            [
              4,
              0,
              5
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.76923,
          "macro_f1": 0.51389,
          "macro_precision": 0.5,
          "macro_recall": 0.59259,
          "balanced_accuracy": 0.59259,
          "confusion_matrix": [
            [
              3,
              0,
              0
            ],
            [
              1,
              0,
              0
            ],
            [
              2,
              0,
              7
            ]
          ],
          "fold": 3
        },
        {
          "accuracy": 0.76923,
          "macro_f1": 0.56022,
          "macro_precision": 0.54167,
          "macro_recall": 0.59259,
          "balanced_accuracy": 0.59259,
          "confusion_matrix": [
            [
              3,
              0,
              0
            ],
            [
              0,
              0,
              1
            ],
            [
              1,
              1,
              7
            ]
          ],
          "fold": 4
        },
        {
          "accuracy": 0.76923,
          "macro_f1": 0.74118,
          "macro_precision": 0.75,
          "macro_recall": 0.9,
          "balanced_accuracy": 0.9,
          "confusion_matrix": [
            [
              2,
              0,
              0
            ],
            [
              0,
              1,
              0
            ],
            [
              0,
              3,
              7
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.6989,
        "std_accuracy": 0.08725,
        "mean_macro_f1": 0.52996,
        "std_macro_f1": 0.11937,
        "mean_macro_precision": 0.56098,
        "std_macro_precision": 0.11002,
        "mean_macro_recall": 0.58963,
        "std_macro_recall": 0.17973,
        "mean_balanced_accuracy": 0.58963,
        "std_balanced_accuracy": 0.17973
      },
      "combined": {
        "accuracy": 0.69697,
        "macro_f1": 0.54482,
        "macro_precision": 0.53433,
        "macro_recall": 0.58642,
        "balanced_accuracy": 0.58642,
        "confusion_matrix": [
          [
            12,
            1,
            1
          ],
          [
            1,
            1,
            3
          ],
          [
            7,
            7,
            33
          ]
        ]
      }
    },
    {
      "target": "angle_3class",
      "model": "ridge_classifier",
      "class_names": [
        "Emphysema/Abnormal (<=131 deg)",
        "Intermediate (132-151 deg)",
        "Normal (>=152 deg)"
      ],
      "num_samples": 66,
      "class_counts": {
        "Emphysema/Abnormal (<=131 deg)": 14,
        "Intermediate (132-151 deg)": 5,
        "Normal (>=152 deg)": 47
      },
      "folds": [
        {
          "accuracy": 0.71429,
          "macro_f1": 0.27778,
          "macro_precision": 0.2381,
          "macro_recall": 0.33333,
          "balanced_accuracy": 0.33333,
          "confusion_matrix": [
            [
              0,
              0,
              3
            ],
            [
              0,
              0,
              1
            ],
            [
              0,
              0,
              10
            ]
          ],
          "fold": 1
        },
        {
          "accuracy": 0.69231,
          "macro_f1": 0.47222,
          "macro_precision": 0.45238,
          "macro_recall": 0.55556,
          "balanced_accuracy": 0.55556,
          "confusion_matrix": [
            [
              3,
              0,
              0
            ],
            [
              0,
              0,
              1
            ],
            [
              3,
              0,
              6
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.69231,
          "macro_f1": 0.44974,
          "macro_precision": 0.42593,
          "macro_recall": 0.48148,
          "balanced_accuracy": 0.48148,
          "confusion_matrix": [
            [
              2,
              0,
              1
            ],
            [
              0,
              0,
              1
            ],
            [
              2,
              0,
              7
            ]
          ],
          "fold": 3
        },
        {
          "accuracy": 0.84615,
          "macro_f1": 0.56667,
          "macro_precision": 0.60606,
          "macro_recall": 0.55556,
          "balanced_accuracy": 0.55556,
          "confusion_matrix": [
            [
              2,
              0,
              1
            ],
            [
              0,
              0,
              1
            ],
            [
              0,
              0,
              9
            ]
          ],
          "fold": 4
        },
        {
          "accuracy": 0.92308,
          "macro_f1": 0.65079,
          "macro_precision": 0.63636,
          "macro_recall": 0.66667,
          "balanced_accuracy": 0.66667,
          "confusion_matrix": [
            [
              2,
              0,
              0
            ],
            [
              0,
              0,
              1
            ],
            [
              0,
              0,
              10
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.77363,
        "std_accuracy": 0.09417,
        "mean_macro_f1": 0.48344,
        "std_macro_f1": 0.12529,
        "mean_macro_precision": 0.47177,
        "std_macro_precision": 0.14297,
        "mean_macro_recall": 0.51852,
        "std_macro_recall": 0.10987,
        "mean_balanced_accuracy": 0.51852,
        "std_balanced_accuracy": 0.10987
      },
      "combined": {
        "accuracy": 0.77273,
        "macro_f1": 0.49711,
        "macro_precision": 0.48352,
        "macro_recall": 0.51216,
        "balanced_accuracy": 0.51216,
        "confusion_matrix": [
          [
            9,
            0,
            5
          ],
          [
            0,
            0,
            5
          ],
          [
            5,
            0,
            42
          ]
        ]
      }
    },
    {
      "target": "angle_3class",
      "model": "ordinal_logistic",
      "class_names": [
        "Emphysema/Abnormal (<=131 deg)",
        "Intermediate (132-151 deg)",
        "Normal (>=152 deg)"
      ],
      "num_samples": 66,
      "class_counts": {
        "Emphysema/Abnormal (<=131 deg)": 14,
        "Intermediate (132-151 deg)": 5,
        "Normal (>=152 deg)": 47
      },
      "folds": [
        {
          "accuracy": 0.71429,
          "macro_f1": 0.28986,
          "macro_precision": 0.25641,
          "macro_recall": 0.33333,
          "balanced_accuracy": 0.33333,
          "confusion_matrix": [
            [
              0,
              1,
              2
            ],
            [
              0,
              0,
              1
            ],
            [
              0,
              0,
              10
            ]
          ],
          "fold": 1
        },
        {
          "accuracy": 0.69231,
          "macro_f1": 0.47222,
          "macro_precision": 0.45238,
          "macro_recall": 0.55556,
          "balanced_accuracy": 0.55556,
          "confusion_matrix": [
            [
              3,
              0,
              0
            ],
            [
              0,
              0,
              1
            ],
            [
              3,
              0,
              6
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.69231,
          "macro_f1": 0.48148,
          "macro_precision": 0.48148,
          "macro_recall": 0.48148,
          "balanced_accuracy": 0.48148,
          "confusion_matrix": [
            [
              2,
              0,
              1
            ],
            [
              0,
              0,
              1
            ],
            [
              1,
              1,
              7
            ]
          ],
          "fold": 3
        },
        {
          "accuracy": 0.76923,
          "macro_f1": 0.54737,
          "macro_precision": 0.6,
          "macro_recall": 0.51852,
          "balanced_accuracy": 0.51852,
          "confusion_matrix": [
            [
              2,
              0,
              1
            ],
            [
              0,
              0,
              1
            ],
            [
              0,
              1,
              8
            ]
          ],
          "fold": 4
        },
        {
          "accuracy": 0.92308,
          "macro_f1": 0.65079,
          "macro_precision": 0.63636,
          "macro_recall": 0.66667,
          "balanced_accuracy": 0.66667,
          "confusion_matrix": [
            [
              2,
              0,
              0
            ],
            [
              0,
              0,
              1
            ],
            [
              0,
              0,
              10
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.75824,
        "std_accuracy": 0.08708,
        "mean_macro_f1": 0.48834,
        "std_macro_f1": 0.11796,
        "mean_macro_precision": 0.48533,
        "std_macro_precision": 0.13376,
        "mean_macro_recall": 0.51111,
        "std_macro_recall": 0.10836,
        "mean_balanced_accuracy": 0.51111,
        "std_balanced_accuracy": 0.10836
      },
      "combined": {
        "accuracy": 0.75758,
        "macro_f1": 0.50401,
        "macro_precision": 0.5041,
        "macro_recall": 0.50507,
        "balanced_accuracy": 0.50507,
        "confusion_matrix": [
          [
            9,
            1,
            4
          ],
          [
            0,
            0,
            5
          ],
          [
            4,
            2,
            41
          ]
        ]
      }
    },
    {
      "target": "angle_3class",
      "model": "angle_ridge_threshold",
      "class_names": [
        "Emphysema/Abnormal (<=131 deg)",
        "Intermediate (132-151 deg)",
        "Normal (>=152 deg)"
      ],
      "num_samples": 66,
      "class_counts": {
        "Emphysema/Abnormal (<=131 deg)": 14,
        "Intermediate (132-151 deg)": 5,
        "Normal (>=152 deg)": 47
      },
      "folds": [
        {
          "accuracy": 0.71429,
          "macro_f1": 0.30303,
          "macro_precision": 0.27778,
          "macro_recall": 0.33333,
          "balanced_accuracy": 0.33333,
          "confusion_matrix": [
            [
              0,
              2,
              1
            ],
            [
              0,
              0,
              1
            ],
            [
              0,
              0,
              10
            ]
          ],
          "fold": 1
        },
        {
          "accuracy": 0.61538,
          "macro_f1": 0.47222,
          "macro_precision": 0.47778,
          "macro_recall": 0.51852,
          "balanced_accuracy": 0.51852,
          "confusion_matrix": [
            [
              3,
              0,
              0
            ],
            [
              0,
              0,
              1
            ],
            [
              2,
              2,
              5
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.61538,
          "macro_f1": 0.6,
          "macro_precision": 0.72222,
          "macro_recall": 0.74074,
          "balanced_accuracy": 0.74074,
          "confusion_matrix": [
            [
              2,
              1,
              0
            ],
            [
              0,
              1,
              0
            ],
            [
              0,
              4,
              5
            ]
          ],
          "fold": 3
        },
        {
          "accuracy": 0.76923,
          "macro_f1": 0.48246,
          "macro_precision": 0.63333,
          "macro_recall": 0.44444,
          "balanced_accuracy": 0.44444,
          "confusion_matrix": [
            [
              1,
              2,
              0
            ],
            [
              0,
              0,
              1
            ],
            [
              0,
              0,
              9
            ]
          ],
          "fold": 4
        },
        {
          "accuracy": 0.76923,
          "macro_f1": 0.44912,
          "macro_precision": 0.41667,
          "macro_recall": 0.63333,
          "balanced_accuracy": 0.63333,
          "confusion_matrix": [
            [
              0,
              2,
              0
            ],
            [
              0,
              1,
              0
            ],
            [
              0,
              1,
              9
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.6967,
        "std_accuracy": 0.06936,
        "mean_macro_f1": 0.46137,
        "std_macro_f1": 0.09487,
        "mean_macro_precision": 0.50556,
        "std_macro_precision": 0.15737,
        "mean_macro_recall": 0.53407,
        "std_macro_recall": 0.14222,
        "mean_balanced_accuracy": 0.53407,
        "std_balanced_accuracy": 0.14222
      },
      "combined": {
        "accuracy": 0.69697,
        "macro_f1": 0.52995,
        "macro_precision": 0.59325,
        "macro_recall": 0.54569,
        "balanced_accuracy": 0.54569,
        "confusion_matrix": [
          [
            6,
            7,
            1
          ],
          [
            0,
            2,
            3
          ],
          [
            2,
            7,
            38
          ]
        ]
      }
    },
    {
      "target": "angle_binary_extreme",
      "model": "logistic",
      "class_names": [
        "Abnormal/emphysema-like (AC <=131 deg)",
        "Normal-like (AC >=152 deg)"
      ],
      "num_samples": 61,
      "class_counts": {
        "Abnormal/emphysema-like (AC <=131 deg)": 14,
        "Normal-like (AC >=152 deg)": 47
      },
      "folds": [
        {
          "accuracy": 0.76923,
          "macro_f1": 0.43478,
          "macro_precision": 0.38462,
          "macro_recall": 0.5,
          "balanced_accuracy": 0.5,
          "confusion_matrix": [
            [
              0,
              3
            ],
            [
              0,
              10
            ]
          ],
          "fold": 1
        },
        {
          "accuracy": 0.75,
          "macro_f1": 0.73333,
          "macro_precision": 0.75,
          "macro_recall": 0.83333,
          "balanced_accuracy": 0.83333,
          "confusion_matrix": [
            [
              3,
              0
            ],
            [
              3,
              6
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.83333,
          "macro_f1": 0.77778,
          "macro_precision": 0.77778,
          "macro_recall": 0.77778,
          "balanced_accuracy": 0.77778,
          "confusion_matrix": [
            [
              2,
              1
            ],
            [
              1,
              8
            ]
          ],
          "fold": 3
        },
        {
          "accuracy": 0.91667,
          "macro_f1": 0.87368,
          "macro_precision": 0.95,
          "macro_recall": 0.83333,
          "balanced_accuracy": 0.83333,
          "confusion_matrix": [
            [
              2,
              1
            ],
            [
              0,
              9
            ]
          ],
          "fold": 4
        },
        {
          "accuracy": 0.91667,
          "macro_f1": 0.80952,
          "macro_precision": 0.95455,
          "macro_recall": 0.75,
          "balanced_accuracy": 0.75,
          "confusion_matrix": [
            [
              1,
              1
            ],
            [
              0,
              10
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.83718,
        "std_accuracy": 0.07053,
        "mean_macro_f1": 0.72582,
        "std_macro_f1": 0.15253,
        "mean_macro_precision": 0.76339,
        "std_macro_precision": 0.20747,
        "mean_macro_recall": 0.73889,
        "std_macro_recall": 0.12373,
        "mean_balanced_accuracy": 0.73889,
        "std_balanced_accuracy": 0.12373
      },
      "combined": {
        "accuracy": 0.83607,
        "macro_f1": 0.75561,
        "macro_precision": 0.77211,
        "macro_recall": 0.74316,
        "balanced_accuracy": 0.74316,
        "confusion_matrix": [
          [
            8,
            6
          ],
          [
            4,
            43
          ]
        ]
      }
    },
    {
      "target": "angle_binary_extreme",
      "model": "linear_svm",
      "class_names": [
        "Abnormal/emphysema-like (AC <=131 deg)",
        "Normal-like (AC >=152 deg)"
      ],
      "num_samples": 61,
      "class_counts": {
        "Abnormal/emphysema-like (AC <=131 deg)": 14,
        "Normal-like (AC >=152 deg)": 47
      },
      "folds": [
        {
          "accuracy": 0.92308,
          "macro_f1": 0.87619,
          "macro_precision": 0.95455,
          "macro_recall": 0.83333,
          "balanced_accuracy": 0.83333,
          "confusion_matrix": [
            [
              2,
              1
            ],
            [
              0,
              10
            ]
          ],
          "fold": 1
        },
        {
          "accuracy": 0.66667,
          "macro_f1": 0.65714,
          "macro_precision": 0.71429,
          "macro_recall": 0.77778,
          "balanced_accuracy": 0.77778,
          "confusion_matrix": [
            [
              3,
              0
            ],
            [
              4,
              5
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.83333,
          "macro_f1": 0.8125,
          "macro_precision": 0.8,
          "macro_recall": 0.88889,
          "balanced_accuracy": 0.88889,
          "confusion_matrix": [
            [
              3,
              0
            ],
            [
              2,
              7
            ]
          ],
          "fold": 3
        },
        {
          "accuracy": 0.83333,
          "macro_f1": 0.8125,
          "macro_precision": 0.8,
          "macro_recall": 0.88889,
          "balanced_accuracy": 0.88889,
          "confusion_matrix": [
            [
              3,
              0
            ],
            [
              2,
              7
            ]
          ],
          "fold": 4
        },
        {
          "accuracy": 0.91667,
          "macro_f1": 0.87368,
          "macro_precision": 0.83333,
          "macro_recall": 0.95,
          "balanced_accuracy": 0.95,
          "confusion_matrix": [
            [
              2,
              0
            ],
            [
              1,
              9
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.83462,
        "std_accuracy": 0.09249,
        "mean_macro_f1": 0.8064,
        "std_macro_f1": 0.07969,
        "mean_macro_precision": 0.82043,
        "std_macro_precision": 0.07779,
        "mean_macro_recall": 0.86778,
        "std_macro_recall": 0.0582,
        "mean_balanced_accuracy": 0.86778,
        "std_balanced_accuracy": 0.0582
      },
      "combined": {
        "accuracy": 0.83607,
        "macro_f1": 0.80297,
        "macro_precision": 0.78263,
        "macro_recall": 0.86854,
        "balanced_accuracy": 0.86854,
        "confusion_matrix": [
          [
            13,
            1
          ],
          [
            9,
            38
          ]
        ]
      }
    },
    {
      "target": "angle_binary_extreme",
      "model": "ridge_classifier",
      "class_names": [
        "Abnormal/emphysema-like (AC <=131 deg)",
        "Normal-like (AC >=152 deg)"
      ],
      "num_samples": 61,
      "class_counts": {
        "Abnormal/emphysema-like (AC <=131 deg)": 14,
        "Normal-like (AC >=152 deg)": 47
      },
      "folds": [
        {
          "accuracy": 0.84615,
          "macro_f1": 0.70455,
          "macro_precision": 0.91667,
          "macro_recall": 0.66667,
          "balanced_accuracy": 0.66667,
          "confusion_matrix": [
            [
              1,
              2
            ],
            [
              0,
              10
            ]
          ],
          "fold": 1
        },
        {
          "accuracy": 0.75,
          "macro_f1": 0.73333,
          "macro_precision": 0.75,
          "macro_recall": 0.83333,
          "balanced_accuracy": 0.83333,
          "confusion_matrix": [
            [
              3,
              0
            ],
            [
              3,
              6
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.83333,
          "macro_f1": 0.77778,
          "macro_precision": 0.77778,
          "macro_recall": 0.77778,
          "balanced_accuracy": 0.77778,
          "confusion_matrix": [
            [
              2,
              1
            ],
            [
              1,
              8
            ]
          ],
          "fold": 3
        },
        {
          "accuracy": 0.91667,
          "macro_f1": 0.87368,
          "macro_precision": 0.95,
          "macro_recall": 0.83333,
          "balanced_accuracy": 0.83333,
          "confusion_matrix": [
            [
              2,
              1
            ],
            [
              0,
              9
            ]
          ],
          "fold": 4
        },
        {
          "accuracy": 0.91667,
          "macro_f1": 0.80952,
          "macro_precision": 0.95455,
          "macro_recall": 0.75,
          "balanced_accuracy": 0.75,
          "confusion_matrix": [
            [
              1,
              1
            ],
            [
              0,
              10
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.85256,
        "std_accuracy": 0.06189,
        "mean_macro_f1": 0.77977,
        "std_macro_f1": 0.0592,
        "mean_macro_precision": 0.8698,
        "std_macro_precision": 0.0879,
        "mean_macro_recall": 0.77222,
        "std_macro_recall": 0.06186,
        "mean_balanced_accuracy": 0.77222,
        "std_balanced_accuracy": 0.06186
      },
      "combined": {
        "accuracy": 0.85246,
        "macro_f1": 0.78596,
        "macro_precision": 0.79407,
        "macro_recall": 0.77888,
        "balanced_accuracy": 0.77888,
        "confusion_matrix": [
          [
            9,
            5
          ],
          [
            4,
            43
          ]
        ]
      }
    }
  ]
}
```
