# TAP-CT Embedding Probe: tapct_b_3d_yaml

- 原始 results.json: [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/results.json](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/results.json)
- features: `/home/felix/Research/nnMamba/regression/embeddings/tapct_b_3d/features.npz`
- metadata: `/home/felix/Research/nnMamba/regression/embeddings/tapct_b_3d/metadata.csv`
- n_splits: `5`
- seed: `42`
- ridge_alpha: `1.0`

## Probe Summary
| target | model | n | mean_accuracy | mean_macro_f1 | mean_bal_acc | mean_mae | mean_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| angle_3class | logistic | 66 | 0.78901 | 0.4776 | 0.51185 |  |  |
| angle_3class | linear_svm | 66 | 0.63626 | 0.52015 | 0.6363 |  |  |
| angle_3class | ridge_classifier | 66 | 0.75824 | 0.44199 | 0.47111 |  |  |
| angle_3class | ordinal_logistic | 66 | 0.75824 | 0.47235 | 0.49704 |  |  |
| angle_3class | angle_ridge_threshold | 66 | 0.62088 | 0.43972 | 0.42 |  |  |
| angle_binary_extreme | logistic | 61 | 0.85385 | 0.73893 | 0.76778 |  |  |
| angle_binary_extreme | linear_svm | 61 | 0.7359 | 0.71539 | 0.82889 |  |  |
| angle_binary_extreme | ridge_classifier | 61 | 0.83718 | 0.70419 | 0.71778 |  |  |

## Related Files
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_angle_ridge_threshold_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_angle_ridge_threshold_class_recall.png) — 82.6 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_angle_ridge_threshold_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_angle_ridge_threshold_confusion_matrix.png) — 111.0 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_linear_svm_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_linear_svm_class_recall.png) — 82.5 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_linear_svm_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_linear_svm_confusion_matrix.png) — 107.8 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_logistic_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_logistic_class_recall.png) — 81.2 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_logistic_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_logistic_confusion_matrix.png) — 111.6 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_metric_comparison.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_metric_comparison.png) — 87.2 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_ordinal_logistic_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_ordinal_logistic_class_recall.png) — 81.4 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_ordinal_logistic_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_ordinal_logistic_confusion_matrix.png) — 110.3 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_ridge_classifier_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_ridge_classifier_class_recall.png) — 81.0 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_ridge_classifier_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_3class_ridge_classifier_confusion_matrix.png) — 109.5 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_linear_svm_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_linear_svm_class_recall.png) — 76.7 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_linear_svm_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_linear_svm_confusion_matrix.png) — 99.4 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_logistic_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_logistic_class_recall.png) — 76.1 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_logistic_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_logistic_confusion_matrix.png) — 100.6 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_metric_comparison.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_metric_comparison.png) — 75.7 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_ridge_classifier_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_ridge_classifier_class_recall.png) — 75.4 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_ridge_classifier_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/angle_binary_extreme_ridge_classifier_confusion_matrix.png) — 98.0 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/predictions.csv](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/predictions.csv) — 101.8 KB
- [regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/probe_metric_overview.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml/probe_metric_overview.png) — 82.3 KB

## Full results.json
```json
{
  "features": "/home/felix/Research/nnMamba/regression/embeddings/tapct_b_3d/features.npz",
  "metadata": "/home/felix/Research/nnMamba/regression/embeddings/tapct_b_3d/metadata.csv",
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
          "accuracy": 0.76923,
          "macro_f1": 0.52451,
          "macro_precision": 0.49167,
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
              2,
              0,
              7
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.76923,
          "macro_f1": 0.45238,
          "macro_precision": 0.58333,
          "macro_recall": 0.44444,
          "balanced_accuracy": 0.44444,
          "confusion_matrix": [
            [
              1,
              0,
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
              9
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
          "accuracy": 0.84615,
          "macro_f1": 0.56667,
          "macro_precision": 0.52222,
          "macro_recall": 0.63333,
          "balanced_accuracy": 0.63333,
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
              1,
              0,
              9
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.78901,
        "std_accuracy": 0.05078,
        "mean_macro_f1": 0.4776,
        "std_macro_f1": 0.10829,
        "mean_macro_precision": 0.48828,
        "std_macro_precision": 0.13165,
        "mean_macro_recall": 0.51185,
        "std_macro_recall": 0.10918,
        "mean_balanced_accuracy": 0.51185,
        "std_balanced_accuracy": 0.10918
      },
      "combined": {
        "accuracy": 0.78788,
        "macro_f1": 0.50092,
        "macro_precision": 0.50909,
        "macro_recall": 0.50253,
        "balanced_accuracy": 0.50253,
        "confusion_matrix": [
          [
            8,
            0,
            6
          ],
          [
            0,
            0,
            5
          ],
          [
            3,
            0,
            44
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
          "accuracy": 0.64286,
          "macro_f1": 0.48148,
          "macro_precision": 0.51389,
          "macro_recall": 0.45556,
          "balanced_accuracy": 0.45556,
          "confusion_matrix": [
            [
              2,
              1,
              0
            ],
            [
              0,
              0,
              1
            ],
            [
              1,
              2,
              7
            ]
          ],
          "fold": 1
        },
        {
          "accuracy": 0.53846,
          "macro_f1": 0.37179,
          "macro_precision": 0.44444,
          "macro_recall": 0.48148,
          "balanced_accuracy": 0.48148,
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
              5,
              0,
              4
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.61538,
          "macro_f1": 0.59402,
          "macro_precision": 0.61111,
          "macro_recall": 0.81481,
          "balanced_accuracy": 0.81481,
          "confusion_matrix": [
            [
              3,
              0,
              0
            ],
            [
              0,
              1,
              0
            ],
            [
              3,
              2,
              4
            ]
          ],
          "fold": 3
        },
        {
          "accuracy": 0.84615,
          "macro_f1": 0.62963,
          "macro_precision": 0.62963,
          "macro_recall": 0.62963,
          "balanced_accuracy": 0.62963,
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
              0,
              1,
              8
            ]
          ],
          "fold": 4
        },
        {
          "accuracy": 0.53846,
          "macro_f1": 0.52381,
          "macro_precision": 0.56667,
          "macro_recall": 0.8,
          "balanced_accuracy": 0.8,
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
              2,
              4,
              4
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.63626,
        "std_accuracy": 0.11284,
        "mean_macro_f1": 0.52015,
        "std_macro_f1": 0.09051,
        "mean_macro_precision": 0.55315,
        "std_macro_precision": 0.06745,
        "mean_macro_recall": 0.6363,
        "std_macro_recall": 0.15188,
        "mean_balanced_accuracy": 0.6363,
        "std_balanced_accuracy": 0.15188
      },
      "combined": {
        "accuracy": 0.63636,
        "macro_f1": 0.5375,
        "macro_precision": 0.53923,
        "macro_recall": 0.63435,
        "balanced_accuracy": 0.63435,
        "confusion_matrix": [
          [
            13,
            1,
            0
          ],
          [
            1,
            2,
            2
          ],
          [
            11,
            9,
            27
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
          "accuracy": 0.76923,
          "macro_f1": 0.52451,
          "macro_precision": 0.49167,
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
              2,
              0,
              7
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.76923,
          "macro_f1": 0.45238,
          "macro_precision": 0.58333,
          "macro_recall": 0.44444,
          "balanced_accuracy": 0.44444,
          "confusion_matrix": [
            [
              1,
              0,
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
              9
            ]
          ],
          "fold": 3
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
          "fold": 4
        },
        {
          "accuracy": 0.76923,
          "macro_f1": 0.45238,
          "macro_precision": 0.43939,
          "macro_recall": 0.46667,
          "balanced_accuracy": 0.46667,
          "confusion_matrix": [
            [
              1,
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
              9
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.75824,
        "std_accuracy": 0.02198,
        "mean_macro_f1": 0.44199,
        "std_macro_f1": 0.08684,
        "mean_macro_precision": 0.44828,
        "std_macro_precision": 0.11491,
        "mean_macro_recall": 0.47111,
        "std_macro_recall": 0.0857,
        "mean_balanced_accuracy": 0.47111,
        "std_balanced_accuracy": 0.0857
      },
      "combined": {
        "accuracy": 0.75758,
        "macro_f1": 0.46771,
        "macro_precision": 0.47273,
        "macro_recall": 0.47163,
        "balanced_accuracy": 0.47163,
        "confusion_matrix": [
          [
            7,
            0,
            7
          ],
          [
            0,
            0,
            5
          ],
          [
            4,
            0,
            43
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
          "accuracy": 0.76923,
          "macro_f1": 0.52451,
          "macro_precision": 0.49167,
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
              2,
              0,
              7
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.69231,
          "macro_f1": 0.43333,
          "macro_precision": 0.57576,
          "macro_recall": 0.40741,
          "balanced_accuracy": 0.40741,
          "confusion_matrix": [
            [
              1,
              0,
              2
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
          "accuracy": 0.84615,
          "macro_f1": 0.56667,
          "macro_precision": 0.52222,
          "macro_recall": 0.63333,
          "balanced_accuracy": 0.63333,
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
              1,
              0,
              9
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.75824,
        "std_accuracy": 0.05338,
        "mean_macro_f1": 0.47235,
        "std_macro_f1": 0.10205,
        "mean_macro_precision": 0.48921,
        "std_macro_precision": 0.12252,
        "mean_macro_recall": 0.49704,
        "std_macro_recall": 0.11225,
        "mean_balanced_accuracy": 0.49704,
        "std_balanced_accuracy": 0.11225
      },
      "combined": {
        "accuracy": 0.75758,
        "macro_f1": 0.49616,
        "macro_precision": 0.51166,
        "macro_recall": 0.48835,
        "balanced_accuracy": 0.48835,
        "confusion_matrix": [
          [
            8,
            1,
            5
          ],
          [
            0,
            0,
            5
          ],
          [
            3,
            2,
            42
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
          "accuracy": 0.64286,
          "macro_f1": 0.27273,
          "macro_precision": 0.25,
          "macro_recall": 0.3,
          "balanced_accuracy": 0.3,
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
              1,
              9
            ]
          ],
          "fold": 1
        },
        {
          "accuracy": 0.38462,
          "macro_f1": 0.35714,
          "macro_precision": 0.6,
          "macro_recall": 0.25926,
          "balanced_accuracy": 0.25926,
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
              5,
              4
            ]
          ],
          "fold": 2
        },
        {
          "accuracy": 0.53846,
          "macro_f1": 0.41667,
          "macro_precision": 0.61905,
          "macro_recall": 0.33333,
          "balanced_accuracy": 0.33333,
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
              3,
              6
            ]
          ],
          "fold": 3
        },
        {
          "accuracy": 0.69231,
          "macro_f1": 0.44737,
          "macro_precision": 0.6,
          "macro_recall": 0.40741,
          "balanced_accuracy": 0.40741,
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
              1,
              8
            ]
          ],
          "fold": 4
        },
        {
          "accuracy": 0.84615,
          "macro_f1": 0.70468,
          "macro_precision": 0.66667,
          "macro_recall": 0.8,
          "balanced_accuracy": 0.8,
          "confusion_matrix": [
            [
              1,
              1,
              0
            ],
            [
              0,
              1,
              0
            ],
            [
              1,
              0,
              9
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.62088,
        "std_accuracy": 0.15424,
        "mean_macro_f1": 0.43972,
        "std_macro_f1": 0.14526,
        "mean_macro_precision": 0.54714,
        "std_macro_precision": 0.15056,
        "mean_macro_recall": 0.42,
        "std_macro_recall": 0.19612,
        "mean_balanced_accuracy": 0.42,
        "std_balanced_accuracy": 0.19612
      },
      "combined": {
        "accuracy": 0.62121,
        "macro_f1": 0.436,
        "macro_precision": 0.56425,
        "macro_recall": 0.41722,
        "balanced_accuracy": 0.41722,
        "confusion_matrix": [
          [
            4,
            7,
            3
          ],
          [
            0,
            1,
            4
          ],
          [
            1,
            10,
            36
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
          "fold": 2
        },
        {
          "accuracy": 0.83333,
          "macro_f1": 0.7,
          "macro_precision": 0.90909,
          "macro_recall": 0.66667,
          "balanced_accuracy": 0.66667,
          "confusion_matrix": [
            [
              1,
              2
            ],
            [
              0,
              9
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
        "mean_accuracy": 0.85385,
        "std_accuracy": 0.05638,
        "mean_macro_f1": 0.73893,
        "std_macro_f1": 0.16477,
        "mean_macro_precision": 0.77541,
        "std_macro_precision": 0.2025,
        "mean_macro_recall": 0.76778,
        "std_macro_recall": 0.16377,
        "mean_balanced_accuracy": 0.76778,
        "std_balanced_accuracy": 0.16377
      },
      "combined": {
        "accuracy": 0.85246,
        "macro_f1": 0.77361,
        "macro_precision": 0.80364,
        "macro_recall": 0.7538,
        "balanced_accuracy": 0.7538,
        "confusion_matrix": [
          [
            8,
            6
          ],
          [
            3,
            44
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
          "accuracy": 0.84615,
          "macro_f1": 0.81944,
          "macro_precision": 0.8,
          "macro_recall": 0.9,
          "balanced_accuracy": 0.9,
          "confusion_matrix": [
            [
              3,
              0
            ],
            [
              2,
              8
            ]
          ],
          "fold": 1
        },
        {
          "accuracy": 0.5,
          "macro_f1": 0.5,
          "macro_precision": 0.66667,
          "macro_recall": 0.66667,
          "balanced_accuracy": 0.66667,
          "confusion_matrix": [
            [
              3,
              0
            ],
            [
              6,
              3
            ]
          ],
          "fold": 2
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
          "fold": 3
        },
        {
          "accuracy": 0.91667,
          "macro_f1": 0.89916,
          "macro_precision": 0.875,
          "macro_recall": 0.94444,
          "balanced_accuracy": 0.94444,
          "confusion_matrix": [
            [
              3,
              0
            ],
            [
              1,
              8
            ]
          ],
          "fold": 4
        },
        {
          "accuracy": 0.66667,
          "macro_f1": 0.625,
          "macro_precision": 0.66667,
          "macro_recall": 0.8,
          "balanced_accuracy": 0.8,
          "confusion_matrix": [
            [
              2,
              0
            ],
            [
              4,
              6
            ]
          ],
          "fold": 5
        }
      ],
      "summary": {
        "mean_accuracy": 0.7359,
        "std_accuracy": 0.14524,
        "mean_macro_f1": 0.71539,
        "std_macro_f1": 0.14106,
        "mean_macro_precision": 0.75167,
        "std_macro_precision": 0.08,
        "mean_macro_recall": 0.82889,
        "std_macro_recall": 0.09548,
        "mean_balanced_accuracy": 0.82889,
        "std_balanced_accuracy": 0.09548
      },
      "combined": {
        "accuracy": 0.7377,
        "macro_f1": 0.71562,
        "macro_precision": 0.73333,
        "macro_recall": 0.82979,
        "balanced_accuracy": 0.82979,
        "confusion_matrix": [
          [
            14,
            0
          ],
          [
            16,
            31
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
          "fold": 2
        },
        {
          "accuracy": 0.83333,
          "macro_f1": 0.7,
          "macro_precision": 0.90909,
          "macro_recall": 0.66667,
          "balanced_accuracy": 0.66667,
          "confusion_matrix": [
            [
              1,
              2
            ],
            [
              0,
              9
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
          "accuracy": 0.83333,
          "macro_f1": 0.7,
          "macro_precision": 0.7,
          "macro_recall": 0.7,
          "balanced_accuracy": 0.7,
          "confusion_matrix": [
            [
              1,
              1
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
        "mean_accuracy": 0.83718,
        "std_accuracy": 0.04686,
        "mean_macro_f1": 0.70419,
        "std_macro_f1": 0.15038,
        "mean_macro_precision": 0.74874,
        "std_macro_precision": 0.20189,
        "mean_macro_recall": 0.71778,
        "std_macro_recall": 0.13637,
        "mean_balanced_accuracy": 0.71778,
        "std_balanced_accuracy": 0.13637
      },
      "combined": {
        "accuracy": 0.83607,
        "macro_f1": 0.74065,
        "macro_precision": 0.78137,
        "macro_recall": 0.71809,
        "balanced_accuracy": 0.71809,
        "confusion_matrix": [
          [
            7,
            7
          ],
          [
            3,
            44
          ]
        ]
      }
    }
  ]
}
```
