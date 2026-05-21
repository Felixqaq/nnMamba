# TAP-CT Embedding Probe: autoplot_smoke

- 原始 results.json: [regression/figures/TAPCT_embedding_probes/autoplot_smoke/results.json](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/autoplot_smoke/results.json)
- features: `regression/embeddings/tapct_s_3d/features.npz`
- metadata: `regression/embeddings/tapct_s_3d/metadata.csv`
- n_splits: `5`
- seed: `42`
- ridge_alpha: `1.0`

## Probe Summary
| target | model | n | mean_accuracy | mean_macro_f1 | mean_bal_acc | mean_mae | mean_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| angle_binary_extreme | ridge_classifier | 61 | 0.88718 | 0.79708 | 0.81111 |  |  |

## Related Files
- [regression/figures/TAPCT_embedding_probes/autoplot_smoke/angle_binary_extreme_metric_comparison.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/autoplot_smoke/angle_binary_extreme_metric_comparison.png) — 60.2 KB
- [regression/figures/TAPCT_embedding_probes/autoplot_smoke/angle_binary_extreme_ridge_classifier_class_recall.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/autoplot_smoke/angle_binary_extreme_ridge_classifier_class_recall.png) — 75.9 KB
- [regression/figures/TAPCT_embedding_probes/autoplot_smoke/angle_binary_extreme_ridge_classifier_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/autoplot_smoke/angle_binary_extreme_ridge_classifier_confusion_matrix.png) — 99.8 KB
- [regression/figures/TAPCT_embedding_probes/autoplot_smoke/predictions.csv](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/autoplot_smoke/predictions.csv) — 13.2 KB
- [regression/figures/TAPCT_embedding_probes/autoplot_smoke/probe_metric_overview.png](/home/felix/Research/nnMamba/regression/figures/TAPCT_embedding_probes/autoplot_smoke/probe_metric_overview.png) — 61.2 KB

## Full results.json
```json
{
  "features": "regression/embeddings/tapct_s_3d/features.npz",
  "metadata": "regression/embeddings/tapct_s_3d/metadata.csv",
  "n_splits": 5,
  "seed": 42,
  "ridge_alpha": 1.0,
  "results": [
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
          "accuracy": 1.0,
          "macro_f1": 1.0,
          "macro_precision": 1.0,
          "macro_recall": 1.0,
          "balanced_accuracy": 1.0,
          "confusion_matrix": [
            [
              2,
              0
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
        "mean_accuracy": 0.88718,
        "std_accuracy": 0.07909,
        "mean_macro_f1": 0.79708,
        "std_macro_f1": 0.19447,
        "mean_macro_precision": 0.79748,
        "std_macro_precision": 0.2196,
        "mean_macro_recall": 0.81111,
        "std_macro_recall": 0.17427,
        "mean_balanced_accuracy": 0.81111,
        "std_balanced_accuracy": 0.17427
      },
      "combined": {
        "accuracy": 0.88525,
        "macro_f1": 0.82392,
        "macro_precision": 0.85909,
        "macro_recall": 0.80015,
        "balanced_accuracy": 0.80015,
        "confusion_matrix": [
          [
            9,
            5
          ],
          [
            2,
            45
          ]
        ]
      }
    }
  ]
}
```
