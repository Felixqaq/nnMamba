# Hard Majority-Vote Ensemble — Normal vs Abnormal

**Model:** Hybrid Mamba-Attention (image-only CT branch), train-only 5× augmentation
**Ensemble:** 5 seed-diverse members (seeds 42, 1, 7, 13, 23), folds pinned by `split_seed=42`, hard majority voting
**Cohort:** 54-case patient-level 5-fold CV (33 Abnormal, 21 Normal)
**Run UUID:** `hybrid_mamba_attention_normal_vs_abnormal_image_only_5x_aug_ensemble_majority_ensemble_2026-07-10_00:23:19`

## Results (5-fold average)

| vote_size | Accuracy | Sensitivity (Abnormal) | Specificity (Normal) |
| --- | --- | --- | --- |
| 1 (single member) | 0.8145 ± 0.0580 | 0.8143 ± 0.1170 | 0.8000 ± 0.1000 |
| 3 | 0.8509 ± 0.0749 | 0.8143 ± 0.1170 | 0.9000 ± 0.1225 |
| 5 | 0.8509 ± 0.0749 | 0.8143 ± 0.1170 | 0.9000 ± 0.1225 |

Notes:
- `std` = population standard deviation (ddof=0) across the 5 outer folds.
- vote_size 3 and 5 are identical: adding members 4–5 changed no predictions (low member diversity — members differ only by random seed).
- The ensemble's gain over the single model comes entirely from the Normal side (specificity 0.80 → 0.90); Abnormal sensitivity is unchanged (0.8143).

## Caveat (evaluation bias)
Each member selects its best epoch on the **test fold** (macro-F1), so these numbers are an **optimistic upper bound**, not an unbiased generalization estimate. The unbiased nested-CV estimate for the same single model is **Accuracy 0.698 ± 0.175 / AUC 0.685**. Report both in the thesis.
