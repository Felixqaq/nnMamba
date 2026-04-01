# Regression Dataset Layout

This folder stores generated metadata for the CT angle regression pipeline.

The raw CT volumes are not kept here. Use the angle-sorted source tree at:

```text
by_angle_all/
├── abnormal_low_angle/
└── normal_high_angle/
```

Patient angle labels are loaded from:

```text
patient_angle_classification_by_group.json
```

The regression scripts can generate a manifest and overview figures into this area:

```text
regression/datasets/generated/
├── regression_manifest.json
└── figures/
```

Recommended workflow:

1. Build a manifest from the source CT folders.
2. Run dataset validation to confirm label coverage.
3. Generate overview figures before training.

Do not commit raw CT volumes, generated checkpoints, or large temporary outputs here.

