"""Tests for Normal vs Abnormal binary classification with TAP-CT late fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config import Config
from data.loader import RegressionLoaderHelper
from data.manifest import (
    NORMAL_V_ABNORMAL_NAMES,
    build_angle_manifest,
    normal_v_abnormal_label,
)


CONFIG_NAME = "config.normal_v_abnormal.tapct_s3d_late_fusion.augmentationX5.yaml"


def test_normal_v_abnormal_config_is_binary_classification() -> None:
    config = Config.from_yaml(Path(__file__).with_name(CONFIG_NAME))

    assert config.data.target_mode == "normal_v_abnormal"
    assert config.is_classification_task() is True
    assert config.model.name == "hybrid_mamba_tapct_fusion"
    assert config.model.num_classes == 2
    assert config.model_output_dim() == 2
    assert config.data.tapct_features == Path("./embeddings/tapct_s_3d/features.npz")
    assert config.data.source_dir == Path(
        "../classification/datasets/normal_v_abnormal_54"
    )
    assert config.task == "Normal_v_Abnormal_classification"


def test_normal_v_abnormal_label_lists_disease_class_first() -> None:
    assert NORMAL_V_ABNORMAL_NAMES == ["Abnormal", "Normal"]

    # Folder name drives the label; Abnormal is class 0 (disease-positive first).
    assert normal_v_abnormal_label("Abnormal") == (0, "Abnormal")
    assert normal_v_abnormal_label("Normal") == (1, "Normal")
    assert normal_v_abnormal_label("abnormal") == (0, "Abnormal")
    assert normal_v_abnormal_label("something-else") is None


def test_normal_v_abnormal_manifest_splits_cohort_by_folder() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    manifest = build_angle_manifest(
        repo_root / "classification/datasets/normal_v_abnormal_54",
        repo_root / "patient_angle_classification_by_group.json",
        target_mode="normal_v_abnormal",
    )

    assert manifest.class_names == ["Abnormal", "Normal"]
    assert manifest.counts["total"] == 54
    assert manifest.class_counts == {"Abnormal": 33, "Normal": 21}

    for record in manifest.records:
        assert record.class_index in (0, 1)
        if record.source_group.lower().startswith("abnormal"):
            assert record.class_index == 0
        else:
            assert record.class_index == 1


def test_normal_v_abnormal_loader_uses_class_targets_and_embeddings() -> None:
    config = Config.from_yaml(Path(__file__).with_name(CONFIG_NAME))
    regression_root = Path(__file__).resolve().parent

    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        target_mode=config.data.target_mode,
        k_folds=config.training.k_folds,
        seed=config.training.seed,
        batch_size=4,
        val_batch_size=4,
        num_workers=0,
        cache_data=False,
        manifest_path=None,
        intensity_window=config.data.intensity_window,
        input_normalization=config.data.input_normalization,
        augmentation_config=config.data.augmentation,
        balanced_sampling=config.data.balanced_sampling,
        tapct_features=regression_root / config.data.tapct_features,
        load_ct_data=False,
    )

    assert len(loader.records) == 54
    assert loader.tapct_embedding_dim == 1152
    assert loader.get_class_names() == ["Abnormal", "Normal"]
    assert loader.targets.dtype == np.int64
    assert set(np.unique(loader.targets).tolist()) == {0, 1}
    assert int((loader.targets == 0).sum()) == 33
    assert int((loader.targets == 1).sum()) == 21
    assert "classification" in loader.split_strategy

    # Every fold's validation split must hold out whole patients, no leakage.
    for train_idx, val_idx in loader.fold_indices:
        train_patients = {loader.patient_ids[i] for i in train_idx}
        val_patients = {loader.patient_ids[i] for i in val_idx}
        assert not (train_patients & val_patients)

    batch = next(iter(loader.get_train_dl(0)))
    assert batch["target"].dtype.is_floating_point is False
    assert batch["tapct_embedding"].shape[-1] == 1152


def _run() -> None:
    tests = [name for name in globals() if name.startswith("test_")]
    for name in sorted(tests):
        globals()[name]()
        print(f"PASS {name}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    _run()
