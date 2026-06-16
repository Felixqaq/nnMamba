"""Tests for OI regression with TAP-CT late fusion."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from core.config import Config
from core.evaluator import RegressionMetrics, save_predictions
from data.loader import RegressionLoaderHelper
from data.manifest import build_angle_manifest, load_oi_label_map


OI_PATIENT_COUNT = 66
# A stable subset known to be present. The OI cohort was expanded from 5 to 66
# patients and the pef field was dropped, so membership/count checks replace the
# old exact-set and exact-value assertions.
OI_SAMPLE_IDS = {"0781915", "1261736", "1596038", "1604378", "1663485"}


def test_oi_tapct_late_fusion_config() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.oi.tapct_late_fusion.yaml")
    )

    assert config.data.target_mode == "oi"
    assert config.is_classification_task() is False
    assert config.model.name == "hybrid_mamba_tapct_fusion"
    assert config.model.num_classes == 1
    assert config.model_output_dim() == 1
    assert config.model.tapct_embedding_dim == 2304
    assert config.data.oi_json == Path("./oi_processed.json")
    assert config.data.manifest == Path(
        "./datasets/generated/oi_manifest.tapct_late_fusion.json"
    )
    assert config.data.tapct_features == Path("./embeddings/tapct_b_3d/features.npz")
    assert config.data.balanced_sampling is False
    assert config.data.augmentation.enabled is False
    assert config.task == "OI_regression"


def test_load_oi_label_map_reads_processed_json() -> None:
    regression_root = Path(__file__).resolve().parent

    labels = load_oi_label_map(regression_root / "oi_processed.json")

    assert len(labels) == OI_PATIENT_COUNT
    assert OI_SAMPLE_IDS <= set(labels)
    assert labels["0781915"]["oi"] == 1.64
    assert labels["0781915"]["a"] == 1.6
    assert labels["0781915"]["fvc"] == 2.62
    assert labels["0781915"]["pef"] is None


def test_oi_manifest_matches_cts_and_preserves_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    regression_root = repo_root / "regression"

    manifest = build_angle_manifest(
        repo_root / "by_angle_all",
        repo_root / "patient_angle_classification_by_group.json",
        pft_json=repo_root / "pft.json",
        target_mode="oi",
        oi_json=regression_root / "oi_processed.json",
    )

    assert manifest.source_json.endswith("oi_processed.json")
    assert manifest.counts["total"] == OI_PATIENT_COUNT
    assert manifest.counts["unique_patients"] == OI_PATIENT_COUNT
    assert OI_SAMPLE_IDS <= {record.patient_id for record in manifest.records}
    assert manifest.class_names == []
    assert manifest.class_counts == {}
    assert manifest.extra_in_source_not_in_json == []

    first = next(record for record in manifest.records if record.patient_id == "0781915")
    assert Path(first.path).exists()
    assert first.target == first.oi == 1.64
    assert first.a == 1.6
    assert first.fvc == 2.62
    assert first.pef is None
    assert first.angle == 176.0


def test_oi_loader_uses_oi_targets_and_tapct_embeddings() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.oi.tapct_late_fusion.yaml")
    )
    regression_root = Path(__file__).resolve().parent

    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        pft_json=regression_root / config.data.pft_json,
        oi_json=regression_root / config.data.oi_json,
        target_mode=config.data.target_mode,
        k_folds=config.training.k_folds,
        seed=config.training.seed,
        batch_size=2,
        val_batch_size=1,
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

    assert len(loader.records) == OI_PATIENT_COUNT
    assert loader.tapct_embedding_dim == 2304
    assert OI_SAMPLE_IDS <= set(loader.patient_ids)
    # Targets are the raw OI values straight from oi_processed.json.
    patient_to_target = dict(zip(loader.patient_ids, loader.targets.tolist()))
    assert round(float(patient_to_target["0781915"]), 2) == 1.64
    assert round(float(loader.targets.min()), 2) == 1.61
    assert round(float(loader.targets.max()), 2) == 26.89
    assert loader.split_strategy == "stratified_bins"

    sample = loader.dataset[0]
    record = loader.records[0]
    assert torch.isclose(sample["target"], torch.tensor(record.oi))
    assert torch.isclose(sample["oi"], torch.tensor(record.oi))
    assert sample["tapct_embedding"].shape == (2304,)

    batch = next(iter(loader.get_train_dl(0)))
    assert batch["target"].shape == (2,)
    assert batch["tapct_embedding"].shape == (2, 2304)
    assert batch["ct"].shape == (2, 1, 1, 1, 1)


def test_oi_prediction_export_uses_oi_field_names(tmp_path: Path) -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.oi.tapct_late_fusion.yaml")
    )
    regression_root = Path(__file__).resolve().parent
    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        pft_json=regression_root / config.data.pft_json,
        oi_json=regression_root / config.data.oi_json,
        target_mode=config.data.target_mode,
        k_folds=config.training.k_folds,
        seed=config.training.seed,
        batch_size=2,
        val_batch_size=1,
        num_workers=0,
        cache_data=False,
        manifest_path=None,
        intensity_window=config.data.intensity_window,
        input_normalization=config.data.input_normalization,
        tapct_features=regression_root / config.data.tapct_features,
        load_ct_data=False,
    )

    metrics = RegressionMetrics(
        mae=0.1,
        rmse=0.1,
        r2=0.0,
        pearson=0.0,
        mean_error=0.1,
        mse=0.01,
        labels=torch.tensor([loader.records[0].oi]),
        preds=torch.tensor([loader.records[0].oi + 0.1]),
    )

    save_predictions(
        metrics=metrics,
        dataset=loader.dataset,
        fold_indices=[0],
        save_path=tmp_path,
        fold=1,
        task_type="oi",
    )

    payload = json.loads((tmp_path / "fold1_predictions.json").read_text())
    row = payload["predictions"][0]
    assert payload["target"] == "OI"
    assert "true_oi" in row
    assert "predicted_oi" in row
    assert "true_angle" not in row
    assert row["a"] is not None
    assert row["fvc"] is not None
    # pef was dropped from the expanded oi_processed.json, so it is omitted.
    assert "pef" not in row
