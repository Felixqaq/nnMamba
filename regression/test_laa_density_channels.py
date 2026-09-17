"""Smoke tests for native-resolution LAA auxiliary input channels."""

from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from core.config import Config
from data.dataset import AngleRegressionDataset, load_laa_density_channels
from data.manifest import AngleRecord
from data.transforms import RandomCTAugmentation, ToTensor


def make_record(path: Path, patient_id: str = "TEST001") -> AngleRecord:
    return AngleRecord(
        patient_id=patient_id,
        path=str(path),
        angle=0.0,
        source_group="Normal",
        target=1.0,
        class_index=1,
        class_label="Normal",
    )


def test_density_loading_and_dataset_concatenation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ct_path = root / "TEST001_ct.nii.gz"
        density_dir = root / "density"
        density_dir.mkdir()
        nib.save(
            nib.Nifti1Image(np.arange(64, dtype=np.float32).reshape(4, 4, 4), np.eye(4)),
            ct_path,
        )
        stored = np.zeros((2, 4, 4, 4), dtype=np.uint8)
        stored[0] = 51
        stored[1] = 204
        np.save(density_dir / "TEST001.npy", stored)

        density = load_laa_density_channels(
            density_dir / "TEST001.npy", (4, 4, 4), "density"
        )
        assert density.shape == (1, 4, 4, 4)
        np.testing.assert_allclose(density, 0.2, atol=1 / 255)

        for mode, expected_channels in (
            ("density", 2),
            ("density_and_occupancy", 3),
        ):
            for cache_data in (False, True):
                dataset = AngleRegressionDataset(
                    data_root=root,
                    labels_json=root / "unused.json",
                    target_mode="normal_v_abnormal",
                    image_size=(4, 4, 4),
                    input_normalization="zscore",
                    laa_density_dir=density_dir,
                    laa_density_mode=mode,
                    records=[make_record(ct_path)],
                    transform=ToTensor(),
                    cache_data=cache_data,
                )
                sample = dataset[0]
                assert torch.is_tensor(sample["ct"])
                assert sample["ct"].shape == (expected_channels, 4, 4, 4)
                torch.testing.assert_close(
                    sample["ct"][1],
                    torch.full((4, 4, 4), 0.2),
                    atol=1 / 255,
                    rtol=0,
                )
                if expected_channels == 3:
                    torch.testing.assert_close(
                        sample["ct"][2],
                        torch.full((4, 4, 4), 0.8),
                        atol=1 / 255,
                        rtol=0,
                    )
                if cache_data:
                    assert isinstance(dataset.cached_data[0]["ct"], np.ndarray)
                    assert dataset.cached_data[0]["ct"].shape == (1, 4, 4, 4)
                    assert dataset.cached_laa_density[0].dtype == np.uint8


def test_density_validation_fails_loudly() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.npy"
        np.save(path, np.zeros((1, 4, 4, 4), dtype=np.uint8))
        try:
            load_laa_density_channels(path, (4, 4, 4), "density")
        except ValueError as exc:
            assert "Expected LAA density shape" in str(exc)
        else:
            raise AssertionError("invalid LAA density shape was accepted")


def test_intensity_augmentation_preserves_auxiliary_channels() -> None:
    augmentation = RandomCTAugmentation(
        enabled=True,
        probability=1.0,
        class_indices=(0, 1),
        rotation_degrees=0.0,
        translation_fraction=0.0,
        scale_range=(1.0, 1.0),
        intensity_scale_range=(2.0, 2.0),
        intensity_shift_range=(1.0, 1.0),
        noise_std=0.0,
    )
    volume = torch.stack(
        (
            torch.ones(2, 2, 2),
            torch.full((2, 2, 2), 0.2),
            torch.full((2, 2, 2), 0.8),
        )
    )
    cpu_result = augmentation._random_intensity(volume)
    torch.testing.assert_close(cpu_result[0], torch.full((2, 2, 2), 3.0))
    torch.testing.assert_close(cpu_result[1:], volume[1:])

    batch = torch.stack((volume, volume + torch.tensor([1.0, 0.0, 0.0])[:, None, None, None]))
    batch_result = augmentation._batch_intensity(batch)
    torch.testing.assert_close(batch_result[:, 0], batch[:, 0] * 2.0 + 1.0)
    torch.testing.assert_close(batch_result[:, 1:], batch[:, 1:])


def test_experiment_configs_match_channel_count() -> None:
    root = Path(__file__).resolve().parent
    two = Config.from_yaml(root / "config.rq1.normal_v_abnormal.image.fev1fvc70.mamba_laa2.yaml")
    three = Config.from_yaml(root / "config.rq1.normal_v_abnormal.image.fev1fvc70.mamba_laa3.yaml")
    assert two.model.in_channels == 2
    assert two.data.laa_density_mode == "density"
    assert three.model.in_channels == 3
    assert three.data.laa_density_mode == "density_and_occupancy"


if __name__ == "__main__":
    test_density_loading_and_dataset_concatenation()
    test_density_validation_fails_loudly()
    test_intensity_augmentation_preserves_auxiliary_channels()
    test_experiment_configs_match_channel_count()
    print("LAA density channel smoke tests passed")
