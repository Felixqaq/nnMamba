"""Smoke tests for regression Grad-CAM generation."""

from __future__ import annotations

from pathlib import Path
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.config import Config
from core.gradcam import generate_gradcam


class TinyImageEncoder(nn.Module):
    """Small CT branch with a late spatial layer named like the real model."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 4, kernel_size=3, padding=1), nn.GELU())
        self.attention_layers = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward_features(self, ct: torch.Tensor) -> torch.Tensor:
        features = self.attention_layers(self.stem(ct))
        return self.pool(features).flatten(1)


class TinyLateFusionModel(nn.Module):
    """Late-fusion model that mirrors CT + TAP-CT embedding inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = TinyImageEncoder()
        self.embedding_branch = nn.Linear(6, 4)
        self.head = nn.Linear(8, 2)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        image_features = self.image_encoder.forward_features(inputs["ct"])
        embedding_features = self.embedding_branch(inputs["tapct_embedding"])
        return self.head(torch.cat([image_features, embedding_features], dim=1))


class TinyDataset(Dataset):
    """Synthetic dataset matching regression late-fusion batch keys."""

    def __init__(self) -> None:
        self.records = [
            type("Record", (), {"patient_id": "case_0", "path": "case_0.nii.gz"})(),
            type("Record", (), {"patient_id": "case_1", "path": "case_1.nii.gz"})(),
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ct = torch.zeros(1, 8, 8, 8, dtype=torch.float32)
        ct[:, 2:6, 2:6, 2:6] = float(idx + 1)
        return {
            "ct": ct,
            "mri": ct,
            "tapct_embedding": torch.ones(6, dtype=torch.float32) * (idx + 1),
            "label": torch.tensor(idx % 2, dtype=torch.long),
            "target": torch.tensor(idx % 2, dtype=torch.long),
        }


def test_generate_gradcam_for_late_fusion_ct_branch() -> None:
    dataset = TinyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = TinyLateFusionModel().eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "gradcam"
        samples = generate_gradcam(
            model=model,
            dataloader=dataloader,
            device=torch.device("cpu"),
            save_dir=save_dir,
            model_name="hybrid_mamba_tapct_fusion",
            dataset=dataset,
            fold_indices=[0, 1],
            class_names=["Significant emphysema", "No significant emphysema"],
            max_samples=2,
            target_class=0,
            task_type="oi_emphysema",
        )

        assert len(samples) == 2
        assert (save_dir / "manifest.json").exists()
        for sample in samples:
            assert (save_dir / sample["image"]).exists()


def test_oi_emphysema_tapct_configs_enable_gradcam() -> None:
    root = Path(__file__).resolve().parent
    for filename in (
        "config.oi.emphysema.tapct_late_fusion.yaml",
        "config.oi.emphysema.tapct_late_fusion.gray_zone.yaml",
    ):
        config = Config.from_yaml(root / filename)
        assert config.gradcam.enabled is True
        assert config.gradcam.target_layer == "image_encoder.attention_layers"
        assert config.gradcam.target_class == 0


if __name__ == "__main__":
    test_generate_gradcam_for_late_fusion_ct_branch()
    test_oi_emphysema_tapct_configs_enable_gradcam()
    print("Regression Grad-CAM smoke tests passed.")
