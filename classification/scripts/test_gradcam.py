#!/usr/bin/env python
"""Smoke test Grad-CAM generation with a tiny synthetic 3D model."""

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CLASSIFICATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLASSIFICATION_ROOT))

from core.gradcam import generate_gradcam  # noqa: E402


class Tiny3DClassifier(nn.Module):
    """Small model with a spatial 3D conv layer suitable for Grad-CAM."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=1),
            nn.Conv3d(2, 2, kernel_size=1),
            nn.Conv3d(2, 2, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(2, 1)
        self._init_predictable_weights()

    def _init_predictable_weights(self) -> None:
        with torch.no_grad():
            for module in self.features:
                module.weight.fill_(0.5)
                module.bias.zero_()
            self.head.weight.fill_(1.0)
            self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class TinyDataset(Dataset):
    """Synthetic dataset matching the classification batch contract."""

    def __init__(self) -> None:
        self.directories = [
            Path("Abnormal/case_tp.nii.gz"),
            Path("Normal/case_tn.nii.gz"),
            Path("Normal/case_fp.nii.gz"),
            Path("Abnormal/case_fn.nii.gz"),
        ]
        self.labels = [1, 0, 0, 1]
        self.signs = [1.0, -1.0, 1.0, -1.0]

    def __len__(self) -> int:
        return len(self.directories)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = torch.ones(1, 8, 8, 8, dtype=torch.float32) * self.signs[idx]
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return {"mri": image, "label": label}


def main() -> None:
    device = torch.device("cpu")
    model = Tiny3DClassifier().to(device).eval()
    dataset = TinyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "gradcam"
        samples = generate_gradcam(
            model=model,
            dataloader=dataloader,
            device=device,
            save_dir=save_dir,
            model_name="tiny",
            dataset=dataset,
            fold_indices=[0, 1, 2, 3],
            labels=["Normal", "Abnormal"],
            max_samples=4,
            target_layer_names=["features.0", "features.1", "features.2"],
            per_outcome=1,
        )

        assert len(samples) == 12
        assert (save_dir / "manifest.json").exists()
        for layer in ("features.0", "features.1", "features.2"):
            layer_dir = save_dir / layer
            assert (layer_dir / "manifest.json").exists()
            layer_samples = [sample for sample in samples if sample["layer_dir"] == layer]
            assert {sample["outcome"] for sample in layer_samples} == {
                "TP",
                "TN",
                "FP",
                "FN",
            }
            for sample in layer_samples:
                assert (layer_dir / sample["image"]).exists()

    print("Grad-CAM smoke test passed.")


if __name__ == "__main__":
    main()
