"""Executable smoke checks for paired windows and frozen-teacher training."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import time
import subprocess
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from core.window_distillation import classification_logit, feature_loss, freeze_teacher, train_stage
from data.window_distillation import WindowDataset, prepare_cache, read_manifest, window_array


class TinyModel(torch.nn.Module):
    """CPU test double; results are explicitly not patient/model efficacy results."""

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 1), torch.nn.BatchNorm3d(2),
                                           torch.nn.AdaptiveAvgPool3d(1), torch.nn.Flatten())
        self.mlp = torch.nn.Linear(2, 1)

    def forward_features(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.mlp(self.forward_features(x))


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["patient_id", "path", "label", "split"])
            for i, split in enumerate(["train"] * 4 + ["val"] * 2 + ["test"] * 2):
                hu = np.linspace(-1000, 400, 64, dtype=np.float32).reshape(4, 4, 4)
                nib.save(nib.Nifti1Image(hu, np.eye(4)), root / f"{i}.nii.gz")
                writer.writerow([str(i), f"{i}.nii.gz", i % 2, split])
        records = read_manifest(manifest)
        stats = prepare_cache(records, root / "cache", (4, 4, 4))
        expected = window_array(hu, "lung")
        assert abs(stats["lung"]["mean"] - expected.mean()) < 1e-6
        loaders = {s: DataLoader(WindowDataset(records, root / "cache", stats, s), batch_size=2)
                   for s in ("train", "val")}
        teacher = freeze_teacher(TinyModel())
        original = {k: v.clone() for k, v in teacher.state_dict().items()}
        student = TinyModel()
        before = student.mlp.weight.detach().clone()
        result = train_stage(student, loaders["train"], loaders["val"], "lung",
                             torch.device("cpu"), 1, 1e-3, 0, root / "test.pt", lambda: None,
                             teacher=teacher, teacher_window="mediastinal")
        assert result["history"] and (root / "test.pt").is_file()
        assert not torch.equal(before, student.mlp.weight)
        assert all(torch.equal(v, original[k]) for k, v in teacher.state_dict().items())
        assert all(p.grad is None for p in teacher.parameters())
        s, t = torch.ones(2, 3, requires_grad=True), torch.zeros(2, 3, requires_grad=True)
        feature_loss(s, t).backward()
        assert s.grad is not None and t.grad is None
        with manifest.open("a") as stream:
            stream.write("0,0.nii.gz,0,test\n")
        try:
            read_manifest(manifest)
        except ValueError:
            pass
        else:
            raise AssertionError("Patient leakage was not rejected")
    print("PASS: HU windows, train normalization, checkpoint, student gradients, frozen teacher, split audit")
    if "--gpu" in sys.argv:
        from networks.ssm_nnMamba import nnMambaEncoder
        from networks.window_mamba_adapter import WindowMambaAdapter
        factory, feature_dim = WindowMambaAdapter, 448
        if "--hybrid" in sys.argv:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from regression.networks.hybrid_mamba_attention_regressor import HybridMambaAttentionRegressor
            factory = lambda: HybridMambaAttentionRegressor(
                num_classes=2, depths=(3, 3, 3), head_hidden_dim=256, dropout=0.3)
            feature_dim = 352
        device = torch.device("cuda")
        student = factory().to(device)
        if "--hybrid" not in sys.argv:
            original_model = nnMambaEncoder().to(device).eval()
            original_model.load_state_dict(student.state_dict(), strict=True)
            student.eval()
            with torch.no_grad():
                probe = torch.randn(2, 1, 32, 32, 32, device=device)
                torch.testing.assert_close(original_model(probe), student(probe))
            del original_model, probe
            student.train()
        teacher = freeze_teacher(factory().to(device))
        shape = (112, 136, 112) if "--full-shape" in sys.argv else (32, 32, 32)
        x = torch.randn(2, 1, *shape, device=device)
        original = {k: v.clone() for k, v in teacher.state_dict().items()}
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            features = student.forward_features(x)
            with torch.no_grad():
                target = teacher.forward_features(x * 0.5)
            logits = classification_logit(student, features)
            loss = 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.tensor([0., 1.], device=device)) + 0.5 * feature_loss(features, target)
            loss.backward()
            optimizer.step()
            assert torch.isfinite(loss)
        torch.cuda.synchronize()
        assert all(torch.equal(v, original[k]) for k, v in teacher.state_dict().items())
        assert features.shape == (2, feature_dim)
        print(f"PASS actual nnMamba GPU backward: shape={shape}, batch=2, "
              f"seconds/step={(time.monotonic()-started)/3:.3f}, "
              f"peak_allocated_MiB={torch.cuda.max_memory_allocated()/2**20:.1f}")
        del student, teacher, optimizer, x, features, target, logits, loss, original
        torch.cuda.empty_cache()
    if "--end-to-end" in sys.argv:
        # Persistent artifacts exercise the actual CLI/output contract. These are
        # synthetic test data and must never be interpreted as clinical results.
        output = Path(__file__).resolve().parents[2] / "weights" / "window_distillation"
        output = output / f"synthetic_smoke_{time.time_ns()}"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["patient_id", "path", "label", "split"])
                for i, split in enumerate(["train"] * 4 + ["val"] * 4 + ["test"] * 4):
                    hu = np.linspace(-1000, 400, 32**3, dtype=np.float32).reshape(32, 32, 32)
                    hu = hu + (i % 2) * 25
                    nib.save(nib.Nifti1Image(hu, np.eye(4)), root / f"{i}.nii.gz")
                    writer.writerow([f"synthetic_{i}", f"{i}.nii.gz", i % 2, split])
            subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1]
                            / "train_window_distillation.py"), "--manifest", str(manifest),
                            "--output", str(output), "--hu-confirmed", "--shape", "32", "32", "32",
                            "--epochs", "1", "--minutes", "5"], check=True)
        report = json.loads((output / "results.json").read_text())
        assert report["status"] == "complete" and len(report["stages"]) == 4
        assert (output / "test_predictions.csv").is_file()
        print(f"PASS full four-stage CLI and artifacts (SYNTHETIC ONLY): {output}")


if __name__ == "__main__":
    main()
