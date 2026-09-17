"""Synthetic CUDA end-to-end verification of the 13-stage pipeline."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import run
from experiments.full_window_distillation.data import WINDOWS


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv3d(1, 4, 1), torch.nn.ReLU())
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool3d(1)
        self.last_linear = torch.nn.Linear(4, 1)


def fake_prepare(records, source_records, cache, fingerprint, deadline):
    cache.mkdir(parents=True, exist_ok=True)
    for row in records:
        volume = np.linspace(-1000, 500, 4*8*8, dtype=np.float32).reshape(4, 8, 8)
        np.save(cache / f"{row['patient_id']}.npy", volume + row["label"]*20)
    return {w: {"mean": .5, "std": .25} for w in WINDOWS}


def main() -> None:
    output = Path(__file__).resolve().parents[3] / "weights/window_distillation" / f"hybrid_synthetic_smoke_{time.time_ns()}"
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        rows, sources = [], []
        for i, split in enumerate(["train"]*4 + ["val"]*4 + ["test"]*4):
            path = root / f"{i}.txt"
            path.touch()
            pid, label = f"synthetic{i}", i % 2
            rows.append([pid, str(path), label, split])
            sources.append({"patient_id": pid, "ok": True, "fev1_fvc_pct": 60 if label else 80,
                            "dicom_dir": str(root)})
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["patient_id", "path", "label", "split"])
            writer.writerows(rows)
        source = root / "source.json"
        source.write_text(json.dumps({"records": sources}))
        argv = ["run", "--manifest", str(manifest), "--source-summary", str(source),
                "--output", str(output), "--epochs", "1", "--hours", "0.2"]
        with patch.object(sys, "argv", argv), patch.object(run, "build", Tiny), patch.object(run, "forward", lambda model, x: (model.last_linear(model.adaptive_avg_pool(model.features(x)).flatten(1)).flatten(), model.adaptive_avg_pool(model.features(x)).flatten(1))), patch.object(run, "prepare", fake_prepare):
            run.main()
            run.main()  # completed runs must not train twice
        status = json.loads((output / "status.json").read_text())
        result = json.loads((output / "results.json").read_text())
        assert status["status"] == "complete" and len(status["stages"]) == 13
        assert set(result["ensemble"]) == {"baseline", "distilled", "control"}
        for w in WINDOWS:
            if w != result["teacher_window"]:
                assert status["stages"][f"control_{w}"]["epochs_run"] == status["stages"][f"distilled_{w}"]["epochs_run"]
        assert (output / "test_predictions.csv").is_file()
    print(f"PASS 13-stage synthetic pipeline, teacher cache, stacking and idempotent completion: {output}")


if __name__ == "__main__":
    main()

