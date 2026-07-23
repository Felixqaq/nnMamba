"""Inline-runner smoke test for scripts/train_production_ensemble.py."""

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import Config
from data.loader import RegressionLoaderHelper as LoaderHelper
from scripts.train_production_ensemble import (
    all_data_train_loader,
    train_one_member,
    write_metrics_json,
)

CONFIG_PATH = Path(__file__).resolve().parent / "config.normal_v_abnormal.imageonly.aug5.ensemble.yaml"


def test_all_data_loader_uses_every_case():
    config = Config.from_yaml(str(CONFIG_PATH))
    helper = LoaderHelper(config)
    n_cases = len(helper.dataset)
    dl = all_data_train_loader(helper)
    assert helper.fold_indices[0][1] == [], "val split must be empty for production training"
    assert len(helper.fold_indices[0][0]) == n_cases, "train split must cover every case"
    assert len(dl) > 0
    print(f"test_all_data_loader_uses_every_case PASS (n={n_cases})")


def test_smoke_train_writes_loadable_checkpoints():
    config = Config.from_yaml(str(CONFIG_PATH))
    helper = LoaderHelper(config)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "release"
        out.mkdir(parents=True)
        seeds = [42, 43]
        for i, seed in enumerate(seeds, start=1):
            model = train_one_member(config, helper, seed=seed, epochs=1, device="cuda")
            torch.save({"state_dict": model.state_dict(), "seed": seed}, out / f"member_{i}.pth")
        write_metrics_json(out, n_cases=len(helper.dataset), epochs=1, seeds=seeds)

        assert (out / "metrics.json").exists()
        meta = json.loads((out / "metrics.json").read_text())
        assert meta["n_training_cases"] == len(helper.dataset)
        assert meta["held_out"] is False

        # must be loadable the way copd-ct-app loads it
        for i in range(1, 3):
            payload = torch.load(out / f"member_{i}.pth", map_location="cpu", weights_only=True)
            assert "state_dict" in payload
    print("test_smoke_train_writes_loadable_checkpoints PASS")


if __name__ == "__main__":
    test_all_data_loader_uses_every_case()
    test_smoke_train_writes_loadable_checkpoints()
