"""Tests for frozen TAP-CT embedding probe targets."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_tapct_embedding_probe import build_probe_command, load_yaml
from scripts.train_embedding_probe import GOLD_STAGE_NAMES, prepare_target, requested_models


def test_gold_probe_target_reads_gold_stage_metadata() -> None:
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    metadata = pd.DataFrame(
        {
            "angle": [160.0, 118.0, 140.0],
            "gold_stage": [0, 3, np.nan],
        }
    )

    x, y, angles, meta, class_names = prepare_target("gold", features, metadata)

    assert x.tolist() == features[:2].tolist()
    assert y.tolist() == [0, 3]
    assert angles.tolist() == [160.0, 118.0]
    assert meta["gold_stage"].tolist() == [0.0, 3.0]
    assert class_names == GOLD_STAGE_NAMES
    assert requested_models("gold", "all") == [
        "logistic",
        "linear_svm",
        "ridge_classifier",
    ]


def test_gold_probe_yaml_runs_existing_embeddings_probe_only() -> None:
    config = load_yaml(Path(__file__).with_name("config.gold.tapct_embedding_probe.yaml"))
    probe_command = build_probe_command(config)

    assert config["run"] == {"extract_embeddings": False, "train_probe": True}
    assert probe_command[probe_command.index("--target") + 1] == "gold"
    assert probe_command[probe_command.index("--n-splits") + 1] == "2"
