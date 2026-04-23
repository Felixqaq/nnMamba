"""Tests for validation-based early stopping configuration."""

from __future__ import annotations

from pathlib import Path

from core.config import Config
from core.trainer import Trainer


class _DummyLoader:
    """Minimal loader surface needed to instantiate Trainer."""

    def get_class_names(self) -> list[str]:
        return []


def test_gold_config_enables_early_stopping() -> None:
    config = Config.from_yaml(Path(__file__).with_name("config.gold.yaml"))

    assert config.early_stopping.enabled is True
    assert config.early_stopping.patience == 6
    assert config.early_stopping.min_delta == 0.005


def test_early_stopping_uses_min_delta_for_improvement() -> None:
    config = Config.from_yaml(Path(__file__).with_name("config.gold.yaml"))
    trainer = Trainer(config, model_factory=lambda: None, loader_helper=_DummyLoader())

    assert trainer._is_better(0.30, float("-inf")) is True
    assert trainer._is_better(0.304, 0.30) is False
    assert trainer._is_better(0.306, 0.30) is True
    assert trainer._should_stop_early(5) is False
    assert trainer._should_stop_early(6) is True
