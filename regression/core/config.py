"""Configuration loader for nnMamba CT angle regression."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


ModelType = Literal["nnmamba_regressor", "nnmamba"]
LossType = Literal["smooth_l1", "mse", "mae"]
TargetNormType = Literal["zscore", "none"]


@dataclass
class ModelConfig:
    name: ModelType = "nnmamba_regressor"
    in_channels: int = 1
    base_channels: int = 32
    blocks: int = 3
    hidden_dim: int = 128
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    epochs: int = 80
    batch_size: int = 4
    eval_batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    k_folds: int = 5
    eval_interval: int = 5
    save_interval: int = 10
    seed: int = 42
    loss: LossType = "smooth_l1"
    clip_grad_norm: float = 1.0


@dataclass
class DataConfig:
    source_dir: Path = field(default_factory=lambda: Path("../by_angle_all"))
    labels_json: Path = field(
        default_factory=lambda: Path("../patient_angle_classification_by_group.json")
    )
    angle_split_manifest: Path = field(
        default_factory=lambda: Path("../by_angle_all/reclassification_manifest.json")
    )
    manifest: Path = field(
        default_factory=lambda: Path("./datasets/generated/regression_manifest.json")
    )
    image_size: tuple[int, int, int] = (112, 136, 112)
    intensity_window: tuple[float, float] = (-1000.0, 400.0)
    target_normalization: TargetNormType = "zscore"
    cache_data: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2
    angle_bin_count: int = 5


@dataclass
class PathConfig:
    weights: Path = field(default_factory=lambda: Path("./weights"))
    logs: Path = field(default_factory=lambda: Path("./train_log"))
    figures: Path = field(default_factory=lambda: Path("./figures"))
    graphs: Path = field(default_factory=lambda: Path("./graphs"))


@dataclass
class ResumeConfig:
    enabled: bool = False
    uuid: str | None = None
    start_fold: int = 0


@dataclass
class GPUConfig:
    device_id: str = "0"


@dataclass
class Config:
    """Main configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    task: str = "PFT_angle_regression"

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(path)
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        data_section = data.get("data", {})
        paths_section = data.get("paths", {})
        gpu_section = data.get("gpu", {})

        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(
                source_dir=Path(data_section.get("source_dir", "../by_angle_all")),
                labels_json=Path(
                    data_section.get(
                        "labels_json", "../patient_angle_classification_by_group.json"
                    )
                ),
                angle_split_manifest=Path(
                    data_section.get(
                        "angle_split_manifest",
                        "../by_angle_all/reclassification_manifest.json",
                    )
                ),
                manifest=Path(
                    data_section.get(
                        "manifest", "./datasets/generated/regression_manifest.json"
                    )
                ),
                image_size=tuple(data_section.get("image_size", [112, 136, 112])),
                intensity_window=tuple(
                    data_section.get("intensity_window", [-1000.0, 400.0])
                ),
                target_normalization=data_section.get(
                    "target_normalization", "zscore"
                ),
                cache_data=data_section.get("cache_data", True),
                num_workers=data_section.get("num_workers", 0),
                pin_memory=data_section.get("pin_memory", True),
                prefetch_factor=data_section.get("prefetch_factor", 2),
                angle_bin_count=data_section.get("angle_bin_count", 5),
            ),
            paths=PathConfig(
                weights=Path(paths_section.get("weights", "./weights")),
                logs=Path(paths_section.get("logs", "./train_log")),
                figures=Path(paths_section.get("figures", "./figures")),
                graphs=Path(paths_section.get("graphs", "./graphs")),
            ),
            resume=ResumeConfig(**data.get("resume", {})),
            gpu=GPUConfig(device_id=gpu_section.get("device_id", "0")),
            task=data.get("task", "PFT_angle_regression"),
        )
