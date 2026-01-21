"""Configuration loader for nnMamba classification."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import yaml


TaskType = Literal["NC_v_AD", "sMCI_v_pMCI", "Normal_v_COPD", "Normal_v_Abnormal"]
ModelType = Literal["nnmamba", "densenet", "vit", "crate"]


@dataclass
class ModelConfig:
    name: ModelType = "nnmamba"
    num_classes: int = 1
    dropout: float = 0.5


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 12
    learning_rate: float = 0.0001
    weight_decay: float = 0.001
    k_folds: int = 5
    eval_interval: int = 5
    save_interval: int = 10
    seed: int = 42
    warmup_gamma: float = 1.4


@dataclass
class DataConfig:
    image_size: tuple[int, int, int] = (112, 136, 112)
    num_workers: int = 6
    pin_memory: bool = True
    prefetch_factor: int = 4


@dataclass
class PathConfig:
    weights: Path = field(default_factory=lambda: Path("../weights"))
    logs: Path = field(default_factory=lambda: Path("../train_log"))
    figures: Path = field(default_factory=lambda: Path("../figures"))
    graphs: Path = field(default_factory=lambda: Path("../graphs"))


@dataclass
class ResumeConfig:
    enabled: bool = False
    uuid: str | None = None
    start_fold: int = 0


@dataclass
class Config:
    """Main configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    task: TaskType = "Normal_v_Abnormal"
    gpu_device_id: str = "0"

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(
                image_size=tuple(
                    data.get("data", {}).get("image_size", [112, 136, 112])
                ),
                num_workers=data.get("data", {}).get("num_workers", 6),
                pin_memory=data.get("data", {}).get("pin_memory", True),
                prefetch_factor=data.get("data", {}).get("prefetch_factor", 4),
            ),
            paths=PathConfig(
                weights=Path(data.get("paths", {}).get("weights", "../weights")),
                logs=Path(data.get("paths", {}).get("logs", "../train_log")),
                figures=Path(data.get("paths", {}).get("figures", "../figures")),
                graphs=Path(data.get("paths", {}).get("graphs", "../graphs")),
            ),
            resume=ResumeConfig(**data.get("resume", {})),
            task=data.get("task", "Normal_v_Abnormal"),
            gpu_device_id=data.get("gpu", {}).get("device_id", "0"),
        )

    def get_labels(self) -> list[str]:
        """Get label names for current task."""
        labels_map = {
            "NC_v_AD": ["NC", "AD"],
            "sMCI_v_pMCI": ["sMCI", "pMCI"],
            "Normal_v_COPD": ["Normal", "COPD"],
            "Normal_v_Abnormal": ["Normal", "Abnormal"],
        }
        return labels_map[self.task]
