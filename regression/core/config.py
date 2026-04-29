"""Configuration loader for nnMamba CT regression/classification tasks."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


ModelType = Literal[
    "hybrid",
    "hybrid_mamba_attention",
    "mamba",
    "swinunetr",
]
LossType = Literal["auto", "smooth_l1", "mse", "mae", "cross_entropy"]
ClassWeightMode = Literal["none", "balanced"]
InputNormType = Literal["zscore", "none"]
TargetNormType = Literal["zscore", "none"]
TargetMode = Literal["angle", "gold", "angle_3class"]
CLASSIFICATION_TARGET_MODES = {"gold", "angle_3class"}


@dataclass
class ModelConfig:
    name: ModelType = "mamba"
    in_channels: int = 1
    num_classes: int = 1
    base_channels: int = 32
    blocks: int = 3
    hidden_dim: int = 128
    dropout: float = 0.3
    feature_size: int = 24
    depths: tuple[int, int, int, int] = (2, 2, 2, 2)
    num_heads: tuple[int, int, int, int] = (3, 6, 12, 24)
    window_size: int | tuple[int, int, int] = 4
    patch_size: int = 2
    use_checkpoint: bool = False
    use_v2: bool = True
    attn_heads: int = 8
    attn_layers: int = 1
    attn_mlp_ratio: float = 2.0
    attn_dropout: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 80
    batch_size: int = 4
    eval_batch_size: int = 2
    swin_batch_size: int = 4
    swin_eval_batch_size: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    k_folds: int = 5
    eval_interval: int = 5
    save_interval: int = 10
    seed: int = 42
    loss: LossType = "auto"
    clip_grad_norm: float = 1.0
    amp: bool = True
    track_train_metrics: bool = False
    class_weight_mode: ClassWeightMode = "none"


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 6
    min_delta: float = 0.005


@dataclass
class AugmentationConfig:
    enabled: bool = False
    balance_to_majority: bool = False
    probability: float = 0.8
    gold_stages: tuple[int, ...] = (2, 3, 4)
    rotation_degrees: float = 7.0
    translation_fraction: float = 0.05
    scale_range: tuple[float, float] = (0.95, 1.05)
    intensity_scale_range: tuple[float, float] = (0.95, 1.05)
    intensity_shift_range: tuple[float, float] = (-25.0, 25.0)
    noise_std: float = 8.0


@dataclass
class DataConfig:
    source_dir: Path = field(default_factory=lambda: Path("../by_angle_all"))
    labels_json: Path = field(
        default_factory=lambda: Path("../patient_angle_classification_by_group.json")
    )
    pft_json: Path = field(default_factory=lambda: Path("../pft.json"))
    target_mode: TargetMode = "angle"
    angle_split_manifest: Path = field(
        default_factory=lambda: Path("../by_angle_all/reclassification_manifest.json")
    )
    manifest: Path = field(
        default_factory=lambda: Path("./datasets/generated/regression_manifest.json")
    )
    image_size: tuple[int, int, int] = (112, 136, 112)
    intensity_window: tuple[float, float] = (-1000.0, 400.0)
    input_normalization: InputNormType = "zscore"
    target_normalization: TargetNormType = "zscore"
    cache_data: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2
    angle_bin_count: int = 5
    balanced_sampling: bool = False
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


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
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    task: str = "PFT_angle_regression"

    def is_classification_task(self) -> bool:
        """Return whether the current config runs categorical prediction."""
        return self.data.target_mode in CLASSIFICATION_TARGET_MODES

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(path)
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        model_section = data.get("model", {})
        data_section = data.get("data", {})
        augmentation_section = data_section.get("augmentation", {})
        paths_section = data.get("paths", {})
        gpu_section = data.get("gpu", {})
        target_mode = data_section.get("target_mode", "angle")
        window_size = model_section.get("window_size", 4)
        if isinstance(window_size, list):
            window_size = tuple(window_size)
        if target_mode == "gold":
            default_num_classes = 4
            default_task = "GOLD_stage_classification"
        elif target_mode == "angle_3class":
            default_num_classes = 3
            default_task = "Angle_3class_classification"
        else:
            default_num_classes = 1
            default_task = "PFT_angle_regression"

        return cls(
            model=ModelConfig(
                name=model_section.get("name", "mamba"),
                in_channels=model_section.get("in_channels", 1),
                num_classes=model_section.get("num_classes", default_num_classes),
                base_channels=model_section.get("base_channels", 32),
                blocks=model_section.get("blocks", 3),
                hidden_dim=model_section.get("hidden_dim", 128),
                dropout=model_section.get("dropout", 0.3),
                feature_size=model_section.get("feature_size", 24),
                depths=tuple(model_section.get("depths", [2, 2, 2, 2])),
                num_heads=tuple(model_section.get("num_heads", [3, 6, 12, 24])),
                window_size=window_size,
                patch_size=model_section.get("patch_size", 2),
                use_checkpoint=model_section.get("use_checkpoint", False),
                use_v2=model_section.get("use_v2", True),
                attn_heads=model_section.get("attn_heads", 8),
                attn_layers=model_section.get("attn_layers", 1),
                attn_mlp_ratio=model_section.get("attn_mlp_ratio", 2.0),
                attn_dropout=model_section.get("attn_dropout", 0.1),
            ),
            training=TrainingConfig(**data.get("training", {})),
            early_stopping=EarlyStoppingConfig(
                **(data.get("early_stopping", {}) or {})
            ),
            data=DataConfig(
                source_dir=Path(data_section.get("source_dir", "../by_angle_all")),
                labels_json=Path(
                    data_section.get(
                        "labels_json", "../patient_angle_classification_by_group.json"
                    )
                ),
                pft_json=Path(data_section.get("pft_json", "../pft.json")),
                target_mode=target_mode,
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
                input_normalization=data_section.get(
                    "input_normalization", "zscore"
                ),
                target_normalization=data_section.get(
                    "target_normalization", "zscore"
                ),
                cache_data=data_section.get("cache_data", True),
                num_workers=data_section.get("num_workers", 0),
                pin_memory=data_section.get("pin_memory", True),
                prefetch_factor=data_section.get("prefetch_factor", 2),
                angle_bin_count=data_section.get("angle_bin_count", 5),
                balanced_sampling=data_section.get("balanced_sampling", False),
                augmentation=AugmentationConfig(
                    enabled=augmentation_section.get("enabled", False),
                    balance_to_majority=augmentation_section.get(
                        "balance_to_majority", False
                    ),
                    probability=augmentation_section.get("probability", 0.8),
                    gold_stages=tuple(
                        augmentation_section.get("gold_stages", [2, 3, 4])
                    ),
                    rotation_degrees=augmentation_section.get("rotation_degrees", 7.0),
                    translation_fraction=augmentation_section.get(
                        "translation_fraction", 0.05
                    ),
                    scale_range=tuple(
                        augmentation_section.get("scale_range", [0.95, 1.05])
                    ),
                    intensity_scale_range=tuple(
                        augmentation_section.get(
                            "intensity_scale_range", [0.95, 1.05]
                        )
                    ),
                    intensity_shift_range=tuple(
                        augmentation_section.get(
                            "intensity_shift_range", [-25.0, 25.0]
                        )
                    ),
                    noise_std=augmentation_section.get("noise_std", 8.0),
                ),
            ),
            paths=PathConfig(
                weights=Path(paths_section.get("weights", "./weights")),
                logs=Path(paths_section.get("logs", "./train_log")),
                figures=Path(paths_section.get("figures", "./figures")),
                graphs=Path(paths_section.get("graphs", "./graphs")),
            ),
            resume=ResumeConfig(**data.get("resume", {})),
            gpu=GPUConfig(device_id=gpu_section.get("device_id", "0")),
            task=data.get("task", default_task),
        )
