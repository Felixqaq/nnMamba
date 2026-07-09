"""Configuration loader for nnMamba CT regression/classification tasks."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


ModelType = Literal[
    "hybrid",
    "hybrid_mamba_attention",
    "hybrid_mamba_tapct_fusion",
    "hybrid_tapct_fusion",
    "tapct_late_fusion",
    "hybrid_mamba_tapct_abmil_fusion",
    "hybrid_tapct_abmil_fusion",
    "tapct_abmil_fusion",
    "hybrid_mamba_tapct_attention_fusion",
    "hybrid_tapct_attention_fusion",
    "tapct_attention_fusion",
    "tapct_abmil",
    "tapct_abmil_classifier",
    "mamba",
    "swinunetr",
]
LossType = Literal[
    "auto",
    "smooth_l1",
    "mse",
    "mae",
    "cross_entropy",
    "ordinal_bce",
    "focal",
]
ClassWeightMode = Literal["none", "balanced"]
ClassificationMode = Literal["multiclass", "ordinal"]
EnsembleVoteType = Literal["majority"]
InputNormType = Literal["zscore", "none"]
TargetNormType = Literal["zscore", "none"]
TargetMode = Literal[
    "angle",
    "gold",
    "gold_severity4",
    "angle_3class",
    "angle_binary_extreme",
    "oi",
    "oi_emphysema",
    "oi_3class",
    "normal_v_abnormal",
]
CLASSIFICATION_TARGET_MODES = {
    "gold",
    "gold_severity4",
    "angle_3class",
    "angle_binary_extreme",
    "oi_emphysema",
    "oi_3class",
    "normal_v_abnormal",
}


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
    tapct_embedding_dim: int = 2304
    tapct_attention_dim: int = 128
    tapct_gated_attention: bool = True
    fusion_projection_dim: int = 128
    fusion_dropout: float = 0.1


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
    classification_mode: ClassificationMode = "multiclass"
    focal_gamma: float = 2.0


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 6
    min_delta: float = 0.005


@dataclass
class EnsembleConfig:
    enabled: bool = False
    member_count: int = 1
    vote_sizes: tuple[int, ...] = (3, 5, 7)
    voting: EnsembleVoteType = "majority"
    split_seed: int | None = None
    member_seeds: tuple[int, ...] = field(default_factory=tuple)
    existing_member_uuids: tuple[str, ...] = field(default_factory=tuple)
    train_missing_members: bool = True


@dataclass
class AugmentationConfig:
    enabled: bool = False
    balance_to_majority: bool = False
    target_per_class: int | None = None
    balance_then_augment: bool = False
    views_per_sample: int = 1
    probability: float = 0.8
    gold_stages: tuple[int, ...] = (2, 3, 4)
    class_indices: tuple[int, ...] | None = None
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
    oi_json: Path = field(default_factory=lambda: Path("./oi_processed.json"))
    oi_threshold: float = 4.38
    oi_thresholds: tuple[float, float] | None = None
    oi_exclude_range: tuple[float, float] | None = None
    target_mode: TargetMode = "angle"
    angle_split_manifest: Path = field(
        default_factory=lambda: Path("../by_angle_all/reclassification_manifest.json")
    )
    manifest: Path = field(
        default_factory=lambda: Path("./datasets/generated/regression_manifest.json")
    )
    tapct_features: Path | None = None
    tapct_feature_key: str = "features"
    tapct_allow_single_instance_fallback: bool = False
    load_ct: bool = True
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
    gold_exclude_class_indices: tuple[int, ...] = field(default_factory=tuple)
    gold_remap_class_indices: bool = False
    reuse_underrepresented_classes_in_folds: bool = False
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
class GradCAMConfig:
    enabled: bool = False
    max_samples: int = 8
    target_layer: str | None = None
    target_class: int | None = None


@dataclass
class GPUConfig:
    device_id: str = "0"


@dataclass
class ExperimentConfig:
    name: str | None = None
    description: str | None = None


@dataclass
class Config:
    """Main configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    gradcam: GradCAMConfig = field(default_factory=GradCAMConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    task: str = "PFT_angle_regression"

    def is_classification_task(self) -> bool:
        """Return whether the current config runs categorical prediction."""
        return self.data.target_mode in CLASSIFICATION_TARGET_MODES

    def is_ordinal_classification(self) -> bool:
        """Return whether classification uses cumulative ordinal thresholds."""
        return self.is_classification_task() and self.training.classification_mode == "ordinal"

    def model_output_dim(self) -> int:
        """Return the number of logits the active model head must emit."""
        if self.is_ordinal_classification():
            return max(1, int(self.model.num_classes) - 1)
        return int(self.model.num_classes)

    def split_seed(self) -> int:
        """Return the seed used only for deterministic fold construction."""
        if self.ensemble.enabled and self.ensemble.split_seed is not None:
            return int(self.ensemble.split_seed)
        return int(self.training.seed)

    @staticmethod
    def _parse_optional_float_range(value) -> tuple[float, float] | None:
        """Parse an optional ``[low, high]`` numeric range from YAML."""
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("oi_exclude_range must be null or a two-value range.")
        low, high = (float(value[0]), float(value[1]))
        if low >= high:
            raise ValueError("oi_exclude_range must satisfy low < high.")
        return low, high

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
        gradcam_section = data.get("gradcam", {}) or {}
        ensemble_section = data.get("ensemble", {}) or {}
        experiment_section = data.get("experiment", {}) or {}
        target_mode = data_section.get("target_mode", "angle")
        member_seeds = ensemble_section.get(
            "member_seeds",
            ensemble_section.get("seeds", []),
        )
        window_size = model_section.get("window_size", 4)
        if isinstance(window_size, list):
            window_size = tuple(window_size)
        gold_exclude_class_indices = tuple(
            int(class_idx)
            for class_idx in data_section.get("gold_exclude_class_indices", [])
        )
        gold_remap_class_indices = bool(
            data_section.get("gold_remap_class_indices", False)
        )
        if target_mode == "gold":
            if gold_remap_class_indices:
                default_num_classes = max(
                    1,
                    5
                    - len(
                        {
                            class_idx
                            for class_idx in gold_exclude_class_indices
                            if 0 <= class_idx < 5
                        }
                    ),
                )
            else:
                default_num_classes = 5
            default_task = "GOLD_stage_classification"
        elif target_mode == "gold_severity4":
            default_num_classes = 4
            default_task = "GOLD_severity4_classification"
        elif target_mode == "angle_3class":
            default_num_classes = 3
            default_task = "Angle_3class_classification"
        elif target_mode == "angle_binary_extreme":
            default_num_classes = 2
            default_task = "Angle_extreme_binary_classification"
        elif target_mode == "oi":
            default_num_classes = 1
            default_task = "OI_regression"
        elif target_mode == "oi_emphysema":
            default_num_classes = 2
            default_task = "OI_emphysema_classification"
        elif target_mode == "oi_3class":
            default_num_classes = 3
            default_task = "OI_3class_classification"
        elif target_mode == "normal_v_abnormal":
            default_num_classes = 2
            default_task = "Normal_v_Abnormal_classification"
        else:
            default_num_classes = 1
            default_task = "PFT_angle_regression"
        default_pft_json = (
            "./GOLD_2026_classification.json"
            if target_mode in {"gold", "gold_severity4"}
            else "../pft.json"
        )

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
                tapct_embedding_dim=model_section.get("tapct_embedding_dim", 2304),
                tapct_attention_dim=model_section.get("tapct_attention_dim", 128),
                tapct_gated_attention=model_section.get(
                    "tapct_gated_attention", True
                ),
                fusion_projection_dim=model_section.get("fusion_projection_dim", 128),
                fusion_dropout=model_section.get("fusion_dropout", 0.1),
            ),
            training=TrainingConfig(**data.get("training", {})),
            early_stopping=EarlyStoppingConfig(
                **(data.get("early_stopping", {}) or {})
            ),
            ensemble=EnsembleConfig(
                enabled=ensemble_section.get("enabled", False),
                member_count=ensemble_section.get("member_count", 1),
                vote_sizes=tuple(ensemble_section.get("vote_sizes", [3, 5, 7])),
                voting=ensemble_section.get("voting", "majority"),
                split_seed=ensemble_section.get("split_seed"),
                member_seeds=tuple(member_seeds),
                existing_member_uuids=tuple(
                    str(uuid)
                    for uuid in ensemble_section.get("existing_member_uuids", [])
                ),
                train_missing_members=ensemble_section.get(
                    "train_missing_members", True
                ),
            ),
            data=DataConfig(
                source_dir=Path(data_section.get("source_dir", "../by_angle_all")),
                labels_json=Path(
                    data_section.get(
                        "labels_json", "../patient_angle_classification_by_group.json"
                    )
                ),
                pft_json=Path(data_section.get("pft_json", default_pft_json)),
                oi_json=Path(data_section.get("oi_json", "./oi_processed.json")),
                oi_threshold=float(data_section.get("oi_threshold", 4.38)),
                oi_thresholds=cls._parse_optional_float_range(
                    data_section.get("oi_thresholds")
                ),
                oi_exclude_range=cls._parse_optional_float_range(
                    data_section.get("oi_exclude_range")
                ),
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
                tapct_features=(
                    Path(data_section["tapct_features"])
                    if data_section.get("tapct_features")
                    else None
                ),
                tapct_feature_key=data_section.get("tapct_feature_key", "features"),
                tapct_allow_single_instance_fallback=data_section.get(
                    "tapct_allow_single_instance_fallback", False
                ),
                load_ct=data_section.get("load_ct", True),
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
                gold_exclude_class_indices=gold_exclude_class_indices,
                gold_remap_class_indices=gold_remap_class_indices,
                reuse_underrepresented_classes_in_folds=data_section.get(
                    "reuse_underrepresented_classes_in_folds", False
                ),
                augmentation=AugmentationConfig(
                    enabled=augmentation_section.get("enabled", False),
                    balance_to_majority=augmentation_section.get(
                        "balance_to_majority", False
                    ),
                    target_per_class=augmentation_section.get("target_per_class"),
                    balance_then_augment=augmentation_section.get(
                        "balance_then_augment", False
                    ),
                    views_per_sample=augmentation_section.get("views_per_sample", 1),
                    probability=augmentation_section.get("probability", 0.8),
                    gold_stages=tuple(
                        augmentation_section.get("gold_stages", [2, 3, 4])
                    ),
                    class_indices=(
                        tuple(augmentation_section["class_indices"])
                        if augmentation_section.get("class_indices") is not None
                        else None
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
            gradcam=GradCAMConfig(
                enabled=gradcam_section.get("enabled", False),
                max_samples=gradcam_section.get("max_samples", 8),
                target_layer=gradcam_section.get("target_layer"),
                target_class=gradcam_section.get("target_class"),
            ),
            gpu=GPUConfig(device_id=gpu_section.get("device_id", "0")),
            experiment=ExperimentConfig(
                name=experiment_section.get("name"),
                description=experiment_section.get("description"),
            ),
            task=data.get("task", default_task),
        )
