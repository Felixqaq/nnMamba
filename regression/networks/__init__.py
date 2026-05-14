"""Network architectures for CT regression."""

from .hybrid_mamba_attention_regressor import HybridMambaAttentionRegressor
from .hybrid_mamba_tapct_abmil_fusion_regressor import (
    HybridMambaTapctABMILFusionRegressor,
)
from .hybrid_mamba_tapct_attention_fusion_regressor import (
    HybridMambaTapctAttentionFusionRegressor,
)
from .hybrid_mamba_tapct_fusion_regressor import HybridMambaTapctFusionRegressor
from .mamba_regressor import MambaAngleRegressor
from .swinunetr_v2_regressor import SwinUNETRV2AngleRegressor
from .tapct_abmil_classifier import TapctABMILClassifier

__all__ = [
    "HybridMambaAttentionRegressor",
    "HybridMambaTapctABMILFusionRegressor",
    "HybridMambaTapctAttentionFusionRegressor",
    "HybridMambaTapctFusionRegressor",
    "MambaAngleRegressor",
    "SwinUNETRV2AngleRegressor",
    "TapctABMILClassifier",
]
