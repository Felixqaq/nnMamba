"""Network architectures for CT regression."""

from .mamba_regressor import MambaAngleRegressor
from .swinunetr_v2_regressor import SwinUNETRV2AngleRegressor

__all__ = ["MambaAngleRegressor", "SwinUNETRV2AngleRegressor"]
