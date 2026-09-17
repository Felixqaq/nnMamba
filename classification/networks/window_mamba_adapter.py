"""Expose original nnMamba features for optional cross-window distillation.

The inherited modules and state-dict keys are unchanged. The original model and
normal training entry point require no edits and do not import this adapter.
"""

from __future__ import annotations

import torch

from .ssm_nnMamba import nnMambaEncoder


class WindowMambaAdapter(nnMambaEncoder):
    """Original classifier with an explicit pre-head representation interface."""

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.in_conv(x)
        c1 = self.mamba_layer_stem(c1) + c1
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        return torch.cat([self.pooling(feature).flatten(1)
                          for feature in (c2, c3, c4)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.forward_features(x))
