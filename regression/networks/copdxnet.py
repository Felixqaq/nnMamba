"""COPDxNet -- a light four-block 3D CNN baseline for COPD detection.

Reconstructed from Ghosh et al., "Light Convolutional Neural Network to Detect
Chronic Obstructive Pulmonary Disease (COPDxNet): A Multicenter Model
Development and External Validation Study" (medRxiv 2025.07.30.25332459), which
reports AUC 0.92 on COPDGene and SPIROMICS and 0.82 on NLST low-dose scans.

What the paper states, and this follows:
  * four convolutional blocks, about 5.4 M parameters
  * the whole inspiratory volume is consumed -- no slice or ROI selection
  * intensities clipped to [-1024, 400] HU and mapped to [-1, 1]

What the paper does not publish, and is reconstructed here:
  * per-block channel widths, kernel sizes and the classifier head
  * the training recipe

So this is a faithful reimplementation at the level the paper specifies, not a
bit-exact replica; treat it as a competent light-CNN baseline rather than as a
reproduction of their numbers. Parameter count is tuned to land near the stated
5.4 M so the capacity comparison is honest.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3x3 convolutions with BN and ReLU, then halve the resolution."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(float(dropout)))
        layers.append(nn.MaxPool3d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class COPDxNet(nn.Module):
    """Light four-block 3D CNN for whole-volume COPD classification."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 40,
        dropout: float = 0.3,
        head_hidden_dim: int = 256,
    ):
        super().__init__()
        c1 = int(base_channels)
        c2, c3, c4 = c1 * 2, c1 * 4, c1 * 8
        self.blocks = nn.Sequential(
            ConvBlock(int(in_channels), c1),
            ConvBlock(c1, c2),
            ConvBlock(c2, c3, dropout=dropout * 0.5),
            ConvBlock(c3, c4, dropout=dropout * 0.5),
        )
        # Global pooling rather than a flatten, so the head does not depend on
        # the input dimensions and the same weights accept any volume size.
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Linear(c4, int(head_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden_dim), int(num_classes)),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        final = self.head[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.head(self.forward_features(x))
        return output.squeeze(-1) if output.shape[-1] == 1 else output
