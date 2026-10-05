"""Attention-pooling regressor whose scorer starts random instead of at zero.

Identical to HybridMambaAttnPoolRegressor except for the initialisation of each
pool's final scoring layer: a small random normal instead of exact zeros. It is
meant to be trained with train_ratio_regression_guided.py, which adds a lung
prior and an entropy penalty on the attention maps; the model itself carries no
guidance and needs no lung mask at inference.

Why the init changed, and what it is not expected to do: under zero init the
content and gate layers receive exactly zero gradient on the first step only --
AdamW rescales every later gradient to a step of about one learning rate, so the
"zero-init deadlock" does not persist. The 2026-09-29 run showed the scorer
moving ~3% of what AdamW allowed because its gradients had no consistent
direction, not because they were blocked. The random start is kept because it
costs nothing; the guidance terms in the trainer are what supply a direction.
"""

from __future__ import annotations

import torch.nn as nn

from .hybrid_mamba_attnpool_regressor import HybridMambaAttnPoolRegressor

SCORE_INIT_STD = 0.02


class HybridMambaGuidedAttnPoolRegressor(HybridMambaAttnPoolRegressor):
    """HybridMambaAttnPoolRegressor with a small random scorer initialisation."""

    def __init__(self, *args, score_init_std: float = SCORE_INIT_STD, **kwargs):
        super().__init__(*args, **kwargs)
        for pool in self.attn_pools:
            nn.init.normal_(pool.score.weight, mean=0.0, std=float(score_init_std))
            nn.init.zeros_(pool.score.bias)
