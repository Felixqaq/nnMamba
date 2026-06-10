"""Tests for classification loss helpers."""

import torch

from core.trainer import FocalLoss, build_loss


def test_focal_loss_builds_for_multiclass_classification() -> None:
    loss = build_loss(
        "focal",
        is_classification=True,
        classification_mode="multiclass",
        focal_gamma=2.0,
    )
    logits = torch.tensor(
        [
            [2.0, 0.0, -1.0, -2.0],
            [-1.0, 2.0, 0.0, -2.0],
        ]
    )
    targets = torch.tensor([0, 1])

    value = loss(logits, targets)

    assert isinstance(loss, FocalLoss)
    assert torch.isfinite(value)


def test_focal_loss_rejects_ordinal_classification() -> None:
    try:
        build_loss(
            "focal",
            is_classification=True,
            classification_mode="ordinal",
        )
    except ValueError as exc:
        assert "multiclass" in str(exc)
    else:
        raise AssertionError("focal loss should reject ordinal classification")
