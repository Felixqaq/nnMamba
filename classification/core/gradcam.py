"""3D Grad-CAM utilities for classification models."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


DEFAULT_TARGET_LAYERS = {
    "nnmamba": "layer3",
    "densenet": "features.denseblock4.denselayer16.conv2",
}

DEFAULT_TARGET_LAYER_SETS = {
    "nnmamba": ("layer1", "layer2", "layer3"),
}

OUTCOME_ORDER = ("TP", "TN", "FP", "FN")


class GradCAM:
    """Compute Grad-CAM volumes from a spatial 3D feature layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, module, inputs, output) -> None:
        del module, inputs
        self.activations = _first_tensor(output)

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        del module, grad_input
        self.gradients = _first_tensor(grad_output)

    def close(self) -> None:
        """Remove PyTorch hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(
        self,
        inputs: torch.Tensor,
        target_class: int | None = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized CAM volumes and raw model outputs.

        Args:
            inputs: Input tensor with shape ``[B, C, D, H, W]``.
            target_class: Class index to explain. For binary one-output models,
                ``1`` explains the positive class and ``0`` explains the negative
                class by negating the positive-class score.
        """
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        outputs = self.model(inputs)
        scores = _select_scores(outputs, target_class)
        scores.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        if self.activations.ndim != 5 or self.gradients.ndim != 5:
            raise ValueError(
                "Grad-CAM target layer must produce a 5D tensor shaped "
                "[batch, channels, depth, height, width]."
            )

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cams = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cams = F.interpolate(
            cams,
            size=inputs.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        cams = _normalize_cams(cams[:, 0])
        return cams.detach(), outputs.detach()


def resolve_target_layer(
    model: nn.Module,
    model_name: str | None = None,
    target_layer_name: str | None = None,
) -> tuple[str, nn.Module]:
    """Resolve a Grad-CAM target layer from an explicit name or model default."""
    modules = dict(model.named_modules())

    if target_layer_name:
        if target_layer_name not in modules:
            raise ValueError(
                f"Unknown Grad-CAM layer '{target_layer_name}'. "
                "Use one of the names from model.named_modules()."
            )
        return target_layer_name, modules[target_layer_name]

    default_name = DEFAULT_TARGET_LAYERS.get((model_name or "").lower())
    if default_name and default_name in modules:
        return default_name, modules[default_name]

    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv3d):
            return name, module

    raise ValueError(
        "No default 3D spatial Grad-CAM layer was found. "
        "Pass --gradcam-layer with a module that returns [B, C, D, H, W]."
    )


def resolve_target_layers(
    model: nn.Module,
    model_name: str | None = None,
    target_layer_name: str | None = None,
    target_layer_names: Iterable[str] | None = None,
) -> list[tuple[str, nn.Module]]:
    """Resolve one or more Grad-CAM target layers."""
    modules = dict(model.named_modules())
    explicit_names = _normalize_layer_names(target_layer_names)
    if explicit_names:
        layers = []
        for name in explicit_names:
            if name not in modules:
                raise ValueError(
                    f"Unknown Grad-CAM layer '{name}'. "
                    "Use one of the names from model.named_modules()."
                )
            layers.append((name, modules[name]))
        return layers

    if target_layer_name:
        return [
            resolve_target_layer(
                model,
                model_name=model_name,
                target_layer_name=target_layer_name,
            )
        ]

    default_names = DEFAULT_TARGET_LAYER_SETS.get((model_name or "").lower())
    if default_names:
        available = [(name, modules[name]) for name in default_names if name in modules]
        if available:
            return available

    return [
        resolve_target_layer(
            model,
            model_name=model_name,
            target_layer_name=None,
        )
    ]


def generate_gradcam(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    model_name: str,
    dataset=None,
    fold_indices: Iterable[int] | None = None,
    labels: list[str] | None = None,
    max_samples: int = 8,
    target_layer_name: str | None = None,
    target_layer_names: Iterable[str] | None = None,
    target_class: int | None = 1,
    threshold: float = 0.5,
    per_class: int | None = None,
    per_outcome: int | None = None,
) -> list[dict]:
    """Generate Grad-CAM overlay PNGs and a JSON manifest.

    When ``per_class`` is set, render at most that many samples for each true
    class label (e.g. ``per_class=1`` yields one Normal and one Abnormal),
    instead of taking the first ``max_samples`` samples in loader order.

    When ``per_outcome`` is set, render at most that many binary-classification
    examples for each of TP/TN/FP/FN. Outcomes absent from the fold are skipped.
    """
    if max_samples <= 0:
        return []

    layers = resolve_target_layers(
        model,
        model_name=model_name,
        target_layer_name=target_layer_name,
        target_layer_names=target_layer_names,
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    if len(layers) > 1:
        combined: list[dict] = []
        layer_entries = []
        for layer_name, target_layer in layers:
            layer_dir = save_dir / _safe_identifier(layer_name)
            layer_samples = _generate_gradcam_for_layer(
                model=model,
                dataloader=dataloader,
                device=device,
                save_dir=layer_dir,
                target_layer=target_layer,
                layer_name=layer_name,
                dataset=dataset,
                fold_indices=fold_indices,
                labels=labels,
                max_samples=max_samples,
                target_class=target_class,
                threshold=threshold,
                per_class=per_class,
                per_outcome=per_outcome,
            )
            for sample in layer_samples:
                combined.append(
                    {
                        "target_layer": layer_name,
                        "layer_dir": layer_dir.name,
                        **sample,
                    }
                )
            layer_entries.append(
                {
                    "target_layer": layer_name,
                    "directory": layer_dir.name,
                    "sample_count": len(layer_samples),
                    "outcomes_found": sorted(
                        {
                            str(sample["outcome"])
                            for sample in layer_samples
                            if sample.get("outcome")
                        }
                    ),
                }
            )
        _write_multi_layer_manifest(
            save_dir=save_dir,
            layer_entries=layer_entries,
            target_class=target_class,
            samples=combined,
        )
        return combined

    layer_name, target_layer = layers[0]
    return _generate_gradcam_for_layer(
        model=model,
        dataloader=dataloader,
        device=device,
        save_dir=save_dir,
        target_layer=target_layer,
        layer_name=layer_name,
        dataset=dataset,
        fold_indices=fold_indices,
        labels=labels,
        max_samples=max_samples,
        target_class=target_class,
        threshold=threshold,
        per_class=per_class,
        per_outcome=per_outcome,
    )


def _generate_gradcam_for_layer(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    target_layer: nn.Module,
    layer_name: str,
    dataset=None,
    fold_indices: Iterable[int] | None = None,
    labels: list[str] | None = None,
    max_samples: int = 8,
    target_class: int | None = 1,
    threshold: float = 0.5,
    per_class: int | None = None,
    per_outcome: int | None = None,
) -> list[dict]:
    """Generate Grad-CAM overlays for one target layer."""
    save_dir.mkdir(parents=True, exist_ok=True)
    cam = GradCAM(model, target_layer)
    fold_indices = list(fold_indices or [])
    labels = labels or ["class_0", "class_1"]
    manifest = []
    seen = 0
    class_counts: dict[int, int] = {}
    balanced = per_class is not None and per_class > 0
    outcome_counts: dict[str, int] = {}
    outcome_mode = per_outcome is not None and per_outcome > 0

    def _quota_full() -> bool:
        if outcome_mode:
            if len(manifest) >= max_samples:
                return True
            return all(
                outcome_counts.get(outcome, 0) >= int(per_outcome)
                for outcome in OUTCOME_ORDER
            )
        if balanced:
            return all(class_counts.get(i, 0) >= per_class for i in range(len(labels)))
        return len(manifest) >= max_samples

    try:
        model.eval()
        for batch in dataloader:
            images = batch["mri"].to(device)
            targets = batch["label"].flatten().cpu()

            for item_idx in range(images.size(0)):
                if _quota_full():
                    _write_manifest(save_dir, layer_name, target_class, manifest)
                    return manifest

                dataset_idx = (
                    fold_indices[seen] if seen < len(fold_indices) else seen
                )
                true_idx = int(targets[item_idx].item())
                seen += 1

                image = images[item_idx : item_idx + 1].detach().clone()
                with torch.no_grad():
                    raw_outputs = model(image)
                    prob_positive = _positive_probabilities(raw_outputs)[0].cpu()
                pred_idx = int(prob_positive >= threshold)
                outcome = _binary_outcome(true_idx, pred_idx)

                if outcome_mode:
                    if outcome_counts.get(outcome, 0) >= int(per_outcome):
                        continue
                elif balanced and class_counts.get(true_idx, 0) >= per_class:
                    continue

                image.requires_grad_(True)
                cam_volume, outputs = cam(image, target_class=target_class)
                prob_positive = float(_positive_probabilities(outputs)[0].cpu().item())
                source_path = _dataset_path(dataset, dataset_idx)

                outcome_prefix = f"{outcome}_" if outcome_mode else ""
                filename = (
                    f"{outcome_prefix}sample_{len(manifest):03d}_"
                    f"{_safe_name(source_path)}.png"
                )
                output_path = save_dir / filename
                save_gradcam_overlay(
                    volume=image[0, 0].detach().cpu().numpy(),
                    cam=cam_volume[0].cpu().numpy(),
                    path=output_path,
                    title=(
                        f"true={_label_name(labels, true_idx)} "
                        f"pred={_label_name(labels, pred_idx)} "
                        f"p={prob_positive:.3f}"
                    ),
                )

                manifest.append(
                    {
                        "image": filename,
                        "source": str(source_path) if source_path else None,
                        "dataset_index": int(dataset_idx),
                        "true_label": _label_name(labels, true_idx),
                        "pred_label": _label_name(labels, pred_idx),
                        "prob_positive": round(prob_positive, 5),
                        "outcome": outcome if outcome_mode else None,
                    }
                )
                if outcome_mode:
                    outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
                else:
                    class_counts[true_idx] = class_counts.get(true_idx, 0) + 1
    finally:
        cam.close()

    _write_manifest(save_dir, layer_name, target_class, manifest)
    return manifest


def save_gradcam_overlay(
    volume: np.ndarray,
    cam: np.ndarray,
    path: Path,
    title: str = "",
    alpha: float = 0.45,
) -> None:
    """Save axial/coronal/sagittal Grad-CAM overlays for one 3D volume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    volume = _normalize_volume(volume)
    cam = _normalize_array(cam)
    z, y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)

    views = [
        ("Axial", volume[z, :, :], cam[z, :, :]),
        ("Coronal", volume[:, y, :], cam[:, y, :]),
        ("Sagittal", volume[:, :, x], cam[:, :, x]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (view_name, image_slice, cam_slice) in zip(axes, views):
        axis.imshow(np.rot90(image_slice), cmap="gray")
        axis.imshow(np.rot90(cam_slice), cmap="jet", alpha=alpha, vmin=0, vmax=1)
        axis.set_title(view_name)
        axis.axis("off")

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _first_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError("Expected a tensor output from the Grad-CAM target layer.")


def _select_scores(outputs: torch.Tensor, target_class: int | None) -> torch.Tensor:
    outputs = outputs.reshape(outputs.shape[0], -1)
    if outputs.shape[1] == 1:
        scores = outputs[:, 0]
        if target_class == 0:
            return -scores
        return scores

    if target_class is None:
        return outputs.gather(1, outputs.argmax(dim=1, keepdim=True))[:, 0]
    return outputs[:, target_class]


def _normalize_cams(cams: torch.Tensor) -> torch.Tensor:
    flat = cams.flatten(start_dim=1)
    mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (cams - mins) / (maxs - mins + 1e-8)


def _normalize_volume(volume: np.ndarray) -> np.ndarray:
    lower, upper = np.percentile(volume, [1, 99])
    if upper <= lower:
        return np.zeros_like(volume, dtype=np.float32)
    clipped = np.clip(volume, lower, upper)
    return ((clipped - lower) / (upper - lower)).astype(np.float32)


def _normalize_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def _positive_probability(value: torch.Tensor) -> torch.Tensor:
    if 0.0 <= float(value) <= 1.0:
        return value
    return torch.sigmoid(value)


def _positive_probabilities(outputs) -> torch.Tensor:
    output = _first_tensor(outputs).detach().float()
    if output.ndim == 0:
        output = output.view(1, 1)
    elif output.ndim == 1:
        output = output.view(-1, 1)
    elif output.ndim != 2:
        output = output.view(output.shape[0], -1)

    if output.shape[1] == 1:
        return torch.sigmoid(output[:, 0])
    positive_idx = 1 if output.shape[1] > 1 else 0
    return torch.softmax(output, dim=1)[:, positive_idx]


def _binary_outcome(true_idx: int, pred_idx: int) -> str:
    if true_idx == 1 and pred_idx == 1:
        return "TP"
    if true_idx == 0 and pred_idx == 0:
        return "TN"
    if true_idx == 0 and pred_idx == 1:
        return "FP"
    if true_idx == 1 and pred_idx == 0:
        return "FN"
    return f"true{true_idx}_pred{pred_idx}"


def _dataset_path(dataset, dataset_idx: int):
    if dataset is None or not hasattr(dataset, "directories"):
        return None
    if dataset_idx >= len(dataset.directories):
        return None
    return Path(dataset.directories[dataset_idx])


def _safe_name(path) -> str:
    if path is None:
        return "sample"
    path = Path(path)
    name = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _safe_identifier(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "layer"


def _normalize_layer_names(layer_names: Iterable[str] | str | None) -> list[str]:
    if layer_names is None:
        return []
    if isinstance(layer_names, str):
        return [name.strip() for name in layer_names.split(",") if name.strip()]
    return [str(name).strip() for name in layer_names if str(name).strip()]


def _label_name(labels: list[str], index: int) -> str:
    if 0 <= index < len(labels):
        return labels[index]
    return str(index)


def _write_manifest(
    save_dir: Path,
    layer_name: str,
    target_class: int | None,
    samples: list[dict],
) -> None:
    payload = {
        "target_layer": layer_name,
        "target_class": target_class,
        "selection": _selection_summary(samples),
        "samples": samples,
    }
    with open(save_dir / "manifest.json", "w") as f:
        json.dump(payload, f, indent=2)


def _write_multi_layer_manifest(
    save_dir: Path,
    layer_entries: list[dict],
    target_class: int | None,
    samples: list[dict],
) -> None:
    payload = {
        "target_layers": [entry["target_layer"] for entry in layer_entries],
        "target_class": target_class,
        "selection": _selection_summary(samples),
        "layers": layer_entries,
        "samples": samples,
    }
    with open(save_dir / "manifest.json", "w") as f:
        json.dump(payload, f, indent=2)


def _selection_summary(samples: list[dict]) -> dict:
    outcomes = [sample.get("outcome") for sample in samples if sample.get("outcome")]
    if outcomes:
        found = sorted(set(str(outcome) for outcome in outcomes))
        return {
            "mode": "per_outcome",
            "target_outcomes": list(OUTCOME_ORDER),
            "outcomes_found": found,
            "missing_outcomes": [
                outcome for outcome in OUTCOME_ORDER if outcome not in found
            ],
        }
    return {"mode": "loader_order"}
