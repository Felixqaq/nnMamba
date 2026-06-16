"""3D Grad-CAM utilities for regression and late-fusion classification models."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


DEFAULT_TARGET_LAYERS = {
    "hybrid": "attention_layers",
    "hybrid_mamba_attention": "attention_layers",
    "hybrid_mamba_attention_regressor": "attention_layers",
    "hybrid_mamba_tapct_fusion": "image_encoder.attention_layers",
    "hybrid_tapct_fusion": "image_encoder.attention_layers",
    "tapct_late_fusion": "image_encoder.attention_layers",
    "hybrid_mamba_tapct_abmil_fusion": "image_encoder.attention_layers",
    "hybrid_tapct_abmil_fusion": "image_encoder.attention_layers",
    "tapct_abmil_fusion": "image_encoder.attention_layers",
    "hybrid_mamba_tapct_attention_fusion": "image_encoder.attention_layers",
    "hybrid_tapct_attention_fusion": "image_encoder.attention_layers",
    "tapct_attention_fusion": "image_encoder.attention_layers",
    "mamba": "stage3",
    "nnmamba": "stage3",
    "nnmamba_regressor": "stage3",
}


class GradCAM:
    """Compute Grad-CAM volumes from a 3D spatial feature layer."""

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
        """Remove registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(
        self,
        model_input: Any,
        target_class: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized CAM volumes and raw model outputs."""
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        outputs = self.model(model_input)
        scores = _select_scores(outputs, target_class)
        scores.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        if self.activations.ndim != 5 or self.gradients.ndim != 5:
            raise ValueError(
                "Grad-CAM target layer must produce [batch, channels, depth, height, width]."
            )

        spatial_size = _input_spatial_size(model_input)
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cams = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cams = F.interpolate(
            cams,
            size=spatial_size,
            mode="trilinear",
            align_corners=False,
        )
        cams = _normalize_cams(cams[:, 0])
        return cams.detach(), _first_tensor(outputs).detach()


def resolve_target_layer(
    model: nn.Module,
    model_name: str | None = None,
    target_layer_name: str | None = None,
) -> tuple[str, nn.Module]:
    """Resolve a Grad-CAM target layer from config or model defaults."""
    modules = dict(model.named_modules())
    if target_layer_name:
        if target_layer_name not in modules:
            raise ValueError(f"Unknown Grad-CAM layer: {target_layer_name}")
        return target_layer_name, modules[target_layer_name]

    default_name = DEFAULT_TARGET_LAYERS.get((model_name or "").lower())
    if default_name and default_name in modules:
        return default_name, modules[default_name]

    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv3d):
            return name, module

    raise ValueError(
        "No 3D spatial layer found for Grad-CAM. Set gradcam.target_layer to a "
        "module that returns [B, C, D, H, W]."
    )


def generate_gradcam(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: Path,
    model_name: str,
    dataset=None,
    fold_indices: Iterable[int] | None = None,
    class_names: list[str] | None = None,
    max_samples: int = 8,
    target_layer_name: str | None = None,
    target_class: int | None = None,
    task_type: str = "angle",
) -> list[dict]:
    """Generate Grad-CAM overlay PNGs and a manifest for validation samples."""
    if max_samples <= 0:
        return []

    layer_name, target_layer = resolve_target_layer(
        model,
        model_name=model_name,
        target_layer_name=target_layer_name,
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    cam = GradCAM(model, target_layer)
    fold_indices = list(fold_indices or [])
    manifest: list[dict] = []
    seen = 0

    try:
        model.eval()
        for batch in dataloader:
            batch_size = int(_batch_ct(batch).shape[0])
            for item_idx in range(batch_size):
                if len(manifest) >= max_samples:
                    _write_manifest(
                        save_dir, layer_name, target_class, task_type, manifest
                    )
                    return manifest

                dataset_idx = fold_indices[seen] if seen < len(fold_indices) else seen
                sample_input = _slice_model_input(batch, item_idx, device)
                ct = _input_ct(sample_input)
                ct.requires_grad_(True)
                if isinstance(sample_input, dict):
                    sample_input["ct"] = ct
                else:
                    sample_input = ct

                cam_volume, outputs = cam(sample_input, target_class=target_class)
                output_cpu = outputs.detach().cpu()
                source = _dataset_record(dataset, int(dataset_idx))
                title, sample_meta = _sample_metadata(
                    batch=batch,
                    item_idx=item_idx,
                    outputs=output_cpu,
                    task_type=task_type,
                    class_names=class_names or [],
                )

                filename = f"sample_{len(manifest):03d}_{_safe_name(source.get('patient_id'))}.png"
                save_gradcam_overlay(
                    volume=ct[0, 0].detach().cpu().numpy(),
                    cam=cam_volume[0].cpu().numpy(),
                    path=save_dir / filename,
                    title=title,
                )
                manifest.append(
                    {
                        "image": filename,
                        "dataset_index": int(dataset_idx),
                        "patient_id": source.get("patient_id"),
                        "source": source.get("path"),
                        **sample_meta,
                    }
                )
                seen += 1
    finally:
        cam.close()

    _write_manifest(save_dir, layer_name, target_class, task_type, manifest)
    return manifest


def save_gradcam_overlay(
    volume: np.ndarray,
    cam: np.ndarray,
    path: Path,
    title: str = "",
    alpha: float = 0.45,
) -> None:
    """Save axial/coronal/sagittal overlays centered on the CAM maximum."""
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
    if isinstance(value, dict):
        for item in value.values():
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError("Expected tensor output from model or target layer.")


def _select_scores(outputs: Any, target_class: int | None) -> torch.Tensor:
    output = _first_tensor(outputs).float()
    if output.ndim == 0:
        return output.view(1)
    if output.ndim == 1:
        return output
    if output.ndim != 2:
        output = output.view(output.shape[0], -1)
    if output.shape[1] == 1:
        return output[:, 0]
    if target_class is None:
        return output.gather(1, output.argmax(dim=1, keepdim=True))[:, 0]
    return output[:, int(target_class)]


def _batch_ct(batch: dict[str, Any]) -> torch.Tensor:
    ct = batch.get("ct")
    if ct is None:
        ct = batch.get("mri")
    if ct is None:
        raise KeyError("Batch is missing ct/mri tensor for Grad-CAM.")
    return ct


def _slice_model_input(batch: dict[str, Any], item_idx: int, device: torch.device):
    ct = _batch_ct(batch)[item_idx : item_idx + 1].to(device, non_blocking=True)
    embedding = batch.get("tapct_embedding")
    if embedding is None:
        return ct
    model_input = {
        "ct": ct,
        "tapct_embedding": embedding[item_idx : item_idx + 1].to(
            device, non_blocking=True
        ),
    }
    if batch.get("tapct_mask") is not None:
        model_input["tapct_mask"] = batch["tapct_mask"][item_idx : item_idx + 1].to(
            device, non_blocking=True
        )
    return model_input


def _input_ct(model_input: Any) -> torch.Tensor:
    if isinstance(model_input, dict):
        return model_input["ct"]
    return model_input


def _input_spatial_size(model_input: Any) -> tuple[int, int, int]:
    return tuple(int(size) for size in _input_ct(model_input).shape[2:])


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


def _dataset_record(dataset, index: int) -> dict[str, str | None]:
    if dataset is None or not hasattr(dataset, "records") or index >= len(dataset.records):
        return {"patient_id": None, "path": None}
    record = dataset.records[index]
    return {
        "patient_id": str(getattr(record, "patient_id", index)),
        "path": str(getattr(record, "path", "")),
    }


def _sample_metadata(
    batch: dict[str, Any],
    item_idx: int,
    outputs: torch.Tensor,
    task_type: str,
    class_names: list[str],
) -> tuple[str, dict]:
    if outputs.ndim == 1:
        outputs = outputs.unsqueeze(0)
    output = outputs[item_idx if outputs.shape[0] > item_idx else 0]
    if output.ndim > 1:
        output = output.flatten()

    if output.numel() > 1:
        probs = torch.softmax(output.float(), dim=0)
        pred_idx = int(torch.argmax(probs).item())
        true_idx = _batch_int(batch, "label", "target", item_idx)
        true_name = _class_name(class_names, true_idx)
        pred_name = _class_name(class_names, pred_idx)
        probability = float(probs[pred_idx].item())
        title = f"true={true_name} pred={pred_name} p={probability:.3f}"
        return title, {
            "true_label": true_name,
            "pred_label": pred_name,
            "pred_probability": round(probability, 5),
            "logits": [round(float(value), 5) for value in output.tolist()],
        }

    pred_value = float(output.view(-1)[0].item())
    target = _batch_float(batch, "target", "angle", item_idx)
    title = f"{task_type}: target={target:.3f} pred={pred_value:.3f}"
    return title, {
        "target": round(target, 5),
        "prediction": round(pred_value, 5),
    }


def _batch_int(batch: dict[str, Any], primary: str, fallback: str, item_idx: int) -> int:
    value = batch.get(primary)
    if value is None:
        value = batch.get(fallback)
    if torch.is_tensor(value):
        return int(value.view(-1)[item_idx].item())
    return int(value[item_idx])


def _batch_float(batch: dict[str, Any], primary: str, fallback: str, item_idx: int) -> float:
    value = batch.get(primary)
    if value is None:
        value = batch.get(fallback)
    if torch.is_tensor(value):
        return float(value.view(-1)[item_idx].item())
    return float(value[item_idx])


def _class_name(class_names: list[str], index: int) -> str:
    if 0 <= index < len(class_names):
        return class_names[index]
    return str(index)


def _safe_name(value) -> str:
    if value is None:
        return "sample"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _write_manifest(
    save_dir: Path,
    layer_name: str,
    target_class: int | None,
    task_type: str,
    samples: list[dict],
) -> None:
    payload = {
        "target_layer": layer_name,
        "target_class": target_class,
        "task_type": task_type,
        "attribution_scope": "ct_image_branch",
        "note": (
            "For TAP-CT late-fusion models, this Grad-CAM explains the CT image "
            "branch while the frozen TAP-CT embedding remains part of the forward pass."
        ),
        "samples": samples,
    }
    with open(save_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
