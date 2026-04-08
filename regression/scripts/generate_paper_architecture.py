#!/usr/bin/env python3
"""Generate a paper-style simple SVG architecture diagram from model code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Config
from models import build_model
from networks.mamba_regressor import DownsampleStage, ResidualMambaBlock
from networks.swinunetr_v2_regressor import SwinUNETRV2AngleRegressor


def _resolve_path(path: Path, base_dir: Path) -> Path:
    """Resolve a config path relative to the config file directory."""
    return path if path.is_absolute() else (base_dir / path).resolve()


def _conv3d_out_dim(size: int, conv: nn.Conv3d, axis: int) -> int:
    """Compute one output dimension of a 3D convolution."""
    kernel = conv.kernel_size[axis]
    stride = conv.stride[axis]
    padding = conv.padding[axis]
    dilation = conv.dilation[axis]
    return ((size + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1


def _conv3d_out_shape(shape: tuple[int, int, int], conv: nn.Conv3d) -> tuple[int, int, int]:
    """Compute the 3D output shape of a Conv3d layer."""
    return tuple(_conv3d_out_dim(size, conv, axis) for axis, size in enumerate(shape))


def _count_mamba_blocks(stage: DownsampleStage) -> int:
    """Count ResidualMambaBlock instances in a stage."""
    return sum(isinstance(module, ResidualMambaBlock) for module in stage.stage)


def _shape_text(channels: int, spatial: tuple[int, int, int]) -> str:
    """Render a compact channels x depth x height x width string."""
    d, h, w = spatial
    return f"{channels} x {d} x {h} x {w}"


def _head_text(head: nn.Sequential) -> str:
    """Render the MLP head dimensions from Linear modules."""
    dims: list[int] = []
    for module in head:
        if isinstance(module, nn.Linear):
            if not dims:
                dims.append(int(module.in_features))
            dims.append(int(module.out_features))
    return " -> ".join(str(dim) for dim in dims)


def _inspect_swin_model(
    model: SwinUNETRV2AngleRegressor,
    input_channels: int,
    input_spatial: tuple[int, int, int],
) -> dict[str, object]:
    """Extract SwinUNETR v2 encoder metadata using a dummy forward pass."""
    with torch.inference_mode():
        dummy = torch.zeros(1, input_channels, *input_spatial)
        padded = model._pad_to_valid_shape(dummy)
        hidden_states = model.backbone(padded, normalize=True)

    stage_depths = [
        0,
        len(model.backbone.layers1[0].blocks),
        len(model.backbone.layers2[0].blocks),
        len(model.backbone.layers3[0].blocks),
        len(model.backbone.layers4[0].blocks),
    ]
    stage_names = [
        "Patch Embed",
        "Swin Stage 1",
        "Swin Stage 2",
        "Swin Stage 3",
        "Swin Stage 4",
    ]
    stages = []
    for name, depth, feature in zip(stage_names, stage_depths, hidden_states):
        detail = (
            f"patch {model.patch_size[0]} x {model.patch_size[1]} x {model.patch_size[2]}"
            if depth == 0
            else f"{depth} x Swin blocks"
        )
        stages.append(
            {
                "name": name,
                "detail": detail,
                "channels": int(feature.shape[1]),
                "spatial": tuple(int(size) for size in feature.shape[2:]),
            }
        )

    return {
        "kind": "swin",
        "model_name": type(model).__name__,
        "input_channels": input_channels,
        "input_spatial": input_spatial,
        "padded_spatial": tuple(int(size) for size in padded.shape[2:]),
        "window_size": tuple(int(size) for size in model.window_size),
        "patch_size": tuple(int(size) for size in model.patch_size),
        "stages": stages,
        "head_dims": _head_text(model.head),
    }


def _inspect_model(config: Config) -> dict[str, object]:
    """Build the model and extract stage metadata without running a forward pass."""
    model = build_model(config.model)

    input_channels = int(config.model.in_channels)
    input_spatial = tuple(int(dim) for dim in config.data.image_size)

    if isinstance(model, SwinUNETRV2AngleRegressor):
        return _inspect_swin_model(model, input_channels, input_spatial)

    if not hasattr(model, "stem") or not hasattr(model, "stage1"):
        raise TypeError(f"Unsupported model type for this script: {type(model).__name__}")

    stem_conv = model.stem[0]
    if not isinstance(stem_conv, nn.Conv3d):
        raise TypeError("Expected model.stem[0] to be Conv3d.")
    stem_spatial = _conv3d_out_shape(input_spatial, stem_conv)

    stages = []
    current_spatial = stem_spatial
    for name in ("stage1", "stage2", "stage3"):
        stage = getattr(model, name)
        if not isinstance(stage, DownsampleStage):
            raise TypeError(f"Expected {name} to be DownsampleStage.")
        reducer = stage.stage[0][0]
        if not isinstance(reducer, nn.Conv3d):
            raise TypeError(f"Expected {name}.stage[0][0] to be Conv3d.")
        current_spatial = _conv3d_out_shape(current_spatial, reducer)
        stages.append(
            {
                "name": name.capitalize().replace("Stage", "Stage "),
                "stride": reducer.stride[0],
                "channels": int(reducer.out_channels),
                "spatial": current_spatial,
                "blocks": _count_mamba_blocks(stage),
            }
        )

    return {
        "kind": "mamba",
        "model_name": type(model).__name__,
        "input_channels": input_channels,
        "input_spatial": input_spatial,
        "stem_channels": int(stem_conv.out_channels),
        "stem_kernel": stem_conv.kernel_size[0],
        "stem_stride": stem_conv.stride[0],
        "stem_spatial": stem_spatial,
        "stages": stages,
        "head_dims": _head_text(model.head),
    }


def _swin_svg_template(info: dict[str, object], title: str) -> str:
    """Render a compact SVG for the SwinUNETR v2 regression backbone."""
    input_channels = int(info["input_channels"])
    input_spatial = tuple(info["input_spatial"])
    padded_spatial = tuple(info["padded_spatial"])
    stages = list(info["stages"])
    head_dims = str(info["head_dims"])
    patch_size = tuple(info["patch_size"])
    window_size = tuple(info["window_size"])

    title_block = ""
    if title:
        title_block = f"""
  <text x="72" y="72" class="title">{title}</text>
"""

    stage_x = [306, 494, 682, 870, 1058]
    stage_w = [164, 164, 164, 164, 186]
    gradients = ["box1", "box2", "box3", "box4", "box5"]
    stage_blocks = []
    arrows = []
    for idx, stage in enumerate(stages):
        x = stage_x[idx]
        width = stage_w[idx]
        cx = x + width // 2
        spatial = tuple(stage["spatial"])
        stage_blocks.append(
            f"""
  <rect x="{x}" y="236" width="{width}" height="188" rx="24" fill="url(#{gradients[idx]})" filter="url(#shadow)"/>
  <text x="{cx}" y="286" class="label">{stage["name"]}</text>
  <text x="{cx}" y="319" class="small">{stage["detail"]}</text>
  <text x="{cx}" y="355" class="small">{int(stage["channels"])} x {spatial[0]} x</text>
  <text x="{cx}" y="378" class="small">{spatial[1]} x {spatial[2]}</text>
"""
        )
        if idx > 0:
            arrows.append(
                f'<path d="M{stage_x[idx - 1] + stage_w[idx - 1]} 330H{x}" class="arrow"/>'
            )

    padding_note = (
        "No padding needed before the encoder"
        if padded_spatial == input_spatial
        else f"Replicate-pad to {padded_spatial[0]} x {padded_spatial[1]} x {padded_spatial[2]} before the encoder"
    )

    return f"""<svg width="1500" height="780" viewBox="0 0 1500 780" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1500" y2="780" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#F9F6EF"/>
      <stop offset="1" stop-color="#EDF4FA"/>
    </linearGradient>
    <linearGradient id="box0" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#4B6E97"/>
      <stop offset="1" stop-color="#6B8DB4"/>
    </linearGradient>
    <linearGradient id="box1" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#3C7E73"/>
      <stop offset="1" stop-color="#56A897"/>
    </linearGradient>
    <linearGradient id="box2" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#C98944"/>
      <stop offset="1" stop-color="#E2A96B"/>
    </linearGradient>
    <linearGradient id="box3" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#C86458"/>
      <stop offset="1" stop-color="#E08A7E"/>
    </linearGradient>
    <linearGradient id="box4" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#7D5B9E"/>
      <stop offset="1" stop-color="#9E7BC2"/>
    </linearGradient>
    <linearGradient id="box5" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#52637B"/>
      <stop offset="1" stop-color="#70849D"/>
    </linearGradient>
    <linearGradient id="poolBox" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#EEF3F8"/>
      <stop offset="1" stop-color="#F9FBFD"/>
    </linearGradient>
    <marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto">
      <path d="M0 0L14 7L0 14V9H8V5H0V0Z" fill="#65788D"/>
    </marker>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="9" flood-color="#233445" flood-opacity="0.12"/>
    </filter>
    <style>
      .title {{ font: 700 28px 'DejaVu Sans', Arial, sans-serif; fill: #1C2731; }}
      .label {{ font: 700 18px 'DejaVu Sans', Arial, sans-serif; fill: #FFFFFF; text-anchor: middle; }}
      .small {{ font: 500 13px 'DejaVu Sans Mono', 'Courier New', monospace; fill: #FFFFFF; text-anchor: middle; }}
      .midLabel {{ font: 700 17px 'DejaVu Sans', Arial, sans-serif; fill: #263646; text-anchor: middle; }}
      .midSmall {{ font: 500 13px 'DejaVu Sans Mono', 'Courier New', monospace; fill: #485C70; text-anchor: middle; }}
      .note {{ font: 500 13px 'DejaVu Sans', Arial, sans-serif; fill: #485C70; }}
      .arrow {{ stroke: #65788D; stroke-width: 4; fill: none; marker-end: url(#arrow); }}
      .merge {{ stroke: #7A8EA3; stroke-width: 3.5; fill: none; marker-end: url(#arrow); }}
    </style>
  </defs>

  <rect x="0" y="0" width="1500" height="780" fill="url(#bg)"/>
{title_block}

  <rect x="54" y="100" width="1392" height="640" rx="28" fill="#FFFFFF" stroke="#D7E0E9" stroke-width="2"/>

  <rect x="92" y="248" width="176" height="176" rx="24" fill="url(#box0)" filter="url(#shadow)"/>
  <text x="180" y="299" class="label">Input</text>
  <text x="180" y="332" class="small">{input_channels} x {input_spatial[0]} x</text>
  <text x="180" y="355" class="small">{input_spatial[1]} x {input_spatial[2]}</text>
  <text x="180" y="382" class="small">CT volume</text>

  <path d="M268 330H306" class="arrow"/>
  {''.join(stage_blocks)}
  {''.join(arrows)}

  <rect x="510" y="504" width="480" height="78" rx="20" fill="url(#poolBox)" stroke="#D6E0EA" stroke-width="2"/>
  <text x="750" y="536" class="midLabel">Multi-scale Global Average Pooling</text>
  <text x="750" y="561" class="midSmall">Patch embed + 4 Swin stages</text>

  <path d="M388 424V504" class="merge"/>
  <path d="M576 424V504" class="merge"/>
  <path d="M764 424V504" class="merge"/>
  <path d="M952 424V504" class="merge"/>
  <path d="M1151 424V504" class="merge"/>

  <rect x="560" y="628" width="380" height="76" rx="22" fill="url(#box4)" filter="url(#shadow)"/>
  <text x="750" y="659" class="label">Concat + MLP Head</text>
  <text x="750" y="687" class="small">{head_dims}</text>

  <rect x="1050" y="628" width="184" height="76" rx="22" fill="url(#box0)" filter="url(#shadow)"/>
  <text x="1142" y="659" class="label">Output</text>
  <text x="1142" y="687" class="small">angle value</text>

  <path d="M750 582V628" class="merge"/>
  <path d="M940 666H1050" class="arrow"/>

  <text x="92" y="690" class="note">Patch size: {patch_size[0]} x {patch_size[1]} x {patch_size[2]} | Window size: {window_size[0]} x {window_size[1]} x {window_size[2]}</text>
  <text x="92" y="714" class="note">{padding_note}</text>
</svg>
"""


def _svg_template(info: dict[str, object], title: str) -> str:
    """Render a compact paper-style SVG from extracted stage metadata."""
    if info.get("kind") == "swin":
        return _swin_svg_template(info, title)
    input_channels = int(info["input_channels"])
    input_spatial = tuple(info["input_spatial"])
    stem_channels = int(info["stem_channels"])
    stem_spatial = tuple(info["stem_spatial"])
    stages = list(info["stages"])
    head_dims = str(info["head_dims"])

    s1, s2, s3 = stages
    title_block = ""
    if title:
        title_block = f"""
  <text x="72" y="72" class="title">{title}</text>
"""

    return f"""<svg width="1500" height="760" viewBox="0 0 1500 760" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1500" y2="760" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="#FAF7F1"/>
      <stop offset="1" stop-color="#EEF4FB"/>
    </linearGradient>
    <linearGradient id="box0" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#4B6E97"/>
      <stop offset="1" stop-color="#6B8DB4"/>
    </linearGradient>
    <linearGradient id="box1" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#367F73"/>
      <stop offset="1" stop-color="#53A493"/>
    </linearGradient>
    <linearGradient id="box2" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#CA8346"/>
      <stop offset="1" stop-color="#E0A166"/>
    </linearGradient>
    <linearGradient id="box3" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#C55C59"/>
      <stop offset="1" stop-color="#DC7D78"/>
    </linearGradient>
    <linearGradient id="box4" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#6B4F96"/>
      <stop offset="1" stop-color="#8C72B8"/>
    </linearGradient>
    <linearGradient id="poolBox" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#EEF3F8"/>
      <stop offset="1" stop-color="#F9FBFD"/>
    </linearGradient>
    <marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto">
      <path d="M0 0L14 7L0 14V9H8V5H0V0Z" fill="#65788D"/>
    </marker>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="9" flood-color="#233445" flood-opacity="0.12"/>
    </filter>
    <style>
      .title {{ font: 700 28px 'DejaVu Sans', Arial, sans-serif; fill: #1C2731; }}
      .label {{ font: 700 18px 'DejaVu Sans', Arial, sans-serif; fill: #FFFFFF; text-anchor: middle; }}
      .small {{ font: 500 14px 'DejaVu Sans Mono', 'Courier New', monospace; fill: #FFFFFF; text-anchor: middle; }}
      .midLabel {{ font: 700 17px 'DejaVu Sans', Arial, sans-serif; fill: #263646; text-anchor: middle; }}
      .midSmall {{ font: 500 13px 'DejaVu Sans Mono', 'Courier New', monospace; fill: #485C70; text-anchor: middle; }}
      .arrow {{ stroke: #65788D; stroke-width: 4; fill: none; marker-end: url(#arrow); }}
      .merge {{ stroke: #7A8EA3; stroke-width: 3.5; fill: none; marker-end: url(#arrow); }}
    </style>
  </defs>

  <rect x="0" y="0" width="1500" height="760" fill="url(#bg)"/>
{title_block}

  <rect x="54" y="100" width="1392" height="606" rx="28" fill="#FFFFFF" stroke="#D7E0E9" stroke-width="2"/>

  <rect x="108" y="284" width="150" height="150" rx="24" fill="url(#box0)" filter="url(#shadow)"/>
  <text x="183" y="332" class="label">Input</text>
  <text x="183" y="365" class="small">{input_channels} x {input_spatial[0]} x</text>
  <text x="183" y="388" class="small">{input_spatial[1]} x {input_spatial[2]}</text>
  <text x="183" y="415" class="small">CT volume</text>

  <rect x="330" y="250" width="184" height="184" rx="24" fill="url(#box1)" filter="url(#shadow)"/>
  <text x="422" y="299" class="label">Stem</text>
  <text x="422" y="332" class="small">Conv7 / s={info["stem_stride"]}</text>
  <text x="422" y="359" class="small">GN + GELU</text>
  <text x="422" y="395" class="small">{stem_channels} x {stem_spatial[0]} x</text>
  <text x="422" y="418" class="small">{stem_spatial[1]} x {stem_spatial[2]}</text>

  <rect x="560" y="206" width="224" height="228" rx="24" fill="url(#box1)" filter="url(#shadow)"/>
  <text x="672" y="255" class="label">{s1["name"]}</text>
  <text x="672" y="288" class="small">{s1["blocks"]} x Mamba blocks</text>
  <text x="672" y="324" class="small">{int(s1["channels"])} x {tuple(s1["spatial"])[0]} x</text>
  <text x="672" y="347" class="small">{tuple(s1["spatial"])[1]} x {tuple(s1["spatial"])[2]}</text>
  <text x="672" y="381" class="small">stride {s1["stride"]}</text>

  <rect x="832" y="246" width="198" height="188" rx="24" fill="url(#box2)" filter="url(#shadow)"/>
  <text x="931" y="295" class="label">{s2["name"]}</text>
  <text x="931" y="328" class="small">{s2["blocks"]} x Mamba blocks</text>
  <text x="931" y="364" class="small">{int(s2["channels"])} x {tuple(s2["spatial"])[0]} x</text>
  <text x="931" y="387" class="small">{tuple(s2["spatial"])[1]} x {tuple(s2["spatial"])[2]}</text>
  <text x="931" y="415" class="small">stride {s2["stride"]}</text>

  <rect x="1082" y="246" width="194" height="188" rx="24" fill="url(#box3)" filter="url(#shadow)"/>
  <text x="1179" y="295" class="label">{s3["name"]}</text>
  <text x="1179" y="328" class="small">{s3["blocks"]} x Mamba blocks</text>
  <text x="1179" y="364" class="small">{int(s3["channels"])} x {tuple(s3["spatial"])[0]} x</text>
  <text x="1179" y="387" class="small">{tuple(s3["spatial"])[1]} x {tuple(s3["spatial"])[2]}</text>
  <text x="1179" y="415" class="small">stride {s3["stride"]}</text>

  <path d="M258 359H330" class="arrow"/>
  <path d="M514 342H560" class="arrow"/>
  <path d="M784 342H832" class="arrow"/>
  <path d="M1030 342H1082" class="arrow"/>

  <rect x="650" y="514" width="420" height="68" rx="20" fill="url(#poolBox)" stroke="#D6E0EA" stroke-width="2"/>
  <text x="860" y="542" class="midLabel">Multi-scale Global Average Pooling</text>
  <text x="860" y="566" class="midSmall">Stage 1 + Stage 2 + Stage 3</text>

  <path d="M672 434V514" class="merge"/>
  <path d="M931 434V514" class="merge"/>
  <path d="M1179 434V514" class="merge"/>

  <rect x="682" y="620" width="356" height="74" rx="22" fill="url(#box4)" filter="url(#shadow)"/>
  <text x="860" y="650" class="label">Concat + MLP Head</text>
  <text x="860" y="678" class="small">{head_dims}</text>

  <rect x="1124" y="620" width="164" height="74" rx="22" fill="url(#box0)" filter="url(#shadow)"/>
  <text x="1206" y="650" class="label">Output</text>
  <text x="1206" y="678" class="small">angle value</text>

  <path d="M860 582V620" class="merge"/>
  <path d="M1038 657H1124" class="arrow"/>
</svg>
"""


def _default_output_path(config: Config, config_dir: Path) -> Path:
    """Pick a sensible default output path for the selected regression model."""
    model_key = str(config.model.name).lower()
    if model_key in {"mamba", "nnmamba", "nnmamba_regressor"}:
        rel_path = Path("./docs/assets/mamba_regressor_architecture_auto.svg")
    else:
        rel_path = Path(f"./docs/assets/{config.model.name}_architecture_auto.svg")
    return _resolve_path(rel_path, config_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a paper-style simple SVG architecture figure from the regression model."
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title written into the SVG. Defaults to the model class name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path. Defaults to docs/assets/<model>_architecture_auto.svg.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    config = Config.from_yaml(config_path)
    info = _inspect_model(config)
    title = args.title or str(info["model_name"])

    output_path = (
        args.output
        if args.output is not None
        else _default_output_path(config, config_dir)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_svg_template(info, title), encoding="utf-8")

    print(f"Saved architecture SVG to: {output_path}")
    print(f"Model: {info['model_name']}")
    print(f"Head:  {info['head_dims']}")


if __name__ == "__main__":
    main()
