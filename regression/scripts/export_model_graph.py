#!/usr/bin/env python3
"""Export the regression model for graph inspection tools such as Netron.

Examples:
    cd regression
    conda run -n nnMamba python scripts/export_model_graph.py --config config.yaml
    conda run -n nnMamba python scripts/export_model_graph.py --config config.yaml --uuid <run_uuid> --fold 1
    conda run -n nnMamba python scripts/export_model_graph.py --config config.yaml --format both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.checkpoints import load_checkpoint
from core.config import Config
from models import build_model


def _resolve_path(path: Path, base_dir: Path) -> Path:
    """Resolve relative paths against the config directory."""
    return path if path.is_absolute() else (base_dir / path).resolve()


def _build_dummy_input(config: Config, batch_size: int, device: torch.device) -> torch.Tensor:
    """Create a synthetic input that matches the configured CT shape."""
    image_size = tuple(int(dim) for dim in config.data.image_size)
    in_channels = int(config.model.in_channels)
    return torch.randn(batch_size, in_channels, *image_size, device=device)


def _load_model(
    config: Config,
    config_dir: Path,
    device: torch.device,
    run_uuid: str | None,
    fold: int,
) -> tuple[torch.nn.Module, Path | None]:
    """Build the model and optionally load a trained checkpoint."""
    model = build_model(config.model, device=device)
    checkpoint_path = None

    if run_uuid:
        weights_root = _resolve_path(config.paths.weights, config_dir)
        checkpoint_path = (
            weights_root / config.task / run_uuid / f"fold{fold}_best_weight.pth"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        load_checkpoint(checkpoint_path, model, device)

    model.eval()
    return model, checkpoint_path


def export_torchscript(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
) -> Path:
    """Export the model as a traced TorchScript module."""
    try:
        with torch.inference_mode():
            traced = torch.jit.trace(model, dummy_input, strict=False)
            traced.save(str(output_path))
    except RuntimeError as exc:
        msg = str(exc)
        if "Expected x.is_cuda() to be true" in msg:
            raise RuntimeError(
                "TorchScript export must run on CUDA for this Mamba implementation "
                "because causal_conv1d is a CUDA custom op. Re-run with --device cuda."
            ) from exc
        raise
    return output_path


def export_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    opset: int,
) -> Path:
    """Export the model to ONNX and validate the generated file."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires the 'onnx' package in the nnMamba conda environment."
        ) from exc

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["ct"],
            output_names=["angle"],
            dynamic_axes={"ct": {0: "batch"}, "angle": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    return output_path


def verify_onnxruntime(output_path: Path, dummy_input: torch.Tensor) -> tuple[tuple[int, ...], str]:
    """Load the exported ONNX file with onnxruntime and run a smoke inference."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX runtime verification requires the 'onnxruntime' package."
        ) from exc

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: dummy_input.detach().cpu().numpy()})[0]
    return tuple(output.shape), str(output.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the nnMamba regression model for Netron or similar graph viewers."
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--uuid",
        default=None,
        help="Optional run UUID. If provided, export the trained fold checkpoint.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold id when exporting a trained checkpoint.",
    )
    parser.add_argument(
        "--format",
        choices=["torchscript", "onnx", "both"],
        default="torchscript",
        help="Export format. TorchScript is the safest local default for Netron.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy input batch size used during export.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Export device. 'auto' prefers CUDA because Mamba tracing is CUDA-only here.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version used when --format includes onnx.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <graphs>/<task>/exports.",
    )
    parser.add_argument(
        "--skip-verify-onnxruntime",
        action="store_true",
        help="Skip the CPU onnxruntime smoke check after ONNX export.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    config = Config.from_yaml(config_path)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested, but CUDA is not available.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, checkpoint_path = _load_model(
        config=config,
        config_dir=config_dir,
        device=device,
        run_uuid=args.uuid,
        fold=args.fold,
    )
    dummy_input = _build_dummy_input(config, args.batch_size, device=device)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else _resolve_path(config.paths.graphs, config_dir) / config.task / "exports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_name = args.uuid or "init"
    stem = f"{config.model.name}_fold{args.fold}_{source_name}"

    print(f"Config:        {config_path}")
    print(f"Model:         {config.model.name}")
    print(f"Device:        {device}")
    print(f"Input shape:   {tuple(dummy_input.shape)}")
    print(f"Checkpoint:    {checkpoint_path if checkpoint_path else 'random init'}")
    print(f"Output dir:    {output_dir}")

    exported: list[Path] = []
    if args.format in {"torchscript", "both"}:
        ts_path = output_dir / f"{stem}.pt"
        exported.append(export_torchscript(model, dummy_input, ts_path))

    if args.format in {"onnx", "both"}:
        onnx_path = output_dir / f"{stem}.onnx"
        exported.append(export_onnx(model, dummy_input, onnx_path, args.opset))
        if not args.skip_verify_onnxruntime:
            output_shape, output_dtype = verify_onnxruntime(onnx_path, dummy_input)
            print(
                f"ONNXRuntime:  ok | output_shape={output_shape} | output_dtype={output_dtype}"
            )

    print("\nExported files:")
    for path in exported:
        print(f"  - {path}")

    print("\nOpen with Netron:")
    for path in exported:
        print(f"  netron {path}")


if __name__ == "__main__":
    main()
