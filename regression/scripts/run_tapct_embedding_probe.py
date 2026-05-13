"""Run TAP-CT embedding extraction and probe training from one YAML config."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
REGRESSION_ROOT = REPO_ROOT / "regression"
EXTRACT_SCRIPT = REGRESSION_ROOT / "scripts" / "extract_tapct_embeddings.py"
PROBE_SCRIPT = REGRESSION_ROOT / "scripts" / "train_embedding_probe.py"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run TAP-CT frozen embedding workflow from YAML."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REGRESSION_ROOT / "config.tapct_embedding_probe.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract embeddings, ignoring run.train_probe.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Only train probes, ignoring run.extract_embeddings.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force embedding re-extraction even if cached case files exist.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def as_bool(value: Any, default: bool = False) -> bool:
    """Convert config values to bool."""
    if value is None:
        return default
    return bool(value)


def resolve_path(value: Any, *, default: Path | None = None) -> Path | None:
    """Resolve config paths relative to the repository root."""
    if value is None:
        return default
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def add_optional_arg(cmd: list[str], flag: str, value: Any) -> None:
    """Append a CLI argument if the config value is not null."""
    if value is None:
        return
    cmd.extend([flag, str(value)])


def build_extract_command(config: dict[str, Any], *, force: bool) -> list[str]:
    """Build the embedding extraction command."""
    tapct = config.get("tapct", {}) or {}
    data = config.get("data", {}) or {}
    embedding = config.get("embedding", {}) or {}

    output_dir = resolve_path(
        embedding.get("output_dir"),
        default=REGRESSION_ROOT / "embeddings" / "tapct_b_3d",
    )
    cmd = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--model-id",
        str(tapct.get("model_id", "fomofo/tap-ct-b-3d")),
        "--source-root",
        str(resolve_path(data.get("source_root"), default=REPO_ROOT / "by_angle_all")),
        "--labels-json",
        str(
            resolve_path(
                data.get("labels_json"),
                default=REPO_ROOT / "patient_angle_classification_by_group.json",
            )
        ),
        "--pft-json",
        str(resolve_path(data.get("pft_json"), default=REPO_ROOT / "pft.json")),
        "--output-dir",
        str(output_dir),
        "--device",
        str(tapct.get("device", "cuda")),
        "--dtype",
        str(tapct.get("dtype", "float32")),
        "--depth-window",
        str(embedding.get("depth_window", 12)),
        "--depth-stride",
        str(embedding.get("depth_stride", 6)),
        "--sw-batch-size",
        str(embedding.get("sw_batch_size", 1)),
        "--pooling",
        str(embedding.get("pooling", "mean_std_max")),
    ]
    add_optional_arg(cmd, "--max-cases", embedding.get("max_cases"))
    for patient_id in embedding.get("patient_ids", []) or []:
        cmd.extend(["--patient-id", str(patient_id)])
    if force or as_bool(embedding.get("force"), default=False):
        cmd.append("--force")
    if as_bool(embedding.get("save_window_embeddings"), default=False):
        cmd.append("--save-window-embeddings")
    return cmd


def build_probe_command(config: dict[str, Any]) -> list[str]:
    """Build the probe training command."""
    embedding = config.get("embedding", {}) or {}
    probe = config.get("probe", {}) or {}
    embedding_dir = resolve_path(
        embedding.get("output_dir"),
        default=REGRESSION_ROOT / "embeddings" / "tapct_b_3d",
    )
    features = resolve_path(
        probe.get("features"),
        default=(embedding_dir / "features.npz" if embedding_dir is not None else None),
    )
    output_dir = resolve_path(
        probe.get("output_dir"),
        default=REGRESSION_ROOT / "figures" / "TAPCT_embedding_probes" / "tapct_b_3d_yaml",
    )
    cmd = [
        sys.executable,
        str(PROBE_SCRIPT),
        "--features",
        str(features),
        "--output-dir",
        str(output_dir),
        "--target",
        str(probe.get("target", "all")),
        "--model",
        str(probe.get("model", "all")),
        "--n-splits",
        str(probe.get("n_splits", 5)),
        "--seed",
        str(probe.get("seed", 42)),
        "--ridge-alpha",
        str(probe.get("ridge_alpha", 1.0)),
        "--plot-dpi",
        str(probe.get("plot_dpi", 180)),
    ]
    add_optional_arg(cmd, "--metadata", resolve_path(probe.get("metadata")))
    if not as_bool(probe.get("plots"), default=True):
        cmd.append("--no-plots")
    return cmd


def print_command(cmd: list[str]) -> None:
    """Print a command in copyable shell form."""
    print(" ".join(shlex.quote(part) for part in cmd), flush=True)


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    """Run or print one workflow command."""
    print_command(cmd)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    """Run the configured TAP-CT embedding workflow."""
    args = parse_args()
    config = load_yaml(args.config)
    run_cfg = config.get("run", {}) or {}

    extract_enabled = as_bool(run_cfg.get("extract_embeddings"), default=True)
    probe_enabled = as_bool(run_cfg.get("train_probe"), default=True)
    if args.extract_only:
        extract_enabled = True
        probe_enabled = False
    if args.probe_only:
        extract_enabled = False
        probe_enabled = True

    experiment_name = (config.get("experiment", {}) or {}).get("name", args.config.stem)
    print(f"TAP-CT workflow: {experiment_name}", flush=True)

    if extract_enabled:
        print("\n[1/2] Extract embeddings", flush=True)
        run_command(build_extract_command(config, force=args.force), dry_run=args.dry_run)
    else:
        print("\n[1/2] Extract embeddings: skipped", flush=True)

    if probe_enabled:
        print("\n[2/2] Train embedding probes", flush=True)
        run_command(build_probe_command(config), dry_run=args.dry_run)
    else:
        print("\n[2/2] Train embedding probes: skipped", flush=True)


if __name__ == "__main__":
    main()
