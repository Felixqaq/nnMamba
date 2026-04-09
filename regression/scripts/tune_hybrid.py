#!/usr/bin/env python3
"""Run a lightweight time-budgeted hyperparameter sweep for the hybrid regressor."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "train.py"
DEFAULT_CONFIG = ROOT / "config.hybrid.yaml"


@dataclass(frozen=True)
class Candidate:
    """One concrete tuning candidate."""

    name: str
    overrides: dict[str, Any]


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested dictionary value using dot-notation."""
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        child = target.get(key)
        if not isinstance(child, dict):
            child = {}
            target[key] = child
        target = child
    target[keys[-1]] = value


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied config with overrides applied."""
    updated = copy.deepcopy(config)
    for key, value in overrides.items():
        _set_nested(updated, key, value)
    return updated


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def build_hourly_candidates(base_config: dict[str, Any]) -> list[Candidate]:
    """Build a compact, mostly safe candidate queue for about one hour."""
    model_cfg = base_config.setdefault("model", {})
    train_cfg = base_config.setdefault("training", {})

    base_hidden_dim = int(model_cfg.get("hidden_dim", 192))
    base_dropout = float(model_cfg.get("dropout", 0.3))
    base_attn_dropout = float(model_cfg.get("attn_dropout", 0.1))
    base_attn_layers = int(model_cfg.get("attn_layers", 1))
    base_attn_heads = int(model_cfg.get("attn_heads", 8))
    base_blocks = int(model_cfg.get("blocks", 3))
    base_lr = float(train_cfg.get("learning_rate", 1e-4))
    base_weight_decay = float(train_cfg.get("weight_decay", 1e-3))

    return [
        Candidate("baseline", {}),
        Candidate("attn_drop_0p00", {"model.attn_dropout": 0.0}),
        Candidate("attn_drop_0p05", {"model.attn_dropout": 0.05}),
        Candidate("attn_layers_2", {"model.attn_layers": max(2, base_attn_layers + 1)}),
        Candidate(
            "attn_layers_2_drop_0p05",
            {
                "model.attn_layers": max(2, base_attn_layers + 1),
                "model.attn_dropout": 0.05,
            },
        ),
        Candidate("hidden_160", {"model.hidden_dim": max(160, base_hidden_dim - 32)}),
        Candidate("hidden_224", {"model.hidden_dim": min(224, max(224, base_hidden_dim))}),
        Candidate("lr_8e5", {"training.learning_rate": 8e-5}),
        Candidate("lr_1p2e4", {"training.learning_rate": 1.2e-4}),
        Candidate(
            "lighter_reg",
            {
                "training.weight_decay": 5e-4,
                "model.dropout": max(0.15, base_dropout - 0.1),
                "model.attn_dropout": min(base_attn_dropout, 0.05),
            },
        ),
        Candidate(
            "stronger_reg",
            {
                "training.weight_decay": 2e-3,
                "model.dropout": min(0.4, base_dropout + 0.05),
                "model.attn_dropout": max(0.1, base_attn_dropout),
            },
        ),
        Candidate("blocks_2", {"model.blocks": max(2, base_blocks - 1)}),
        Candidate(
            "best_guess",
            {
                "model.attn_layers": max(2, base_attn_layers + 1),
                "model.attn_dropout": 0.05,
                "model.hidden_dim": min(224, max(192, base_hidden_dim)),
                "training.learning_rate": min(1.2e-4, max(base_lr, 9e-5)),
                "training.weight_decay": base_weight_decay,
            },
        ),
    ]


def _extract_run_uuid(stdout: str) -> str | None:
    """Parse the run UUID from train.py output."""
    marker = "Training complete. Run UUID:"
    for line in reversed(stdout.splitlines()):
        if marker in line:
            return line.split(marker, maxsplit=1)[-1].strip()
    return None


def _load_results(results_path: Path) -> dict[str, Any]:
    with open(results_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_config_path(base_dir: Path, path_value: str | Path) -> Path:
    """Resolve a config path relative to the regression root."""
    path = Path(path_value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _candidate_score(result: dict[str, Any]) -> tuple[float, float, float]:
    """Lower is better for tuple sorting."""
    summary = result["summary"]
    mae = float(summary.get("mean_mae", float("inf")))
    rmse = float(summary.get("mean_rmse", float("inf")))
    r2 = -float(summary.get("mean_r2", float("-inf")))
    return (mae, rmse, r2)


def _write_leaderboard(entries: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    """Write CSV and JSON leaderboard artifacts."""
    json_path = output_dir / "leaderboard.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)

    csv_path = output_dir / "leaderboard.csv"
    fieldnames = [
        "rank",
        "candidate",
        "run_uuid",
        "status",
        "mean_mae",
        "mean_rmse",
        "mean_r2",
        "mean_pearson",
        "duration_seconds",
        "results_json",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, entry in enumerate(entries, start=1):
            row = {
                "rank": idx,
                "candidate": entry["candidate"],
                "run_uuid": entry.get("run_uuid"),
                "status": entry.get("status"),
                "mean_mae": entry.get("summary", {}).get("mean_mae"),
                "mean_rmse": entry.get("summary", {}).get("mean_rmse"),
                "mean_r2": entry.get("summary", {}).get("mean_r2"),
                "mean_pearson": entry.get("summary", {}).get("mean_pearson"),
                "duration_seconds": entry.get("duration_seconds"),
                "results_json": entry.get("results_json"),
            }
            writer.writerow(row)

    return csv_path, json_path


def run_candidate(
    *,
    candidate: Candidate,
    base_config: dict[str, Any],
    output_dir: Path,
    python_executable: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Run one tuning candidate and return a leaderboard entry."""
    candidate_dir = output_dir / "candidate_configs"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    config_payload = _apply_overrides(base_config, candidate.overrides)
    config_path = candidate_dir / f"{candidate.name}.yaml"
    _write_yaml(config_path, config_payload)

    entry: dict[str, Any] = {
        "candidate": candidate.name,
        "status": "pending",
        "config_path": str(config_path),
        "overrides": candidate.overrides,
    }

    if dry_run:
        entry["status"] = "planned"
        return entry

    cmd = [python_executable, str(TRAIN_PY), "--config", str(config_path)]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nnmamba_mpl"))
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"\n=== Running {candidate.name} ===", flush=True)
    print(f"Overrides: {candidate.overrides}", flush=True)
    started = perf_counter()
    log_path = output_dir / "logs" / f"{candidate.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_chunks: list[str] = []
    with open(log_path, "w", encoding="utf-8") as log_handle:
        assert process.stdout is not None
        for line in process.stdout:
            output_chunks.append(line)
            print(line, end="", flush=True)
            log_handle.write(line)
        process.wait()
    duration = perf_counter() - started
    combined_output = "".join(output_chunks)

    entry["duration_seconds"] = round(duration, 3)
    entry["returncode"] = process.returncode
    entry["stdout_log"] = str(log_path)
    entry["stderr_log"] = None

    if process.returncode != 0:
        entry["status"] = "failed"
        error_lines = [line.strip() for line in combined_output.splitlines() if line.strip()]
        entry["error"] = error_lines[-1] if error_lines else ""
        return entry

    run_uuid = _extract_run_uuid(combined_output)
    if not run_uuid:
        entry["status"] = "failed"
        entry["error"] = "Could not parse run UUID from train.py output."
        return entry

    figures_root = _resolve_config_path(
        ROOT,
        config_payload.get("paths", {}).get("figures", "./figures"),
    )
    results_path = figures_root / config_payload["task"] / run_uuid / "results.json"
    if not results_path.exists():
        entry["status"] = "failed"
        entry["run_uuid"] = run_uuid
        entry["error"] = f"Missing results.json at {results_path}"
        return entry

    results = _load_results(results_path)
    entry["status"] = "completed"
    entry["run_uuid"] = run_uuid
    entry["results_json"] = str(results_path)
    entry["summary"] = results.get("summary", {})
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time-budgeted hyperparameter tuning for hybrid Mamba-attention regression."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base YAML config to clone and override.",
    )
    parser.add_argument(
        "--budget-minutes",
        type=float,
        default=60.0,
        help="Stop launching new runs after this wall-clock budget.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=12,
        help="Maximum number of candidates to try.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for tuning artifacts. Defaults to figures/<task>/tuning_runs/<timestamp>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write candidate configs and planned order without launching training.",
    )
    args = parser.parse_args()

    base_config = _load_yaml(args.base_config)
    task_name = str(base_config.get("task", "PFT_angle_regression"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    figures_root = _resolve_config_path(
        ROOT,
        base_config.get("paths", {}).get("figures", "./figures"),
    )
    output_dir = args.out_dir or (figures_root / task_name / "tuning_runs" / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_hourly_candidates(copy.deepcopy(base_config))[: max(1, args.max_runs)]
    planned = {
        "base_config": str(args.base_config.resolve()),
        "budget_minutes": args.budget_minutes,
        "max_runs": args.max_runs,
        "dry_run": args.dry_run,
        "candidates": [
            {"candidate": candidate.name, "overrides": candidate.overrides}
            for candidate in candidates
        ],
    }
    with open(output_dir / "plan.json", "w", encoding="utf-8") as handle:
        json.dump(planned, handle, indent=2)

    started = perf_counter()
    results: list[dict[str, Any]] = []
    budget_seconds = max(0.0, float(args.budget_minutes) * 60.0)

    for index, candidate in enumerate(candidates, start=1):
        elapsed = perf_counter() - started
        if not args.dry_run and elapsed >= budget_seconds:
            print(
                f"Budget reached after {elapsed / 60.0:.1f} minutes. "
                f"Stopping before candidate {index}: {candidate.name}"
            )
            break

        print(f"[{index}/{len(candidates)}] {candidate.name}")
        result = run_candidate(
            candidate=candidate,
            base_config=base_config,
            output_dir=output_dir,
            python_executable=sys.executable,
            dry_run=args.dry_run,
        )
        results.append(result)

        if result.get("status") == "completed":
            summary = result.get("summary", {})
            print(
                "Candidate complete: "
                f"{candidate.name} | "
                f"MAE={summary.get('mean_mae')} | "
                f"RMSE={summary.get('mean_rmse')} | "
                f"R2={summary.get('mean_r2')}",
                flush=True,
            )
        elif result.get("status") == "failed":
            print(
                "Candidate failed and will be skipped: "
                f"{candidate.name} | {result.get('error', 'unknown error')}",
                flush=True,
            )

        partial_completed = [
            entry for entry in results if entry.get("status") == "completed"
        ]
        partial_failed = [
            entry for entry in results if entry.get("status") != "completed"
        ]
        partial_ranked = sorted(partial_completed, key=_candidate_score) + partial_failed
        _write_leaderboard(partial_ranked, output_dir)

    completed_entries = [entry for entry in results if entry.get("status") == "completed"]
    failed_entries = [entry for entry in results if entry.get("status") != "completed"]
    ranked_entries = sorted(completed_entries, key=_candidate_score) + failed_entries

    csv_path, json_path = _write_leaderboard(ranked_entries, output_dir)
    best_entry = ranked_entries[0] if ranked_entries and ranked_entries[0].get("status") == "completed" else None

    if best_entry is not None:
        best_config = _apply_overrides(base_config, best_entry["overrides"])
        best_config_path = output_dir / "best_config.yaml"
        _write_yaml(best_config_path, best_config)
        print("\nBest candidate:")
        print(
            f"  {best_entry['candidate']} | "
            f"MAE={best_entry['summary'].get('mean_mae')} | "
            f"RMSE={best_entry['summary'].get('mean_rmse')} | "
            f"R2={best_entry['summary'].get('mean_r2')}"
        )
        print(f"  Saved config: {best_config_path}")

    print(f"\nSaved leaderboard CSV to: {csv_path}")
    print(f"Saved leaderboard JSON to: {json_path}")


if __name__ == "__main__":
    main()
