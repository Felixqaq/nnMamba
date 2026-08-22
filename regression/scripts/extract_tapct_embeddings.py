"""Extract frozen TAP-CT embeddings for angle-derived classification probes.

The script keeps TAP-CT as a frozen feature extractor. Each CT volume is
preprocessed by the model's Hugging Face image processor, split into axial
windows, embedded window-by-window, and pooled into one patient-level vector.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


REPO_ROOT = Path(__file__).resolve().parents[2]
REGRESSION_ROOT = REPO_ROOT / "regression"


def load_manifest_helpers():
    """Load manifest.py directly without importing data/__init__.py."""
    manifest_path = REGRESSION_ROOT / "data" / "manifest.py"
    spec = importlib.util.spec_from_file_location(
        "nnmamba_regression_manifest",
        manifest_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load manifest helper from {manifest_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MANIFEST = load_manifest_helpers()
angle_binary_extreme_label = _MANIFEST.angle_binary_extreme_label
build_angle_manifest = _MANIFEST.build_angle_manifest


DEFAULT_MODEL_ID = "fomofo/tap-ct-s-3d"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "by_angle_all"
DEFAULT_LABELS_JSON = REPO_ROOT / "patient_angle_classification_by_group.json"
DEFAULT_PFT_JSON = REPO_ROOT / "pft.json"
DEFAULT_OUTPUT_DIR = REGRESSION_ROOT / "embeddings" / "tapct_s_3d"


@dataclass(frozen=True)
class EmbeddingRecord:
    """Metadata for one extracted patient embedding."""

    patient_id: str
    path: str
    angle: float
    angle_3class: int
    angle_3class_label: str
    angle_binary_extreme: int
    angle_binary_extreme_label: str
    source_group: str
    gold_stage: int | None
    gold_stage_label: str | None
    post_fev1_percent_predicted: float | None
    embedding_path: str
    num_windows: int


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract frozen TAP-CT patient-level embeddings."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--labels-json", type=Path, default=DEFAULT_LABELS_JSON)
    parser.add_argument("--pft-json", type=Path, default=DEFAULT_PFT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--resize-dim",
        type=int,
        default=0,
        help="override the processor's in-plane resize (0 = keep its 224 default)",
    )
    parser.add_argument(
        "--target-mode",
        default="angle_3class",
        help="manifest target mode used only to enumerate cases; use "
        "normal_v_abnormal for cohorts whose label is the parent folder name",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument("--depth-window", type=int, default=12)
    parser.add_argument("--depth-stride", type=int, default=6)
    parser.add_argument("--sw-batch-size", type=int, default=4)
    parser.add_argument(
        "--pooling",
        choices=("mean", "mean_std", "mean_std_max"),
        default="mean_std_max",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--patient-id", action="append", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-window-embeddings", action="store_true")
    return parser.parse_args()


def sanitize_model_id(model_id: str) -> str:
    """Make a filesystem-safe model identifier."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")


def resolve_device(device: str) -> torch.device:
    """Resolve requested device with a CPU fallback."""
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def load_records(args: argparse.Namespace) -> list:
    """Load records from the existing manifest builder for the chosen target mode.

    The extractor itself is label-agnostic — it only needs the list of CT paths —
    but the manifest builder drops any patient the target mode cannot label. Under
    "angle_3class" that silently excludes every case without an angle annotation,
    so cohorts labelled by folder name need "normal_v_abnormal" instead.
    """
    manifest = build_angle_manifest(
        data_root=args.source_root,
        labels_json=args.labels_json,
        pft_json=args.pft_json if args.pft_json.exists() else None,
        target_mode=args.target_mode,
    )
    records = list(manifest.records)
    if args.patient_id:
        wanted = {str(patient_id) for patient_id in args.patient_id}
        records = [record for record in records if record.patient_id in wanted]
    if args.max_cases is not None:
        records = records[: max(0, int(args.max_cases))]
    return records


def load_ct_volume(path: Path) -> np.ndarray:
    """Read a NIfTI CT volume as a (1, 1, D, H, W) numpy array in LPS space."""
    volume = sitk.ReadImage(str(path))
    volume = sitk.DICOMOrient(volume, "LPS")
    array = sitk.GetArrayFromImage(volume).astype(np.float32)
    return np.expand_dims(array, axis=(0, 1))


def preprocess_volume(preprocessor, path: Path) -> torch.Tensor:
    """Apply the TAP-CT image processor and return a 5D tensor."""
    array = load_ct_volume(path)
    encoded = preprocessor(array)
    pixel_values = encoded["pixel_values"]
    tensor = torch.as_tensor(pixel_values, dtype=torch.float32)
    if tensor.ndim != 5:
        raise ValueError(
            f"Expected TAP-CT preprocessor to return 5D pixel_values, got {tensor.shape}"
        )
    return tensor


def depth_starts(depth: int, window: int, stride: int) -> list[int]:
    """Return starts that cover the full axial depth."""
    if window <= 0:
        raise ValueError("--depth-window must be positive.")
    if stride <= 0:
        raise ValueError("--depth-stride must be positive.")
    if depth <= window:
        return [0]
    starts = list(range(0, depth - window + 1, stride))
    last = depth - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def pad_to_window(x: torch.Tensor, window: int) -> torch.Tensor:
    """Pad shallow volumes along depth so one TAP-CT window can be extracted."""
    depth = int(x.shape[2])
    if depth >= window:
        return x
    return F.pad(x, (0, 0, 0, 0, 0, window - depth), mode="constant", value=0.0)


def output_to_embedding(output) -> torch.Tensor:
    """Extract the pooled embedding from a TAP-CT model output."""
    pooler = getattr(output, "pooler_output", None)
    if pooler is not None:
        return pooler
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden[:, 0]
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported TAP-CT output type: {type(output)!r}")


def embed_volume(
    x: torch.Tensor,
    model,
    device: torch.device,
    *,
    depth_window: int,
    depth_stride: int,
    sw_batch_size: int,
    dtype: torch.dtype,
) -> np.ndarray:
    """Embed all axial windows from one preprocessed CT volume."""
    x = pad_to_window(x, depth_window)
    starts = depth_starts(int(x.shape[2]), depth_window, depth_stride)
    chunks: list[torch.Tensor] = []

    for start in range(0, len(starts), sw_batch_size):
        batch_starts = starts[start : start + sw_batch_size]
        windows = torch.cat(
            [x[:, :, z : z + depth_window, :, :] for z in batch_starts],
            dim=0,
        )
        windows = windows.to(device=device, dtype=dtype, non_blocking=True)
        with torch.inference_mode():
            output = model(windows)
            embedding = output_to_embedding(output).detach().float().cpu()
        chunks.append(embedding)

    return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def pool_window_embeddings(window_embeddings: np.ndarray, mode: str) -> np.ndarray:
    """Aggregate window-level embeddings into one patient-level vector."""
    window_embeddings = np.nan_to_num(
        window_embeddings,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    mean = window_embeddings.mean(axis=0)
    if mode == "mean":
        return np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    std = window_embeddings.std(axis=0)
    if mode == "mean_std":
        return np.nan_to_num(
            np.concatenate([mean, std]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)
    max_values = window_embeddings.max(axis=0)
    return np.nan_to_num(
        np.concatenate([mean, std, max_values]),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def case_output_path(output_dir: Path, patient_id: str) -> Path:
    """Return the per-patient embedding path."""
    return output_dir / "cases" / f"{patient_id}.npz"


def save_case_embedding(
    path: Path,
    embedding: np.ndarray,
    window_embeddings: np.ndarray,
    *,
    save_window_embeddings: bool,
) -> None:
    """Save one patient embedding for resume-friendly extraction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"embedding": embedding.astype(np.float32)}
    if save_window_embeddings:
        payload["window_embeddings"] = window_embeddings.astype(np.float32)
    np.savez_compressed(path, **payload)


def load_case_embedding(path: Path) -> np.ndarray:
    """Load a previously extracted patient embedding."""
    with np.load(path) as data:
        return data["embedding"].astype(np.float32)


def write_metadata(path: Path, rows: list[EmbeddingRecord]) -> None:
    """Write metadata CSV alongside feature arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_feature_bundle(
    output_dir: Path,
    rows: list[EmbeddingRecord],
    features: list[np.ndarray],
    *,
    model_id: str,
    args: argparse.Namespace,
) -> None:
    """Write features.npz, metadata.csv, and extraction_config.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_matrix = np.stack(features).astype(np.float32)
    np.savez_compressed(
        output_dir / "features.npz",
        features=feature_matrix,
        patient_ids=np.array([row.patient_id for row in rows]),
        angles=np.array([row.angle for row in rows], dtype=np.float32),
        angle_3class=np.array([row.angle_3class for row in rows], dtype=np.int64),
        angle_binary_extreme=np.array(
            [row.angle_binary_extreme for row in rows],
            dtype=np.int64,
        ),
    )
    write_metadata(output_dir / "metadata.csv", rows)

    config = {
        "model_id": model_id,
        "source_root": str(args.source_root),
        "labels_json": str(args.labels_json),
        "target_mode": args.target_mode,
        "pft_json": str(args.pft_json),
        "depth_window": args.depth_window,
        "depth_stride": args.depth_stride,
        "sw_batch_size": args.sw_batch_size,
        "pooling": args.pooling,
        "dtype": args.dtype,
        "num_cases": len(rows),
        "feature_dim": int(feature_matrix.shape[1]),
    }
    with (output_dir / "extraction_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def main() -> None:
    """Run TAP-CT embedding extraction."""
    args = parse_args()
    output_dir = args.output_dir
    if output_dir == DEFAULT_OUTPUT_DIR and args.model_id != DEFAULT_MODEL_ID:
        output_dir = REGRESSION_ROOT / "embeddings" / sanitize_model_id(args.model_id)

    device = resolve_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" and device.type == "cuda" else torch.float32
    records = load_records(args)
    if not records:
        raise SystemExit("No records found for embedding extraction.")

    print(f"Loading TAP-CT model: {args.model_id}")
    preprocessor = AutoImageProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    if args.resize_dim:
        # The encoder interpolates its positional embeddings, so it accepts grids
        # other than the 224x224 it was pretrained on. Raising this keeps more of
        # the native 0.6 mm detail (emphysema is a 2-10 mm texture finding) at the
        # cost of quadratic attention, and risks a scale mismatch: an 8x8 patch
        # covers 11 mm at 224 but only 4.8 mm at 512, so every learned spatial
        # relationship shifts. Whether that helps is empirical.
        preprocessor.resize_dims = (args.resize_dim, args.resize_dim)
        print(f"resize_dims overridden to {preprocessor.resize_dims}")
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
    model.eval().to(device=device, dtype=dtype)

    rows: list[EmbeddingRecord] = []
    features: list[np.ndarray] = []

    for record in tqdm(records, desc="Extracting TAP-CT embeddings"):
        embedding_path = case_output_path(output_dir, record.patient_id)
        if embedding_path.exists() and not args.force:
            embedding = load_case_embedding(embedding_path)
            num_windows = -1
        else:
            x = preprocess_volume(preprocessor, Path(record.path))
            window_embeddings = embed_volume(
                x,
                model,
                device,
                depth_window=args.depth_window,
                depth_stride=args.depth_stride,
                sw_batch_size=args.sw_batch_size,
                dtype=dtype,
            )
            embedding = pool_window_embeddings(window_embeddings, args.pooling)
            num_windows = int(window_embeddings.shape[0])
            save_case_embedding(
                embedding_path,
                embedding,
                window_embeddings,
                save_window_embeddings=args.save_window_embeddings,
            )

        extreme = angle_binary_extreme_label(record.angle)
        if extreme is None:
            extreme_index = -1
            extreme_label = "Gray zone (132-151 deg)"
        else:
            extreme_index, extreme_label = extreme

        rows.append(
            EmbeddingRecord(
                patient_id=record.patient_id,
                path=record.path,
                angle=float(record.angle),
                angle_3class=int(record.class_index),
                angle_3class_label=str(record.class_label),
                angle_binary_extreme=int(extreme_index),
                angle_binary_extreme_label=str(extreme_label),
                source_group=record.source_group,
                gold_stage=record.gold_stage,
                gold_stage_label=record.gold_stage_label,
                post_fev1_percent_predicted=record.post_fev1_percent_predicted,
                embedding_path=str(embedding_path),
                num_windows=num_windows,
            )
        )
        features.append(embedding)

    write_feature_bundle(output_dir, rows, features, model_id=args.model_id, args=args)
    print(f"Saved TAP-CT embeddings to {output_dir}")
    print(f"Feature matrix: {len(rows)} x {features[0].shape[0]}")


if __name__ == "__main__":
    main()
