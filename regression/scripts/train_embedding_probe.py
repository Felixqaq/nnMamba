"""Train small classifiers on frozen TAP-CT embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight


REPO_ROOT = Path(__file__).resolve().parents[2]
REGRESSION_ROOT = REPO_ROOT / "regression"
DEFAULT_FEATURES = REGRESSION_ROOT / "embeddings" / "tapct_s_3d" / "features.npz"
DEFAULT_OUTPUT_ROOT = REGRESSION_ROOT / "figures" / "TAPCT_embedding_probes"

ANGLE_3CLASS_NAMES = [
    "Emphysema/Abnormal (<=131 deg)",
    "Intermediate (132-151 deg)",
    "Normal (>=152 deg)",
]
ANGLE_BINARY_EXTREME_NAMES = [
    "Abnormal/emphysema-like (AC <=131 deg)",
    "Normal-like (AC >=152 deg)",
]
GOLD_STAGE_NAMES = [
    "Class 0 (No COPD)",
    "GOLD 1 (Mild)",
    "GOLD 2 (Moderate)",
    "GOLD 3 (Severe)",
    "GOLD 4 (Very Severe)",
]

TARGET_MODEL_SETS = {
    "angle_3class": [
        "logistic",
        "linear_svm",
        "ridge_classifier",
        "ordinal_logistic",
        "angle_ridge_threshold",
    ],
    "angle_binary_extreme": [
        "logistic",
        "linear_svm",
        "ridge_classifier",
    ],
    "gold": [
        "logistic",
        "linear_svm",
        "ridge_classifier",
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run sklearn probes on frozen TAP-CT embeddings."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--target",
        choices=("all", "angle_3class", "angle_binary_extreme", "gold"),
        default="all",
    )
    parser.add_argument(
        "--model",
        choices=(
            "all",
            "logistic",
            "linear_svm",
            "ridge_classifier",
            "ordinal_logistic",
            "angle_ridge_threshold",
        ),
        default="all",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip automatic PNG figure generation after writing results.json.",
    )
    parser.add_argument("--plot-dpi", type=int, default=180)
    return parser.parse_args()


def timestamp() -> str:
    """Return a filesystem-safe timestamp."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def load_feature_bundle(features_path: Path, metadata_path: Path | None):
    """Load features and metadata produced by extract_tapct_embeddings.py."""
    if metadata_path is None:
        metadata_path = features_path.with_name("metadata.csv")
    with np.load(features_path) as data:
        features = data["features"].astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != len(features):
        raise ValueError(
            f"Metadata rows ({len(metadata)}) do not match features ({len(features)})."
        )
    return features, metadata


def angle_to_3class(angle: float) -> int:
    """Map a continuous angle prediction to the three fixed classes."""
    if angle <= 131.0:
        return 0
    if angle < 152.0:
        return 1
    return 2


def estimator_for(model_name: str, *, ridge_alpha: float) -> Pipeline:
    """Build a small sklearn classifier for frozen embeddings."""
    if model_name == "logistic":
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
        )
    elif model_name == "linear_svm":
        estimator = LinearSVC(
            class_weight="balanced",
            dual="auto",
            max_iter=20000,
        )
    elif model_name == "ridge_classifier":
        estimator = RidgeClassifier(
            alpha=ridge_alpha,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported estimator: {model_name}")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def predict_with_estimator(
    model_name: str,
    *,
    ridge_alpha: float,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Return a fit/predict function for standard classification models."""

    def fit_predict(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        _angle_train: np.ndarray,
    ) -> np.ndarray:
        model = clone(estimator_for(model_name, ridge_alpha=ridge_alpha))
        model.fit(x_train, y_train)
        return model.predict(x_test).astype(int)

    return fit_predict


def predict_with_angle_ridge(
    *,
    ridge_alpha: float,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Return a fit/predict function that regresses angle then thresholds it."""

    def fit_predict(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        angle_train: np.ndarray,
    ) -> np.ndarray:
        sample_weight = compute_sample_weight("balanced", y_train)
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=ridge_alpha)),
            ]
        )
        model.fit(x_train, angle_train, model__sample_weight=sample_weight)
        pred_angles = model.predict(x_test)
        return np.array([angle_to_3class(float(angle)) for angle in pred_angles])

    return fit_predict


def predict_with_ordinal_logistic(
    *,
    ridge_alpha: float,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Return a fit/predict function for two-threshold ordinal classification."""

    def fit_predict(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        _angle_train: np.ndarray,
    ) -> np.ndarray:
        gt_low = (y_train > 0).astype(int)
        ge_high = (y_train == 2).astype(int)

        low_model = estimator_for("logistic", ridge_alpha=1.0)
        high_model = estimator_for("logistic", ridge_alpha=1.0)
        low_model.fit(x_train, gt_low)
        high_model.fit(x_train, ge_high)

        low_prob = low_model.predict_proba(x_test)[:, 1]
        high_prob = high_model.predict_proba(x_test)[:, 1]
        preds = []
        for p_low, p_high in zip(low_prob, high_prob, strict=True):
            if p_low < 0.5:
                preds.append(0)
            elif p_high >= 0.5:
                preds.append(2)
            else:
                preds.append(1)
        return np.array(preds, dtype=int)

    return fit_predict


def predictor_for(
    model_name: str,
    *,
    ridge_alpha: float,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Build the fold-level fit/predict function for a probe model."""
    if model_name in {"logistic", "linear_svm", "ridge_classifier"}:
        return predict_with_estimator(model_name, ridge_alpha=ridge_alpha)
    if model_name == "angle_ridge_threshold":
        return predict_with_angle_ridge(ridge_alpha=ridge_alpha)
    if model_name == "ordinal_logistic":
        return predict_with_ordinal_logistic(ridge_alpha=ridge_alpha)
    raise ValueError(f"Unsupported model: {model_name}")


def metric_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[int],
) -> dict:
    """Compute classification metrics for one fold or combined predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    sensitivity, specificity = classification_sensitivity_specificity(cm)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "macro_f1": round(float(f1), 5),
        "macro_precision": round(float(precision), 5),
        "macro_recall": round(float(recall), 5),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, y_pred)),
            5,
        ),
        "sensitivity": round(float(sensitivity), 5),
        "specificity": round(float(specificity), 5),
        "confusion_matrix": cm.tolist(),
    }


def classification_sensitivity_specificity(cm: np.ndarray) -> tuple[float, float]:
    """Return binary class-0 or multiclass macro sensitivity/specificity."""
    if cm.size == 0:
        return 0.0, 0.0

    cm = np.asarray(cm, dtype=float)
    total = float(cm.sum())
    true_positives = np.diag(cm)
    false_negatives = cm.sum(axis=1) - true_positives
    false_positives = cm.sum(axis=0) - true_positives
    true_negatives = total - true_positives - false_negatives - false_positives

    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivities = np.divide(
            true_positives,
            true_positives + false_negatives,
            out=np.zeros_like(true_positives, dtype=float),
            where=(true_positives + false_negatives) != 0,
        )
        specificities = np.divide(
            true_negatives,
            true_negatives + false_positives,
            out=np.zeros_like(true_negatives, dtype=float),
            where=(true_negatives + false_positives) != 0,
        )

    if cm.shape == (2, 2):
        return float(sensitivities[0]), float(specificities[0])
    return float(sensitivities.mean()), float(specificities.mean())


def summarize_folds(folds: list[dict]) -> dict:
    """Aggregate fold metrics into mean/std summary."""
    keys = [
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
    ]
    output = {}
    for key in keys:
        values = np.array([fold[key] for fold in folds], dtype=float)
        output[f"mean_{key}"] = round(float(values.mean()), 5)
        output[f"std_{key}"] = round(float(values.std()), 5)
    return output


def prepare_target(
    target: str,
    features: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Filter and return arrays for the requested target."""
    if target == "angle_3class":
        mask = np.ones(len(metadata), dtype=bool)
        y = metadata["angle_3class"].to_numpy(dtype=int)
        class_names = ANGLE_3CLASS_NAMES
    elif target == "angle_binary_extreme":
        y_all = metadata["angle_binary_extreme"].to_numpy(dtype=int)
        mask = y_all >= 0
        y = y_all[mask]
        class_names = ANGLE_BINARY_EXTREME_NAMES
    elif target == "gold":
        mask = metadata["gold_stage"].notna().to_numpy(dtype=bool)
        y = metadata.loc[mask, "gold_stage"].to_numpy(dtype=int)
        class_names = GOLD_STAGE_NAMES
    else:
        raise ValueError(f"Unsupported target: {target}")

    x = features[mask]
    angles = metadata.loc[mask, "angle"].to_numpy(dtype=float)
    meta = metadata.loc[mask].reset_index(drop=True)
    return x, y, angles, meta, class_names


def run_probe(
    *,
    target: str,
    model_name: str,
    features: np.ndarray,
    metadata: pd.DataFrame,
    n_splits: int,
    seed: int,
    ridge_alpha: float,
) -> tuple[dict, list[dict]]:
    """Run one target/model probe with stratified k-fold CV."""
    x, y, angles, meta, class_names = prepare_target(target, features, metadata)
    labels = list(range(len(class_names)))
    min_class_count = int(np.bincount(y, minlength=len(labels)).min())
    split_count = min(int(n_splits), min_class_count)
    if split_count < 2:
        raise ValueError(
            f"Need at least two samples per class for CV, got counts "
            f"{np.bincount(y, minlength=len(labels)).tolist()}"
        )

    splitter = StratifiedKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=seed,
    )
    predictor = predictor_for(model_name, ridge_alpha=ridge_alpha)

    fold_results: list[dict] = []
    prediction_rows: list[dict] = []
    all_true: list[int] = []
    all_pred: list[int] = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(x, y), start=1):
        y_pred = predictor(
            x[train_idx],
            y[train_idx],
            x[test_idx],
            angles[train_idx],
        )
        y_true = y[test_idx]
        fold_metrics = metric_summary(y_true, y_pred, labels=labels)
        fold_metrics["fold"] = fold_index
        fold_results.append(fold_metrics)

        for local_index, true_value, pred_value in zip(
            test_idx,
            y_true,
            y_pred,
            strict=True,
        ):
            record = meta.iloc[int(local_index)]
            prediction_rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "fold": fold_index,
                    "patient_id": record["patient_id"],
                    "path": record["path"],
                    "angle": float(record["angle"]),
                    "true_index": int(true_value),
                    "true_label": class_names[int(true_value)],
                    "pred_index": int(pred_value),
                    "pred_label": class_names[int(pred_value)],
                    "correct": bool(int(true_value) == int(pred_value)),
                }
            )
        all_true.extend(int(value) for value in y_true)
        all_pred.extend(int(value) for value in y_pred)

    combined = metric_summary(
        np.array(all_true, dtype=int),
        np.array(all_pred, dtype=int),
        labels=labels,
    )
    result = {
        "target": target,
        "model": model_name,
        "class_names": class_names,
        "num_samples": int(len(y)),
        "class_counts": {
            class_names[index]: int(count)
            for index, count in enumerate(np.bincount(y, minlength=len(labels)))
        },
        "folds": fold_results,
        "summary": summarize_folds(fold_results),
        "combined": combined,
    }
    return result, prediction_rows


def requested_targets(target: str) -> list[str]:
    """Expand target selector."""
    if target == "all":
        return ["angle_3class", "angle_binary_extreme"]
    return [target]


def requested_models(target: str, model: str) -> list[str]:
    """Expand model selector for a target."""
    if model == "all":
        return list(TARGET_MODEL_SETS[target])
    if model not in TARGET_MODEL_SETS[target]:
        raise ValueError(f"Model {model!r} is not configured for target {target!r}.")
    return [model]


def write_predictions(path: Path, rows: list[dict]) -> None:
    """Write all fold predictions to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_figures(results_json: Path, *, dpi: int) -> list[Path]:
    """Generate PNG report figures for a completed probe run."""
    try:
        from plot_embedding_probe_results import plot_results_file
    except ImportError as exc:
        print(f"Skipping plot generation: {exc}")
        return []
    return plot_results_file(results_json, dpi=dpi)


def main() -> None:
    """Run all requested frozen-embedding probes."""
    args = parse_args()
    features, metadata = load_feature_bundle(args.features, args.metadata)
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / timestamp())
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    predictions = []
    for target in requested_targets(args.target):
        for model_name in requested_models(target, args.model):
            print(f"Running probe: target={target}, model={model_name}")
            result, rows = run_probe(
                target=target,
                model_name=model_name,
                features=features,
                metadata=metadata,
                n_splits=args.n_splits,
                seed=args.seed,
                ridge_alpha=args.ridge_alpha,
            )
            results.append(result)
            predictions.extend(rows)

    run_summary = {
        "features": str(args.features),
        "metadata": str(args.metadata or args.features.with_name("metadata.csv")),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "ridge_alpha": float(args.ridge_alpha),
        "results": results,
    }
    results_json = output_dir / "results.json"
    with results_json.open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)
    write_predictions(output_dir / "predictions.csv", predictions)

    print(f"Saved probe results to {output_dir}")
    for result in results:
        summary = result["summary"]
        print(
            f"{result['target']} / {result['model']}: "
            f"Acc={summary['mean_accuracy']:.4f}, "
            f"Macro-F1={summary['mean_macro_f1']:.4f}, "
            f"Bal Acc={summary['mean_balanced_accuracy']:.4f}"
        )

    if not args.no_plots:
        written = generate_figures(results_json, dpi=args.plot_dpi)
        if written:
            print(f"Wrote {len(written)} figures to {output_dir}")


if __name__ == "__main__":
    main()
