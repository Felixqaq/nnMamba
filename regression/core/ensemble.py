"""Majority-vote ensemble orchestration for classification experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .checkpoints import generate_uuid, load_checkpoint
from .config import Config
from .evaluator import (
    ClassificationMetrics,
    compute_classification_metrics,
    evaluate,
    save_predictions,
)
from .trainer import Trainer
from data.loader import LoaderHelper
from models import build_model


CLASSIFICATION_SUMMARY_KEYS = (
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
)


def majority_vote_binary(
    member_preds: torch.Tensor,
    vote_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return binary hard-vote predictions and vote-fraction probabilities."""
    if member_preds.ndim != 2:
        raise ValueError("member_preds must have shape [members, samples].")

    selected = member_preds if vote_size is None else member_preds[:vote_size]
    if selected.shape[0] == 0:
        raise ValueError("At least one member prediction row is required.")
    if selected.shape[0] % 2 == 0:
        raise ValueError("Binary majority voting requires an odd number of members.")
    if not ((selected == 0) | (selected == 1)).all():
        raise ValueError("Binary majority voting expects class ids 0 or 1.")

    threshold = selected.shape[0] // 2 + 1
    votes_for_one = selected.long().sum(dim=0)
    preds = (votes_for_one >= threshold).long()
    prob_one = votes_for_one.float() / float(selected.shape[0])
    probs = torch.stack((1.0 - prob_one, prob_one), dim=1)
    return preds.cpu(), probs.cpu()


class MajorityVotingEnsembleTrainer:
    """Train missing ensemble members and evaluate hard majority votes."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.member_entries: list[dict] = []
        self.uuid = generate_uuid(str(config.model.name), self._ensemble_name())

    def train(self) -> str:
        """Train configured members, evaluate votes, and return the ensemble UUID."""
        self._validate_config()
        member_seeds = self._resolve_member_seeds()
        existing_uuids = list(self.config.ensemble.existing_member_uuids)
        self._validate_existing_checkpoints(existing_uuids)

        print(f"\n{'=' * 72}")
        print(f"Majority ensemble: {self.uuid}")
        print(
            f"Model: {self.config.model.name} | Task: {self.config.task} | "
            f"members={self.config.ensemble.member_count} | "
            f"vote_sizes={list(self.config.ensemble.vote_sizes)}"
        )
        print(
            f"Split seed: {self.config.split_seed()} | "
            f"member seeds: {list(member_seeds)}"
        )
        if existing_uuids:
            print(f"Existing members reused: {len(existing_uuids)}")
        print(f"{'=' * 72}\n")

        for index, uuid in enumerate(existing_uuids, start=1):
            self.member_entries.append(
                {
                    "member_index": index,
                    "seed": int(member_seeds[index - 1]),
                    "uuid": uuid,
                    "trained_in_this_run": False,
                }
            )

        for index in range(
            len(existing_uuids) + 1,
            self.config.ensemble.member_count + 1,
        ):
            seed = int(member_seeds[index - 1])
            uuid = f"{self.uuid}_member{index:02d}_seed{seed}"
            member_config = self._member_config(index=index, seed=seed, uuid=uuid)

            print(f"\nTraining ensemble member {index}: seed={seed}, uuid={uuid}")
            loader_helper = LoaderHelper(member_config)
            model_factory = lambda cfg=member_config: build_model(
                cfg.model,
                output_dim=cfg.model_output_dim(),
            )
            trainer = Trainer(
                config=member_config,
                model_factory=model_factory,
                loader_helper=loader_helper,
            )
            trainer.train()
            self.member_entries.append(
                {
                    "member_index": index,
                    "seed": seed,
                    "uuid": uuid,
                    "trained_in_this_run": True,
                }
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self._evaluate_majority_votes()
        return self.uuid

    def _ensemble_name(self) -> str:
        base = self.config.experiment.name or str(self.config.model.name)
        return f"{base} majority ensemble"

    def _validate_config(self) -> None:
        cfg = self.config
        ensemble = cfg.ensemble
        if not ensemble.enabled:
            raise ValueError("Ensemble trainer requires ensemble.enabled=true.")
        if not cfg.is_classification_task():
            raise ValueError("Majority voting is only supported for classification tasks.")
        if cfg.is_ordinal_classification():
            raise ValueError("Majority voting expects multiclass logits, not ordinal logits.")
        if int(cfg.model.num_classes) != 2:
            raise ValueError("Hard majority voting is currently limited to binary tasks.")
        if ensemble.voting != "majority":
            raise ValueError(f"Unsupported ensemble voting mode: {ensemble.voting}")
        if int(ensemble.member_count) < 1:
            raise ValueError("ensemble.member_count must be at least 1.")
        if len(ensemble.existing_member_uuids) > int(ensemble.member_count):
            raise ValueError("existing_member_uuids cannot exceed member_count.")
        missing_members = len(ensemble.existing_member_uuids) < int(
            ensemble.member_count
        )
        if missing_members and not ensemble.train_missing_members:
            raise ValueError(
                "train_missing_members=false requires existing_member_uuids for every member."
            )

        for vote_size in ensemble.vote_sizes:
            if int(vote_size) < 1:
                raise ValueError("ensemble.vote_sizes must be positive.")
            if int(vote_size) % 2 == 0:
                raise ValueError("Binary majority vote sizes must be odd.")
            if int(vote_size) > int(ensemble.member_count):
                raise ValueError("ensemble.vote_sizes cannot exceed member_count.")

    def _resolve_member_seeds(self) -> tuple[int, ...]:
        member_count = int(self.config.ensemble.member_count)
        seeds = tuple(int(seed) for seed in self.config.ensemble.member_seeds)
        if not seeds:
            base_seed = int(self.config.training.seed)
            seeds = tuple(base_seed + offset for offset in range(member_count))
        if len(seeds) < member_count:
            raise ValueError(
                "ensemble.member_seeds must provide at least member_count seeds."
            )
        return seeds[:member_count]

    def _validate_existing_checkpoints(self, uuids: list[str]) -> None:
        for uuid in uuids:
            for fold in range(1, int(self.config.training.k_folds) + 1):
                path = self._checkpoint_path(uuid, fold)
                if not path.exists():
                    raise FileNotFoundError(f"Missing ensemble member checkpoint: {path}")

    def _member_config(self, index: int, seed: int, uuid: str) -> Config:
        experiment_name = self.config.experiment.name or str(self.config.model.name)
        member_experiment = replace(
            self.config.experiment,
            name=f"{experiment_name} majority member {index:02d} seed {seed}",
        )
        return replace(
            self.config,
            training=replace(self.config.training, seed=seed),
            resume=replace(
                self.config.resume,
                enabled=True,
                uuid=uuid,
                start_fold=0,
            ),
            experiment=member_experiment,
        )

    def _checkpoint_path(self, uuid: str, fold: int) -> Path:
        return (
            self.config.paths.weights
            / self.config.task
            / uuid
            / f"fold{fold}_best_weight.pth"
        )

    def _evaluate_majority_votes(self) -> None:
        loader_helper = LoaderHelper(self.config)
        vote_results = {
            int(vote_size): [] for vote_size in self.config.ensemble.vote_sizes
        }
        member_uuids = [entry["uuid"] for entry in self.member_entries]

        save_dir = self.config.paths.figures / self.config.task / self.uuid
        save_dir.mkdir(parents=True, exist_ok=True)

        for fold in range(int(self.config.training.k_folds)):
            labels, member_preds, sample_indices = self._collect_member_predictions(
                loader_helper=loader_helper,
                member_uuids=member_uuids,
                fold=fold,
            )
            for vote_size in vote_results:
                preds, probs = majority_vote_binary(member_preds, vote_size=vote_size)
                metrics = compute_classification_metrics(
                    labels=labels,
                    preds=preds,
                    probs=probs,
                    num_classes=int(self.config.model.num_classes),
                    sample_indices=sample_indices,
                )
                metrics.fold = fold + 1
                vote_results[vote_size].append(metrics)
                vote_dir = save_dir / f"ensemble_at_{vote_size}"
                save_predictions(
                    metrics=metrics,
                    dataset=loader_helper.dataset,
                    fold_indices=loader_helper.fold_indices[fold][1],
                    save_path=vote_dir,
                    fold=fold + 1,
                    task_type=self.config.data.target_mode,
                    class_names=loader_helper.get_class_names(),
                )

        self._write_results(save_dir, vote_results, loader_helper.get_class_names())
        self._write_summary_csv(save_dir, vote_results)
        print(f"\nEnsemble majority results saved to: {save_dir}")

    def _collect_member_predictions(
        self,
        loader_helper,
        member_uuids: list[str],
        fold: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int] | None]:
        labels: torch.Tensor | None = None
        sample_indices: list[int] | None = None
        member_preds: list[torch.Tensor] = []
        dataloader = loader_helper.get_test_dl(fold)

        for uuid in member_uuids:
            model = build_model(
                self.config.model,
                device=self.device,
                output_dim=self.config.model_output_dim(),
            )
            checkpoint = load_checkpoint(
                self._checkpoint_path(uuid, fold + 1),
                model,
                self.device,
            )
            stats = checkpoint.get("target_stats", {})
            del checkpoint
            metrics = evaluate(
                model=model,
                dataloader=dataloader,
                device=self.device,
                task_type=self.config.data.target_mode,
                target_mean=float(stats.get("mean", 0.0)),
                target_std=float(stats.get("std", 1.0)),
                use_amp=bool(self.config.training.amp and self.device.type == "cuda"),
                num_classes=int(self.config.model.num_classes),
                classification_mode=self.config.training.classification_mode,
            )
            if not isinstance(metrics, ClassificationMetrics):
                raise TypeError("Ensemble voting expected classification metrics.")
            if labels is None:
                labels = metrics.labels
                sample_indices = metrics.sample_indices
            elif not torch.equal(labels, metrics.labels):
                raise ValueError(
                    f"Label order mismatch for member {uuid} fold {fold + 1}."
                )
            if sample_indices != metrics.sample_indices:
                raise ValueError(
                    f"Sample order mismatch for member {uuid} fold {fold + 1}."
                )
            member_preds.append(metrics.preds.long().cpu())
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if labels is None:
            raise ValueError(f"No predictions collected for fold {fold + 1}.")
        return labels.cpu(), torch.stack(member_preds, dim=0), sample_indices

    def _write_results(
        self,
        save_dir: Path,
        vote_results: dict[int, list[ClassificationMetrics]],
        class_names: list[str],
    ) -> None:
        folds_by_vote_size = {
            str(vote_size): [self._fold_entry(metrics) for metrics in folds]
            for vote_size, folds in vote_results.items()
        }
        summary_by_vote_size = {
            str(vote_size): self._summary(folds)
            for vote_size, folds in vote_results.items()
        }
        total_confusion_matrix_by_vote_size = {}
        for vote_size, folds in vote_results.items():
            matrices = [
                np.asarray(metrics.confusion_matrix, dtype=int)
                for metrics in folds
                if metrics.confusion_matrix is not None
            ]
            if matrices:
                total_confusion_matrix_by_vote_size[str(vote_size)] = (
                    np.sum(matrices, axis=0).astype(int).tolist()
                )

        payload = {
            "meta": {
                "uuid": self.uuid,
                "model": self.config.model.name,
                "experiment": {
                    "name": self._ensemble_name(),
                    "description": self.config.experiment.description,
                },
                "task": self.config.task,
                "task_type": self.config.data.target_mode,
                "timestamp": datetime.now().isoformat(),
                "class_names": class_names,
                "ensemble": {
                    "voting": self.config.ensemble.voting,
                    "member_count": self.config.ensemble.member_count,
                    "vote_sizes": list(self.config.ensemble.vote_sizes),
                    "split_seed": self.config.split_seed(),
                    "member_seeds": [
                        int(entry["seed"]) for entry in self.member_entries
                    ],
                    "existing_member_uuids": list(
                        self.config.ensemble.existing_member_uuids
                    ),
                    "members": self.member_entries,
                },
            },
            "folds_by_vote_size": folds_by_vote_size,
            "summary_by_vote_size": summary_by_vote_size,
            "total_confusion_matrix_by_vote_size": total_confusion_matrix_by_vote_size,
        }
        with (save_dir / "ensemble_results.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _write_summary_csv(
        self,
        save_dir: Path,
        vote_results: dict[int, list[ClassificationMetrics]],
    ) -> None:
        fieldnames = ["vote_size"]
        for key in CLASSIFICATION_SUMMARY_KEYS:
            fieldnames.extend((f"mean_{key}", f"std_{key}"))

        with (save_dir / "ensemble_summary.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for vote_size, folds in vote_results.items():
                row = {"vote_size": vote_size}
                row.update(self._summary(folds))
                writer.writerow(row)

    def _fold_entry(self, metrics: ClassificationMetrics) -> dict:
        return {
            "fold": metrics.fold,
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
            "balanced_accuracy": metrics.balanced_accuracy,
            "sensitivity": metrics.sensitivity,
            "specificity": metrics.specificity,
            "confusion_matrix": metrics.confusion_matrix,
        }

    def _summary(
        self,
        folds: list[ClassificationMetrics],
    ) -> dict[str, float | None]:
        summary: dict[str, float | None] = {}
        for key in CLASSIFICATION_SUMMARY_KEYS:
            values = [float(getattr(metrics, key)) for metrics in folds]
            if values:
                summary[f"mean_{key}"] = round(float(np.mean(values)), 5)
                summary[f"std_{key}"] = round(float(np.std(values)), 5)
            else:
                summary[f"mean_{key}"] = None
                summary[f"std_{key}"] = None
        return summary
