"""Merge per-patient predictions from several models into one hard-vote table.

Each model is given as ``NAME=PATH`` where PATH is either

* a run directory containing ``fold*_predictions.json`` (written by the trainer), or
* a flat JSON file for a model trained elsewhere, in the schema::

      {"4204917": {"pred_label": "Abnormal", "prob_positive": 0.83}, ...}

  ``pred_label`` must be one of the task's exact class strings (taken from the
  run directories); ``prob_positive`` is optional.

Hard majority vote over ``n_models`` members. When some models are still missing,
a patient is already decided if the available votes cannot be overturned by the
remaining ones; otherwise it is reported as ``pending``.

Usage::

    python scripts/build_vote_table.py \
        --model image=figures/RQ1_normal_v_abnormal/<run> \
        --model fusion=figures/RQ1_normal_v_abnormal/<run> \
        --n-models 3 \
        --out docs/per_patient_vote_rq1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

def load_run_dir(path: Path) -> tuple[dict[str, dict], list[str]]:
    """Read fold*_predictions.json from a trainer run directory.

    Returns the per-patient records and the task's class names, ordered as the
    trainer wrote them (index 0 is the positive/disease class).
    """
    records: dict[str, dict] = {}
    class_names: list[str] | None = None
    fold_files = sorted(path.glob("fold*_predictions.json"))
    if not fold_files:
        raise FileNotFoundError(f"No fold*_predictions.json under {path}")

    for fold_file in fold_files:
        payload = json.loads(fold_file.read_text())
        fold = payload["fold"]
        if class_names is None:
            class_names = list(payload["class_names"])
        elif list(payload["class_names"]) != class_names:
            raise ValueError(f"Inconsistent class_names across folds of {path}")
        positive = class_names[0]
        for row in payload["predictions"]:
            pid = str(row["patient_id"])
            if pid in records:
                raise ValueError(f"Patient {pid} appears in more than one fold of {path}")
            records[pid] = {
                "fold": fold,
                "true_label": row["true_label"],
                "gold_stage_label": row.get("gold_stage_label"),
                "pred_label": row["pred_label"],
                "prob_positive": row["probabilities"].get(positive),
            }
    return records, class_names


def load_flat_json(path: Path) -> tuple[dict[str, dict], None]:
    """Read an externally produced per-patient prediction file."""
    payload = json.loads(path.read_text())
    records = {
        str(pid): {
            "fold": None,
            "true_label": entry.get("true_label"),
            "gold_stage_label": entry.get("gold_stage_label"),
            "pred_label": entry["pred_label"],
            # `prob_abnormal` accepted for backward compatibility.
            "prob_positive": entry.get("prob_positive", entry.get("prob_abnormal")),
        }
        for pid, entry in payload.items()
    }
    return records, None


def load_model(spec: str) -> tuple[str, dict[str, dict], list[str] | None]:
    name, _, raw_path = spec.partition("=")
    if not name or not raw_path:
        raise ValueError(f"--model expects NAME=PATH, got {spec!r}")
    path = Path(raw_path)
    records, class_names = load_run_dir(path) if path.is_dir() else load_flat_json(path)
    return name, records, class_names


def hard_vote(votes: list[str], n_models: int) -> tuple[str | None, str]:
    """Return (winning label, status) for a majority vote over n_models members."""
    threshold = n_models // 2 + 1
    counts = Counter(votes)
    label, count = counts.most_common(1)[0]
    if count >= threshold:
        return label, "decided"

    missing = n_models - len(votes)
    runner_up = counts.most_common(2)[1][1] if len(counts) > 1 else 0
    if count + missing < threshold:
        # Nobody can reach the threshold -> tie even when fully populated.
        return None, "tie"
    if missing == 0:
        return None, "tie"
    if count > runner_up + missing:
        return label, "decided"
    return None, "pending"


def build_rows(models: dict[str, dict[str, dict]], n_models: int) -> list[dict]:
    all_ids: set[str] = set()
    for records in models.values():
        all_ids |= set(records)

    rows = []
    for pid in sorted(all_ids):
        present = {name: recs[pid] for name, recs in models.items() if pid in recs}

        truths = {r["true_label"] for r in present.values() if r["true_label"]}
        if len(truths) > 1:
            raise ValueError(f"Patient {pid} has conflicting true labels: {truths}")
        folds = {r["fold"] for r in present.values() if r["fold"] is not None}
        if len(folds) > 1:
            raise ValueError(f"Patient {pid} is assigned to different folds: {folds}")
        stages = {r["gold_stage_label"] for r in present.values() if r["gold_stage_label"]}

        votes = [r["pred_label"] for r in present.values()]
        vote_label, status = hard_vote(votes, n_models)

        rows.append(
            {
                "patient_id": pid,
                "fold": next(iter(folds)) if folds else None,
                "true_label": next(iter(truths)) if truths else None,
                "gold_stage_label": next(iter(stages)) if stages else None,
                "models": {
                    name: {
                        "pred_label": r["pred_label"],
                        "prob_positive": r["prob_positive"],
                        "correct": (
                            None if not r["true_label"] else r["pred_label"] == r["true_label"]
                        ),
                    }
                    for name, r in present.items()
                },
                "missing_models": sorted(set(models) - set(present)),
                "votes": dict(Counter(votes)),
                "vote_label": vote_label,
                "vote_status": status,
                "vote_correct": (
                    None
                    if vote_label is None or not truths
                    else vote_label == next(iter(truths))
                ),
            }
        )
    return rows


def confusion_metrics(pairs: list[tuple[str, str]], positive: str) -> dict:
    """Accuracy / sensitivity / specificity from (true, pred) pairs."""
    if not pairs:
        return {"n": 0}
    tp = sum(1 for t, p in pairs if t == positive and p == positive)
    fn = sum(1 for t, p in pairs if t == positive and p != positive)
    tn = sum(1 for t, p in pairs if t != positive and p != positive)
    fp = sum(1 for t, p in pairs if t != positive and p == positive)
    return {
        "n": len(pairs),
        "accuracy": round((tp + tn) / len(pairs), 4),
        "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
    }


def summarize(rows: list[dict], model_names: list[str], positive: str) -> dict:
    summary = {}
    for name in model_names:
        pairs = [
            (r["true_label"], r["models"][name]["pred_label"])
            for r in rows
            if name in r["models"] and r["true_label"]
        ]
        summary[name] = confusion_metrics(pairs, positive)

    decided = [r for r in rows if r["vote_status"] == "decided" and r["true_label"]]
    summary["vote"] = confusion_metrics(
        [(r["true_label"], r["vote_label"]) for r in decided], positive
    )
    summary["vote"]["status_counts"] = dict(Counter(r["vote_status"] for r in rows))
    return summary


def short_label(name: str, class_names: list[str]) -> str:
    """Trim a verbose class name for table cells, unless that would collide."""
    trimmed = {c: c.split("(")[0].split("/")[0].strip() for c in class_names}
    if len(set(trimmed.values())) < len(class_names):
        return name
    return trimmed.get(name, name)


def render_markdown(payload: dict) -> str:
    meta = payload["meta"]
    rows = payload["patients"]
    names = meta["model_names"]
    class_names = meta["class_names"]
    positive = meta["positive_class"]
    short = {c: short_label(c, class_names) for c in class_names}
    lines = [
        "# Per-patient predictions and hard-vote result",
        "",
        f"- Generated: {meta['generated_at']}",
        f"- Cohort: {meta['n_patients']} patients",
        f"- Vote: hard majority over {meta['n_models']} models "
        f"({len(names)} available: {', '.join(names)})",
        "",
        "> **Caveat.** These per-patient predictions were saved at each fold's "
        "best epoch, and that epoch was selected on the test fold itself "
        "(`trainer.py` only calls `save_predictions` when the fold score improves). "
        "The numbers below are therefore an optimistic upper bound, not an unbiased "
        "generalization estimate.",
        "",
        "## Classes",
        "",
        f"- Positive: `{positive}`" + (f" (shown as **{short[positive]}**)" if short[positive] != positive else ""),
    ]
    negative = next(c for c in class_names if c != positive)
    lines.append(
        f"- Negative: `{negative}`"
        + (f" (shown as **{short[negative]}**)" if short[negative] != negative else "")
    )

    lines += ["", "## Sources", ""]
    for name in names:
        lines.append(f"- `{name}`: `{meta['model_sources'][name]}`")

    lines += [
        "",
        "## Summary",
        "",
        f"| model | n | Accuracy | Sensitivity ({short[positive]}) | "
        f"Specificity ({short[negative]}) | TP | FN | TN | FP |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in names + ["vote"]:
        stats = payload["summary"][key]
        if not stats.get("n"):
            lines.append(f"| {key} | 0 | – | – | – | – | – | – | – |")
            continue
        cm = stats["confusion"]
        lines.append(
            f"| {key} | {stats['n']} | {stats['accuracy']} | "
            f"{stats['sensitivity']} | {stats['specificity']} | "
            f"{cm['tp']} | {cm['fn']} | {cm['tn']} | {cm['fp']} |"
        )

    status_counts = payload["summary"]["vote"]["status_counts"]
    pending = [r for r in rows if r["vote_status"] == "pending"]
    lines += [
        "",
        f"Vote status: {status_counts}. "
        "`decided` = the available votes already form a majority no missing model can "
        "overturn; `pending` = the outcome depends on a model that is not in this table.",
    ]

    if pending:
        decided_correct = sum(1 for r in rows if r["vote_correct"])
        total = sum(1 for r in rows if r["true_label"])
        lines += [
            "",
            f"**The vote row above is not comparable to the single-model rows.** It covers "
            f"only the {payload['summary']['vote']['n']} decided patients, and the "
            f"{len(pending)} pending ones are exactly the cases the models disagree on — "
            "i.e. the hard ones are excluded. Over the full cohort the final accuracy will "
            f"land between {decided_correct}/{total} = "
            f"{decided_correct / total:.4f} (missing model gets every pending case wrong) "
            f"and {decided_correct + len(pending)}/{total} = "
            f"{(decided_correct + len(pending)) / total:.4f} (gets them all right).",
            "",
            f"### Pending — decided by the missing model ({len(pending)})",
            "",
            "| patient_id | fold | true | GOLD | "
            + " | ".join(names)
            + " |",
            "| " + " | ".join("---" for _ in range(4 + len(names))) + " |",
        ]
        for row in pending:
            cells = [
                row["patient_id"],
                str(row["fold"] or "–"),
                short.get(row["true_label"], "–"),
                row["gold_stage_label"] or "–",
            ]
            cells += [
                short[row["models"][n]["pred_label"]] if n in row["models"] else "–"
                for n in names
            ]
            lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Per-patient table", ""]

    header = ["patient_id", "fold", "true", "GOLD"]
    for name in names:
        header += [f"{name} pred", f"{name} p(pos)"]
    header += ["vote", "status", "vote correct"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    for row in rows:
        cells = [
            row["patient_id"],
            str(row["fold"] or "–"),
            short.get(row["true_label"], "–"),
            row["gold_stage_label"] or "–",
        ]
        for name in names:
            entry = row["models"].get(name)
            if entry is None:
                cells += ["–", "–"]
            else:
                mark = "" if entry["correct"] is None else ("" if entry["correct"] else " ✗")
                prob = entry["prob_positive"]
                cells += [
                    f"{short[entry['pred_label']]}{mark}",
                    "–" if prob is None else f"{prob:.3f}",
                ]
        cells += [
            short.get(row["vote_label"], "–"),
            row["vote_status"],
            "–" if row["vote_correct"] is None else ("yes" if row["vote_correct"] else "no"),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Run directory or flat prediction JSON, repeatable.",
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=None,
        help="Total members the vote is meant to have (default: number of --model).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path without extension; .json and .md are written.",
    )
    args = parser.parse_args()

    models: dict[str, dict[str, dict]] = {}
    sources: dict[str, str] = {}
    class_names: list[str] | None = None
    for spec in args.model:
        name, records, run_classes = load_model(spec)
        if name in models:
            raise ValueError(f"Duplicate model name {name!r}")
        if run_classes is not None:
            if class_names is None:
                class_names = run_classes
            elif run_classes != class_names:
                raise ValueError(
                    f"Model {name!r} uses class names {run_classes}, expected {class_names}"
                )
        models[name] = records
        sources[name] = spec.partition("=")[2]

    if class_names is None:
        raise ValueError(
            "Class names could not be determined: pass at least one run directory."
        )
    if len(class_names) != 2:
        raise ValueError(f"Expected a binary task, got classes {class_names}")

    # Externally produced files must use the exact class strings, or the vote is silently wrong.
    known = set(class_names)
    for name, records in models.items():
        unknown = {r["pred_label"] for r in records.values()} - known
        if unknown:
            raise ValueError(
                f"Model {name!r} predicts unknown labels {sorted(unknown)}; "
                f"expected one of {class_names}"
            )

    n_models = args.n_models or len(models)
    if n_models < len(models):
        raise ValueError("--n-models cannot be smaller than the number of --model entries")
    if n_models % 2 == 0:
        raise ValueError("Hard majority voting requires an odd number of models")

    rows = build_rows(models, n_models)
    names = list(models)
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "n_patients": len(rows),
            "n_models": n_models,
            "model_names": names,
            "model_sources": sources,
            "vote": "hard majority",
            "class_names": class_names,
            "positive_class": class_names[0],
            "caveat": (
                "Per-patient predictions come from each fold's best epoch, selected on "
                "the test fold itself; metrics are an optimistic upper bound."
            ),
        },
        "summary": summarize(rows, names, class_names[0]),
        "patients": rows,
    }

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    out.with_suffix(".md").write_text(render_markdown(payload))
    print(f"Wrote {out.with_suffix('.json')}")
    print(f"Wrote {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
