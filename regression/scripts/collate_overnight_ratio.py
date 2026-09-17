#!/usr/bin/env python3
"""Collate the overnight two-stage runs into one table to read over breakfast.

Each run wrote its own JSON; on their own they are seven isolated numbers. What
decides anything is the paired comparison: 1-channel against 2-channel across the
same three seeds, and the two-stage regressor against the direct classifier at
matched settings. Seeds are reported with their spread, because a single seed on
n=200 moves by more than any effect seen on this cohort so far.

Runs that crashed are listed rather than skipped -- a missing row is a result
about the run, and the previous overnight batch hid an OOM kill by omitting it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    return p.parse_args()


def load_ratio_run(directory: Path) -> dict | None:
    hits = sorted(directory.glob("results_seed*.json"))
    if not hits:
        return None
    payload = json.loads(hits[0].read_text(encoding="utf-8-sig"))
    hold = payload.get("holdout") or {}
    out = {"kind": "ratio",
           "seed": payload.get("meta", {}).get("seed"),
           "ratio_mae": payload.get("holdout_ratio", {}).get("mae"),
           "ratio_r": payload.get("holdout_ratio", {}).get("pearson_r")}
    for tag in ("fixed70", "gli"):
        block = hold.get(f"{tag}_at_label_rule")
        if not block:
            continue
        out[f"{tag}_auc"] = block["auc"]
        out[f"{tag}_balacc"] = block["balanced_accuracy"]
        out[f"{tag}_sens"] = block["sensitivity"]
        out[f"{tag}_border"] = block.get("by_band", {}).get("borderline", {}).get("auc")
        cal = hold.get(f"{tag}_at_calibrated_offset")
        if cal:
            out[f"{tag}_balacc_cal"] = cal["balanced_accuracy"]
    return out


def load_cls_run(directory: Path) -> dict | None:
    hits = sorted(directory.glob("*holdout_predictions.json"))
    if not hits:
        return None
    payload = json.loads(hits[0].read_text(encoding="utf-8-sig"))
    m = payload.get("metrics") or {}
    return {"kind": "classification",
            "seed": (payload.get("meta") or {}).get("seeds"),
            "fixed70_auc": m.get("auc_mean_member_probability") or m.get("auc"),
            "fixed70_balacc": m.get("balanced_accuracy"),
            "fixed70_sens": m.get("sensitivity")}


def summarise(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return "  -"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"


def main() -> None:
    args = parse_args()
    groups: dict[str, list[dict]] = {}
    missing: list[str] = []

    for directory in sorted(p for p in args.run_dir.iterdir() if p.is_dir()):
        name = directory.name
        row = load_ratio_run(directory) or load_cls_run(directory)
        if row is None:
            missing.append(name)
            continue
        family = name.rsplit("_seed", 1)[0]
        groups.setdefault(family, []).append(row)

    print("")
    print("=" * 96)
    print("過夜二階段回歸結果")
    print("=" * 96)
    if missing:
        print("\n沒有產出結果的執行(崩了或還沒跑完):")
        for name in missing:
            print(f"  {name}")

    print("\n--- 固定比值 70 標籤 ---")
    print("{:24s}{:>6}{:>18}{:>18}{:>18}{:>14}".format(
        "執行", "n", "AUC", "平衡準確", "敏感度", "邊界AUC"))
    print("-" * 96)
    for family, rows in sorted(groups.items()):
        print("{:24s}{:>6}{:>18}{:>18}{:>18}{:>14}".format(
            family, len(rows),
            summarise(rows, "fixed70_auc"), summarise(rows, "fixed70_balacc"),
            summarise(rows, "fixed70_sens"), summarise(rows, "fixed70_border")))

    ratio_groups = {f: r for f, r in groups.items()
                    if any(x.get("kind") == "ratio" for x in r)}
    if ratio_groups:
        print("\n--- GLI LLN 標籤(同一個模型,只換判讀規則,沒有重新訓練)---")
        print("{:24s}{:>6}{:>18}{:>18}{:>18}{:>14}".format(
            "執行", "n", "AUC", "平衡準確", "敏感度", "邊界AUC"))
        print("-" * 96)
        for family, rows in sorted(ratio_groups.items()):
            print("{:24s}{:>6}{:>18}{:>18}{:>18}{:>14}".format(
                family, len(rows),
                summarise(rows, "gli_auc"), summarise(rows, "gli_balacc"),
                summarise(rows, "gli_sens"), summarise(rows, "gli_border")))

        print("\n--- 比值預測本身的品質 ---")
        print("{:24s}{:>6}{:>18}{:>18}".format("執行", "n", "MAE (點)", "Pearson r"))
        print("-" * 68)
        for family, rows in sorted(ratio_groups.items()):
            print("{:24s}{:>6}{:>18}{:>18}".format(
                family, len(rows),
                summarise(rows, "ratio_mae"), summarise(rows, "ratio_r")))
        print("\n  參考:要在 70 附近分辨阻塞與否,MAE 需要接近 2 點。")

    print("\n對照基準(同一份凍結 200):")
    print("  只用年齡性別身高,無影像   固定70 AUC 0.6302   GLI AUC 0.5601")
    print("  %LAA-950 單獨              固定70 AUC 0.6194   GLI AUC 0.6145")
    print("  Mamba5 分類 5 模型         固定70 AUC 0.6424")
    print("  TapCT LoRA                 固定70 AUC 0.7253")
    print("\n  提醒:這份凍結 200 已被評分超過八次,名次帶有樂觀偏差。")
    print("  要選模型請看各執行自己的訓練 OOF,不要看這張表挑最大值。")


if __name__ == "__main__":
    main()
