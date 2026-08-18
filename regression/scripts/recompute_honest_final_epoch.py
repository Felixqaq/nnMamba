#!/usr/bin/env python3
"""Recompute RQ metrics from the final training epoch instead of the best one.

The trainer reports `best_fold_result`: the highest-scoring evaluation across the
20 checkpoints of a fold (100 epochs / eval_interval 5). Those checkpoints are
scored on the held-out fold itself -- the split is 2-way (train/val), so the
"val" fold *is* the test fold. Taking the max over it selects on the test data.

This script re-reads the run logs and reports the epoch-100 evaluation instead,
which is the fixed-budget protocol rq_overview.md already claims to use.

Read-only: parses train_log/run_all_rq/*.log and writes to new files only.
Existing results.json / rq_overview.md are never touched.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
LOG_DIR = REG / "train_log" / "run_all_rq"
OUT_JSON = LOG_DIR / "honest_final_epoch.json"
OUT_MD = REG / "docs" / "rq_honest_recompute.md"

FINAL_EPOCH = 100

CLS_RE = re.compile(
    rf"^Epoch {FINAL_EPOCH}: Val Acc=([\d.]+), Macro-F1=([\d.]+), Bal Acc=([\d.]+)",
    re.M,
)
REG_RE = re.compile(
    rf"^Epoch {FINAL_EPOCH}: Val MAE=([\d.-]+), RMSE=([\d.-]+), R2=([\d.-]+)", re.M
)

# Reported (best-epoch) values from docs/rq_overview.md, for the inflation column.
REPORTED = {
    "config.rq1.normal_v_abnormal.image": 0.879,
    "config.rq1.normal_v_abnormal.fusion": 0.866,
    "config.rq2a.angle_3class.image": 0.518,
    "config.rq2a.angle_3class.fusion": 0.774,
    "config.rq2b.angle_binary.image": 0.885,
    "config.rq2b.angle_binary.fusion": 0.871,
    "config.rq3.oi_emphysema.image": 0.803,
    "config.rq3.oi_emphysema.fusion": 0.834,
    "config.rq2c.angle_reg.image": 15.48,  # MAE
    "config.rq2c.angle_reg.fusion": 14.61,
}


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.pstdev(values)


def parse(log: Path) -> dict | None:
    text = log.read_text(encoding="utf-8", errors="replace")
    is_reg = "angle_reg" in log.stem

    if is_reg:
        rows = [tuple(map(float, m)) for m in REG_RE.findall(text)]
        keys = ("mae", "rmse", "r2")
    else:
        rows = [tuple(map(float, m)) for m in CLS_RE.findall(text)]
        keys = ("accuracy", "macro_f1", "balanced_accuracy")

    if not rows:
        return None

    folds = [dict(zip(keys, r)) for r in rows]
    summary = {}
    for k in keys:
        m, s = mean_std([f[k] for f in folds])
        summary[f"mean_{k}"] = round(m, 5)
        summary[f"std_{k}"] = round(s, 5)

    return {
        "run": log.stem,
        "task_type": "regression" if is_reg else "classification",
        "n_folds": len(folds),
        "folds": folds,
        "summary": summary,
    }


def main() -> None:
    results = []
    for log in sorted(LOG_DIR.glob("config.*.log")):
        parsed = parse(log)
        if parsed is None:
            print(f"  !! no epoch-{FINAL_EPOCH} rows in {log.name}, skipped")
            continue
        if parsed["n_folds"] != 5:
            print(f"  !! {log.name}: expected 5 folds, found {parsed['n_folds']}")
        results.append(parsed)

    OUT_JSON.write_text(
        json.dumps(
            {
                "note": (
                    f"Metrics from the final epoch ({FINAL_EPOCH}) of each fold, parsed "
                    "from run logs. The published results.json reports the best epoch, "
                    "which is selected on the held-out fold itself."
                ),
                "source_logs": str(LOG_DIR),
                "runs": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# RQ1/2/3 — 誠實版重算（最終 epoch）",
        "",
        "本檔案由 `scripts/recompute_honest_final_epoch.py` 從 `train_log/run_all_rq/*.log`",
        "重算而來，**不修改** `results.json` 或 `rq_overview.md`。",
        "",
        "## 為什麼要重算",
        "",
        "trainer 回報的是 `best_fold_result` — 一折 100 epochs、每 5 epoch 評一次，",
        "共 20 個評估點取**最大值**。但資料只切 train/val 兩份（`loader.py` 的",
        "`StratifiedKFold`），那個 val 折就是 held-out 測試折，所以 max 是在測試資料上挑的。",
        "`rq_overview.md` 宣稱的協定是「最終 epoch 評估」，與程式實際行為不符。",
        "下表改用每折第 100 epoch 的評估值，即文件宣稱的固定預算協定。",
        "",
        "## 分類任務",
        "",
        "| run | 誠實版 Acc (mean±std) | 報告值 Acc | 差距 |",
        "|-----|----------------------|-----------|------|",
    ]
    for r in results:
        if r["task_type"] != "classification":
            continue
        m = r["summary"]["mean_accuracy"]
        s = r["summary"]["std_accuracy"]
        rep = REPORTED.get(r["run"])
        delta = f"+{rep - m:.3f}" if rep is not None else "—"
        rep_s = f"{rep:.3f}" if rep is not None else "—"
        lines.append(f"| {r['run']} | {m:.3f} ± {s:.3f} | {rep_s} | {delta} |")

    lines += [
        "",
        "## 迴歸任務 (RQ2c)",
        "",
        "| run | 誠實版 MAE (mean±std) | 誠實版 R² | 報告值 MAE | 差距 |",
        "|-----|----------------------|----------|-----------|------|",
    ]
    for r in results:
        if r["task_type"] != "regression":
            continue
        m = r["summary"]["mean_mae"]
        s = r["summary"]["std_mae"]
        r2 = r["summary"]["mean_r2"]
        rep = REPORTED.get(r["run"])
        delta = f"{rep - m:+.2f}" if rep is not None else "—"
        rep_s = f"{rep:.2f}" if rep is not None else "—"
        lines.append(f"| {r['run']} | {m:.2f} ± {s:.2f} | {r2:.3f} | {rep_s} | {delta} |")

    lines += ["", "## 每折明細", ""]
    for r in results:
        vals = ", ".join(
            f"{f.get('accuracy', f.get('mae')):.4f}" for f in r["folds"]
        )
        lines.append(f"- `{r['run']}`: {vals}")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
