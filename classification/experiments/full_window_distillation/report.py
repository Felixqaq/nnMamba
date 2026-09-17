"""Create a compact report and figures only after the full run completes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    data = json.loads((args.run / "results.json").read_text())
    status = json.loads((args.run / "status.json").read_text())
    if data["status"] != "complete" or status["status"] != "complete":
        raise RuntimeError("Run has not completed")
    names = ["baseline", "control", "distilled"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    positions = np.arange(3)
    for i, key in enumerate(["auc", "balanced_accuracy", "sensitivity"]):
        axes[0].bar(positions+(i-1)*.25, [data["ensemble"][n][key] for n in names], .25, label=key)
    axes[0].set(xticks=positions, xticklabels=names, ylim=(0, 1), title="Five-window ensembles")
    axes[0].legend(fontsize=8)
    for ax, name in zip(axes[1:], ["control", "distilled"]):
        cm = np.array(data["ensemble"][name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=200)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-COPD", "COPD"],
               yticklabels=["Non-COPD", "COPD"], xlabel="Predicted", ylabel="True", title=name)
    fig.savefig(args.run / "comparison.png", dpi=160)
    plt.close(fig)
    lines = ["# SE-ResNet50 五視窗蒸餾結果", "",
             "777 人：461 訓練、116 驗證、200 固定測試；單一種子、本機資料集保留測試，非獨立外部驗證。", "",
             f"老師視窗：{data['teacher_window']}。原尺寸 32×512×512，五個基礎模型、四個蒸餾學生、四個配對對照。", "",
             "| Ensemble | AUC | Accuracy | Balanced accuracy | Sensitivity | Specificity |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name in names:
        m = data["ensemble"][name]
        lines.append(f"| {name} | " + " | ".join(f"{m[k]:.4f}" for k in
                     ["auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity"]) + " |")
    lines += ["", f"蒸餾－等訓練量對照 AUC：{data['distilled_minus_matched_control_auc']:.4f}；",
              f"配對 bootstrap 95% CI：{data['matched_delta_auc_95ci']}。", "",
              f"程式計時（含資料準備、訓練及評估）：{data['invocation_seconds']/3600:.2f} 小時。", "",
              f"蒸餾－初始基礎 AUC：{data['delta_auc']:.4f}；配對 95% CI：{data['delta_auc_95ci']}。",
              "兩組差值區間均跨過 0，本次沒有證據顯示蒸餾提升 AUC。", "",
              "三組 stacking 都只使用驗證集訓練，閾值也只從驗證集選擇。測試集 140 非 COPD、60 COPD；全部預測非 COPD 的 accuracy 為 70%，因此不能只看 accuracy。", "",
              "限制：microbatch 1＋梯度累積不等於論文 batch 4 的 BatchNorm；切片範圍為原文不明確處的實作假設。",
              "使用 AMP 混合精度。資料標籤沿用原始清單，包含無 post-BD 時使用 pre-BD 的情形。不能直接聲稱重現原論文效能。", "",
              "![Comparison](comparison.png)", "", "## 各視窗測試 AUC", "",
              "| Model | AUC |", "| --- | --- |"]
    for name, m in data["individual"].items():
        lines.append(f"| {name} | {m['auc']:.4f} |")
    (args.run / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(args.run / "report.md")


if __name__ == "__main__":
    main()
