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
    lines = ["# Hybrid-Mamba 五視窗蒸餾結果", "",
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
              "限制：hybrid 使用 GroupNorm，架構與論文 SE-ResNet50 不同；microbatch 1＋梯度累積 4；切片範圍為原文不明確處的實作假設。",
              "初次 FP16 在第一個 epoch 發生非有限 loss；本次改用 FP32，與 SE-ResNet50 的 AMP 精度不同。資料標籤沿用原始清單，包含無 post-BD 時使用 pre-BD 的情形。不能直接聲稱重現原論文效能。", "",
              "![Comparison](comparison.png)", "", "## 各視窗測試 AUC", "",
              "| Model | AUC |", "| --- | --- |"]
    for name, m in data["individual"].items():
        lines.append(f"| {name} | {m['auc']:.4f} |")
    previous_path = args.run.parent / "seresnet50_full777/results.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        lines += ["", "## 與 SE-ResNet50 的固定測試比較", "",
                  "| 方法 | SE-ResNet50 AUC | Hybrid-Mamba AUC |", "| --- | --- | --- |"]
        for name in names:
            lines.append(f"| {name} | {previous['ensemble'][name]['auc']:.4f} | {data['ensemble'][name]['auc']:.4f} |")
        lines += ["", "沿用相同資料與切分。架構、正規化層及數值精度不同；此表是本次實驗的描述性比較，不能據此判定架構普遍優劣。",
                  "初次 FP16 失敗與修復時間不包含在 FP32 程式計時內；從原始啟動至結果寫入約 7 小時 50 分。"]
    (args.run / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(args.run / "report.md")


if __name__ == "__main__":
    main()


