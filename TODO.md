# TODO: Angle 3-Class Augmentation Follow-Up

## Current Best Result

目前 `50/class` 是最值得當主線的設定。

| Target per class | Accuracy | Macro-F1 | Balanced Acc | Class 0 Recall | Class 1 Recall | Class 2 Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50/class | 0.7275 | 0.5605 | 0.6193 | 0.7143 | 0.4000 | 0.7660 |
| 75/class | 0.6967 | 0.4502 | 0.4570 | 0.5714 | 0.0000 | 0.8085 |
| 100/class | 0.7593 | 0.4677 | 0.5119 | 0.3571 | 0.2000 | 0.9362 |

結論：`50/class` 的 Macro-F1 和 Balanced Accuracy 最高，而且三類 recall 最平均。

## Why Not Prioritize 200/Class

目前不建議優先跑 `200/class`。

原因：

- 原始 training fold 大約只有 `11 / 4 / 37`。
- 拉到 `200/class` 會變成 `200 / 200 / 200`，也就是 `600 samples/epoch`。
- Class 1 原始只有約 4 位 training patients，補到 200 等於約放大 50 倍。
- 這比較像反覆看同幾位病人的小變形，不一定增加真正資訊。
- 50 -> 75 -> 100 的結果已經顯示「補更多不一定更好」。

預期風險：

- Training time 會明顯增加。
- Class 1 recall 可能仍不穩，甚至維持很低。
- Accuracy 可能上升，但 Macro-F1 / Balanced Accuracy 不一定提升。
- 模型可能更偏向 Normal 類。

## Next Experiments

優先順序：

1. 試 `40/class`。
2. 試 `60/class`。
3. 和目前 `50/class` 比較。
4. 若教授明確想看補更多的極端情況，再把 `200/class` 當 ablation。

建議主要比較指標：

- Macro-F1
- Balanced Accuracy
- Class 0 Recall
- Class 1 Recall
- Confusion Matrix

不要只看 Accuracy，因為 validation/test 裡 Normal 類最多，Accuracy 容易被 Normal 類撐高。

## Suggested Explanation For Professor

```text
目前 50/class 是最好的折衷點。它比 75/class 和 100/class 有更高的
Macro-F1 和 Balanced Accuracy，也保留較好的 class 0/class 1 recall。
200/class 可能只是把少數幾位病人的 CT 重複變形更多次，資訊增益有限，
所以建議先在 50 附近測 40/class 和 60/class，再決定是否需要跑 200/class。
```
