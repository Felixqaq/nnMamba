# RQ 1/2/3 兩主模型 × 66 病人 對齊矩陣 — 實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 產出 10 個完全對齊的 config(2 主模型 × 5 任務)、必要的 66 病人資料、一鍵驅動腳本與 RQ 組織文件,讓 RQ1/2/3 都用同一協定在 66 病人上比較 image-only 與 TAP-CT late fusion。

**Architecture:** loader (`data/loader.py`) 在記憶體中由 `data_root`+`labels_json`+`target_mode` 建 manifest 並切 fold;config 的 `manifest:` 只是輸出 dump 路徑。因此對齊工作 = 產生協定一致的 config + 為 RQ1 準備一個臨床分組(33/33)的資料夾。訓練入口是 `python train.py --config <path>`。

**Tech Stack:** Python 3、PyTorch、conda env `nnMamba`(有 numpy/torch;系統 python3 沒有 numpy)、YAML config。

## Global Constraints

- 所有指令在 `regression/` 目錄、conda env `nnMamba` 下執行(例:`conda run -n nnMamba python train.py --config ...`)。
- **統一協定(全 10 config 一致,兩模型唯一差異 = 有無 TAP-CT):**
  - `training.seed: 42`、`training.k_folds: 5`、`training.epochs: 100`
  - `early_stopping.enabled: false`
  - `data.balanced_sampling: true`
  - `data.augmentation`: `enabled: true`、`balance_then_augment: true`、`views_per_sample: 5`、`probability: 1.0`、`rotation_degrees: 7.0`、`translation_fraction: 0.05`、`scale_range: [0.95,1.05]`、`intensity_scale_range: [0.95,1.05]`、`intensity_shift_range: [-0.1,0.1]`、`noise_std: 0.03`
  - `data.image_size: [112,136,112]`、`intensity_window: [-1000.0,400.0]`、`input_normalization: zscore`
  - `gradcam.enabled: false`
  - 決策 argmax、不調閾值(不新增 threshold 邏輯)
- **兩模型差異只有這三行:** `model.name`(`hybrid_mamba_attention` vs `hybrid_mamba_tapct_fusion`)、`data.tapct_features`(`null` vs `./embeddings/tapct_s_3d/features.npz`)、fusion 需 `model.tapct_embedding_dim: 1152`。
- **任務本質差異(允許):** 分類→`loss: cross_entropy`、`target_normalization: none`、`num_classes` 依任務;迴歸(angle 值)→`num_classes: 1`、`loss: auto`、`target_normalization: zscore`。
- **舊 config / manifest 一律不動、不刪**,只新增。
- **不做「量化標記」方法、不改模型架構、不碰 merlin/abmil。**

### 共用協定 config 模板(下列各 Task 以此為基礎,只改標示欄位)

```yaml
model:
  name: <MODEL>                 # hybrid_mamba_attention | hybrid_mamba_tapct_fusion
  in_channels: 1
  num_classes: <NUM_CLASSES>
  hidden_dim: 256
  dropout: 0.3
  base_channels: 32
  blocks: 3
  attn_heads: 8
  attn_layers: 1
  attn_mlp_ratio: 2.0
  attn_dropout: 0.1
  tapct_embedding_dim: 1152     # fusion 用;image-only 保留無妨
  fusion_projection_dim: 128
  fusion_dropout: 0.1
  feature_size: 24
  depths: [2, 2, 2, 2]
  num_heads: [3, 6, 12, 24]
  window_size: 4
  patch_size: 2
  use_checkpoint: false
  use_v2: true

training:
  epochs: 100
  batch_size: 12
  eval_batch_size: 12
  swin_batch_size: 5
  swin_eval_batch_size: 6
  learning_rate: 0.0001
  weight_decay: 0.001
  k_folds: 5
  eval_interval: 5
  save_interval: 10
  seed: 42
  loss: <LOSS>                  # cross_entropy | auto
  clip_grad_norm: 1.0
  amp: false
  track_train_metrics: false
  class_weight_mode: none

early_stopping:
  enabled: false
  patience: 6
  min_delta: 0.005

gradcam:
  enabled: false
  max_samples: 8
  target_layer: image_encoder.attention_layers
  target_class: 0

data:
  target_mode: <TARGET_MODE>
  source_dir: <SOURCE_DIR>
  labels_json: ../patient_angle_classification_by_group.json
  pft_json: ../pft.json         # 迴歸/OI 任務需要;正異常可省略
  oi_json: ./oi_processed.json  # 僅 oi_emphysema 需要
  manifest: <MANIFEST_OUT>
  tapct_features: <TAPCT>       # null | ./embeddings/tapct_s_3d/features.npz
  image_size: [112, 136, 112]
  intensity_window: [-1000.0, 400.0]
  input_normalization: zscore
  target_normalization: <TGT_NORM>   # none | zscore
  cache_data: true
  num_workers: 4
  pin_memory: true
  prefetch_factor: 4
  angle_bin_count: 5
  balanced_sampling: true
  augmentation:
    enabled: true
    balance_then_augment: true
    views_per_sample: 5
    probability: 1.0
    class_indices: <CLASS_INDICES>   # 分類填 [0,1] 或 [0,1,2];迴歸省略
    rotation_degrees: 7.0
    translation_fraction: 0.05
    scale_range: [0.95, 1.05]
    intensity_scale_range: [0.95, 1.05]
    intensity_shift_range: [-0.1, 0.1]
    noise_std: 0.03

paths:
  weights: ./weights
  logs: ./train_log
  figures: ./figures
  graphs: ./graphs

task: <TASK>

resume:
  enabled: false
  uuid: null
  start_fold: 0

gpu:
  device_id: "0"
```

---

## Task 1: 建立 66 病人 normal_v_abnormal 臨床分組資料夾

RQ1 的 loader 由**父資料夾名**(`startswith("abnormal"/"normal")`)決定 label。`by_angle_all` 的資料夾是**角度**分組(35/31),不是臨床 33/33,所以必須另建一個 `Abnormal/`+`Normal/` 資料夾,內容為 66 個 CT 的 symlink,依 `patient_angle_classification_by_group.json` 的臨床組別歸類。

**Files:**
- Create: `regression/scripts/build_nva66.py`
- Output dir: `classification/datasets/normal_v_abnormal_66/{Abnormal,Normal}/`

**Interfaces:**
- Consumes: `../patient_angle_classification_by_group.json`(keys `abnormal_group_33`, `normal_group_21`,各含 `by_angle.{low_angle,high_angle}` 的 `patient_id→angle`)、`../by_angle_all/**/*.nii.gz`。
- Produces: `normal_v_abnormal_66/` 供 Task 2 的 config `source_dir` 使用。

- [ ] **Step 1: 寫建置腳本**

```python
# regression/scripts/build_nva66.py
"""Build a 66-patient Normal/Abnormal dir (clinical grouping) from by_angle_all."""
from __future__ import annotations
import json
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
LABELS = REG / ".." / "patient_angle_classification_by_group.json"
SRC = REG / ".." / "by_angle_all"
OUT = REG / ".." / "classification" / "datasets" / "normal_v_abnormal_66"


def patient_ids_for(group_block: dict) -> set[str]:
    ids: set[str] = set()
    for angle_side in group_block.get("by_angle", {}).values():
        ids.update(str(pid) for pid in angle_side.keys())
    return ids


def ct_files_by_pid() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in SRC.rglob("*.nii.gz"):
        pid = path.name.split("_", 1)[0].split(" ", 1)[0]
        mapping.setdefault(pid, path)
    return mapping


def main() -> None:
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    abnormal_ids = patient_ids_for(data["abnormal_group_33"])
    normal_ids = patient_ids_for(data["normal_group_21"])
    files = ct_files_by_pid()

    for cls, ids in (("Abnormal", abnormal_ids), ("Normal", normal_ids)):
        dst_dir = OUT / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for pid in sorted(ids):
            src = files.get(pid)
            if src is None:
                raise FileNotFoundError(f"No CT for patient {pid} in {SRC}")
            link = dst_dir / src.name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src.resolve())
    print(f"Abnormal={len(abnormal_ids)} Normal={len(normal_ids)} -> {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 執行並驗證計數**

Run:
```bash
cd regression && conda run -n nnMamba python scripts/build_nva66.py
ls ../classification/datasets/normal_v_abnormal_66/Abnormal | wc -l
ls ../classification/datasets/normal_v_abnormal_66/Normal | wc -l
```
Expected: 印出 `Abnormal=33 Normal=33`,兩個 `wc -l` 各為 `33`。若任一 CT 找不到會丟 `FileNotFoundError`。

- [ ] **Step 3: 驗證 loader 能正確切 33/33 fold(不訓練)**

Run:
```bash
cd regression && conda run -n nnMamba python -c "
from data.loader import RegressionLoaderHelper as L
h = L(data_root='../classification/datasets/normal_v_abnormal_66',
      labels_json='../patient_angle_classification_by_group.json',
      target_mode='normal_v_abnormal', load_ct_data=False, cache_data=False)
import numpy as np
print('n=', len(h.records), 'counts=', np.bincount(h.targets))
"
```
Expected: `n= 66 counts= [33 33]`(index0=Abnormal, index1=Normal)。

- [ ] **Step 4: Commit**

```bash
git add regression/scripts/build_nva66.py
git commit -m "feat: build 66-patient clinical Normal/Abnormal dir from by_angle_all"
```

---

## Task 2: RQ1 config(normal_v_abnormal,image + fusion)

**Files:**
- Create: `regression/config.rq1.normal_v_abnormal.image.yaml`
- Create: `regression/config.rq1.normal_v_abnormal.fusion.yaml`

**Interfaces:**
- Consumes: Task 1 的 `normal_v_abnormal_66/`;`embeddings/tapct_s_3d/features.npz`。
- Produces: 兩個可被 `train.py --config` 執行的 config,供 Task 7 驅動。

以協定模板填入,兩檔差異僅 `model.name` / `tapct_features` / manifest 輸出路徑。共同值:
`target_mode: normal_v_abnormal`、`source_dir: ../classification/datasets/normal_v_abnormal_66`、
`num_classes: 2`、`loss: cross_entropy`、`target_normalization: none`、`class_indices: [0,1]`、
`task: RQ1_normal_v_abnormal`。省略 `pft_json`/`oi_json`。

- image 檔:`model.name: hybrid_mamba_attention`、`data.tapct_features: null`、
  `manifest: ./datasets/generated/rq1_nva66_manifest.image.json`
- fusion 檔:`model.name: hybrid_mamba_tapct_fusion`、
  `data.tapct_features: ./embeddings/tapct_s_3d/features.npz`、`model.tapct_embedding_dim: 1152`、
  `manifest: ./datasets/generated/rq1_nva66_manifest.fusion.json`

- [ ] **Step 1: 建立兩個 config 檔**(依模板 + 上述欄位)

- [ ] **Step 2: 驗證兩檔都能被 Config 解析且 loader 建 66/33-33 fold**

Run:
```bash
cd regression && for c in config.rq1.normal_v_abnormal.image.yaml config.rq1.normal_v_abnormal.fusion.yaml; do
  conda run -n nnMamba python -c "
from core.config import Config
from data.loader import RegressionLoaderHelper as L
c=Config.from_yaml('$c'); h=L(c)
import numpy as np
print('$c','n=',len(h.records),'counts=',np.bincount(h.targets),'folds=',len(h.fold_indices))
"
done
```
Expected: 兩行都 `n= 66 counts= [33 33] folds= 5`;fusion 那行會印出 `TAP-CT embeddings: .../tapct_s_3d/features.npz | 66 patients ...`。

- [ ] **Step 3: Commit**

```bash
git add regression/config.rq1.normal_v_abnormal.image.yaml regression/config.rq1.normal_v_abnormal.fusion.yaml
git commit -m "feat: RQ1 aligned normal_v_abnormal image/fusion configs on 66 patients"
```

---

## Task 3: RQ2a config(angle_3class,image + fusion)

loader 由 `by_angle_all` + `labels_json` 依角度值切 3 類(≤131 / <152 / ≥152),涵蓋 66。

**Files:**
- Create: `regression/config.rq2a.angle_3class.image.yaml`
- Create: `regression/config.rq2a.angle_3class.fusion.yaml`

以模板填入。共同值:`target_mode: angle_3class`、`source_dir: ../by_angle_all`、
`pft_json: ../pft.json`、`num_classes: 3`、`loss: cross_entropy`、`target_normalization: none`、
`class_indices: [0,1,2]`、`task: RQ2a_angle_3class`。省略 `oi_json`。
image/fusion 差異同 Task 2;manifest 輸出各為
`./datasets/generated/rq2a_angle3class_manifest.image.json` /
`....fusion.json`。

- [ ] **Step 1: 建立兩個 config 檔**
- [ ] **Step 2: 驗證解析 + fold**

Run:
```bash
cd regression && for c in config.rq2a.angle_3class.image.yaml config.rq2a.angle_3class.fusion.yaml; do
  conda run -n nnMamba python -c "
from core.config import Config
from data.loader import RegressionLoaderHelper as L
c=Config.from_yaml('$c'); h=L(c)
import numpy as np
print('$c','n=',len(h.records),'counts=',np.bincount(h.targets),'folds=',len(h.fold_indices))
"
done
```
Expected: 兩行 `n= 66`,`counts=` 為 3 類的分佈,`folds= 5`。

- [ ] **Step 3: Commit**

```bash
git add regression/config.rq2a.angle_3class.image.yaml regression/config.rq2a.angle_3class.fusion.yaml
git commit -m "feat: RQ2a aligned angle_3class image/fusion configs (66)"
```

---

## Task 4: RQ2b config(angle_binary_extreme,image + fusion)

extreme binary 依定義排除灰區 → 61 病人(既有 manifest 已證實 61)。這是任務本質,非缺陷。

**Files:**
- Create: `regression/config.rq2b.angle_binary.image.yaml`
- Create: `regression/config.rq2b.angle_binary.fusion.yaml`

共同值:`target_mode: angle_binary_extreme`、`source_dir: ../by_angle_all`、`pft_json: ../pft.json`、
`num_classes: 2`、`loss: cross_entropy`、`target_normalization: none`、`class_indices: [0,1]`、
`task: RQ2b_angle_binary_extreme`。manifest 輸出
`./datasets/generated/rq2b_anglebin_manifest.{image,fusion}.json`。image/fusion 差異同前。

- [ ] **Step 1: 建立兩個 config 檔**
- [ ] **Step 2: 驗證解析 + fold(預期 61)**

Run:
```bash
cd regression && for c in config.rq2b.angle_binary.image.yaml config.rq2b.angle_binary.fusion.yaml; do
  conda run -n nnMamba python -c "
from core.config import Config
from data.loader import RegressionLoaderHelper as L
c=Config.from_yaml('$c'); h=L(c)
import numpy as np
print('$c','n=',len(h.records),'counts=',np.bincount(h.targets),'folds=',len(h.fold_indices))
"
done
```
Expected: `n= 61`、2 類分佈、`folds= 5`。

- [ ] **Step 3: Commit**

```bash
git add regression/config.rq2b.angle_binary.image.yaml regression/config.rq2b.angle_binary.fusion.yaml
git commit -m "feat: RQ2b aligned angle_binary_extreme image/fusion configs (61, gray-zone excluded)"
```

---

## Task 5: RQ2c config(angle 迴歸,image + fusion)

角度數值迴歸,涵蓋 66。迴歸不套 classification augmentation(loader 的 `_build_train_augmentation` 只對分類啟用),故 aug 欄位保留但不生效——這是既有行為,兩模型一致即可。

**Files:**
- Create: `regression/config.rq2c.angle_reg.image.yaml`
- Create: `regression/config.rq2c.angle_reg.fusion.yaml`

共同值:`target_mode: angle`、`source_dir: ../by_angle_all`、`pft_json: ../pft.json`、
`num_classes: 1`、`loss: auto`、`target_normalization: zscore`、`task: RQ2c_angle_regression`。
省略 `class_indices`/`oi_json`。manifest 輸出
`./datasets/generated/rq2c_anglereg_manifest.{image,fusion}.json`。image/fusion 差異同前。

- [ ] **Step 1: 建立兩個 config 檔**
- [ ] **Step 2: 驗證解析 + fold(迴歸分層,預期 66)**

Run:
```bash
cd regression && for c in config.rq2c.angle_reg.image.yaml config.rq2c.angle_reg.fusion.yaml; do
  conda run -n nnMamba python -c "
from core.config import Config
from data.loader import RegressionLoaderHelper as L
c=Config.from_yaml('$c'); h=L(c)
print('$c','n=',len(h.records),'strategy=',h.split_strategy,'folds=',len(h.fold_indices))
"
done
```
Expected: `n= 66`、`folds= 5`(迴歸走 stratified_bins 或 kfold)。

- [ ] **Step 3: Commit**

```bash
git add regression/config.rq2c.angle_reg.image.yaml regression/config.rq2c.angle_reg.fusion.yaml
git commit -m "feat: RQ2c aligned angle regression image/fusion configs (66)"
```

---

## Task 6: RQ3 config(oi_emphysema,image + fusion)

OI 二分類(emphysema 與否),label 由 `oi_json` + threshold 決定,涵蓋 66。

**Files:**
- Create: `regression/config.rq3.oi_emphysema.image.yaml`
- Create: `regression/config.rq3.oi_emphysema.fusion.yaml`

共同值:`target_mode: oi_emphysema`、`source_dir: ../by_angle_all`、`pft_json: ../pft.json`、
`oi_json: ./oi_processed.json`、`num_classes: 2`、`loss: cross_entropy`、
`target_normalization: none`、`class_indices: [0,1]`、`task: RQ3_oi_emphysema`。manifest 輸出
`./datasets/generated/rq3_oiemph_manifest.{image,fusion}.json`。image/fusion 差異同前。

> 注意:image-only 的 oi_emphysema 是**全新**(先前只有 fusion)。沿用既有 fusion 的 OI 門檻設定
> (預設 loader `oi_threshold=4.38`),兩模型一致即可。

- [ ] **Step 1: 建立兩個 config 檔**
- [ ] **Step 2: 驗證解析 + fold(預期 66)**

Run:
```bash
cd regression && for c in config.rq3.oi_emphysema.image.yaml config.rq3.oi_emphysema.fusion.yaml; do
  conda run -n nnMamba python -c "
from core.config import Config
from data.loader import RegressionLoaderHelper as L
c=Config.from_yaml('$c'); h=L(c)
import numpy as np
print('$c','n=',len(h.records),'counts=',np.bincount(h.targets),'classes=',h.class_names)
"
done
```
Expected: `n= 66`、2 類分佈、印出兩個 emphysema 類別名。

- [ ] **Step 3: Commit**

```bash
git add regression/config.rq3.oi_emphysema.image.yaml regression/config.rq3.oi_emphysema.fusion.yaml
git commit -m "feat: RQ3 aligned oi_emphysema image/fusion configs incl. new image-only arm (66)"
```

---

## Task 7: 一鍵驅動腳本 run_all_rq.py

依序跑 10 個 config,每個獨立 log、記錄 UUID、可續跑(輸出已存在則跳過)、失敗不中斷、最後印總結。

**Files:**
- Create: `regression/scripts/run_all_rq.py`

**Interfaces:**
- Consumes: Task 2–6 的 10 個 config;`train.py`。
- Produces: `regression/train_log/run_all_rq/<config>.log` 與 `regression/train_log/run_all_rq/summary.json`。

- [ ] **Step 1: 寫驅動腳本**

```python
# regression/scripts/run_all_rq.py
"""Run all RQ1/2/3 aligned configs sequentially with per-config logs and resume."""
from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
LOG_DIR = REG / "train_log" / "run_all_rq"

CONFIGS = [
    "config.rq1.normal_v_abnormal.image.yaml",
    "config.rq1.normal_v_abnormal.fusion.yaml",
    "config.rq2a.angle_3class.image.yaml",
    "config.rq2a.angle_3class.fusion.yaml",
    "config.rq2b.angle_binary.image.yaml",
    "config.rq2b.angle_binary.fusion.yaml",
    "config.rq2c.angle_reg.image.yaml",
    "config.rq2c.angle_reg.fusion.yaml",
    "config.rq3.oi_emphysema.image.yaml",
    "config.rq3.oi_emphysema.fusion.yaml",
]


def load_summary() -> dict:
    path = LOG_DIR / "summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_summary(summary: dict) -> None:
    (LOG_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    for config in CONFIGS:
        if summary.get(config, {}).get("status") == "done":
            print(f"[skip] {config} already done")
            continue
        log_path = LOG_DIR / f"{Path(config).stem}.log"
        print(f"[run ] {config} -> {log_path}")
        started = datetime.now().isoformat(timespec="seconds")
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [sys.executable, "train.py", "--config", config],
                cwd=REG, stdout=log, stderr=subprocess.STDOUT, text=True,
            )
        uuid = None
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if "Run UUID:" in line or "Weights saved to:" in line:
                uuid = line.strip().split()[-1]
        summary[config] = {
            "status": "done" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "uuid": uuid,
            "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "log": str(log_path),
        }
        save_summary(summary)
        print(f"[{'ok  ' if proc.returncode == 0 else 'FAIL'}] {config} rc={proc.returncode}")

    print("\n=== SUMMARY ===")
    for config in CONFIGS:
        info = summary.get(config, {})
        print(f"  {info.get('status','pending'):8} {config}  uuid={info.get('uuid')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 乾跑檢查(不真的訓練)——確認清單與 skip 邏輯**

Run:
```bash
cd regression && conda run -n nnMamba python -c "
import importlib.util, pathlib
spec=importlib.util.spec_from_file_location('r','scripts/run_all_rq.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
missing=[c for c in m.CONFIGS if not pathlib.Path(c).exists()]
print('configs=',len(m.CONFIGS),'missing=',missing)
"
```
Expected: `configs= 10 missing= []`。

- [ ] **Step 3: 用一個 config 做真實 smoke(確認入口可跑)**

先臨時把 `config.rq1.normal_v_abnormal.image.yaml` 複製為 `config.rq1.smoke.yaml` 並把 `epochs` 改 2、`k_folds` 改 2,執行:
```bash
cd regression && conda run -n nnMamba python train.py --config config.rq1.smoke.yaml 2>&1 | tail -5
```
Expected: 印出 `Training complete. Run UUID: ...` 與 `Weights saved to: ...`。確認後刪除 `config.rq1.smoke.yaml`。

- [ ] **Step 4: Commit**

```bash
git add regression/scripts/run_all_rq.py
git commit -m "feat: add run_all_rq.py driver for the 10 aligned RQ configs"
```

---

## Task 8: RQ 組織文件 rq_overview.md

**Files:**
- Create: `regression/docs/rq_overview.md`

- [ ] **Step 1: 寫文件**,包含:
  - 三個 RQ 敘述(RQ1 正異常、RQ2 塌陷角含 3 任務、RQ3 OI 二分類)。
  - 2 模型 × 任務矩陣表(對應 10 個 config 檔名)。
  - 66 病人統一世代說明 + RQ2b=61(排除灰區)、fusion 用 tapct_s_3d(1152)、merlin 淘汰。
  - 統一協定摘要(100ep、無早停、aug 5x、seed 42、5-fold、argmax)。
  - **結果表骨架**(每個 RQ/任務一列,欄位 image Acc/AUC、fusion Acc/AUC),值先填 `TBD`,由 Task 9 跑完後回填。
  - 指向 spec:`docs/superpowers/specs/2026-07-14-rq123-two-model-matrix-design.md`。

- [ ] **Step 2: Commit**

```bash
git add regression/docs/rq_overview.md
git commit -m "docs: RQ1/2/3 two-model matrix overview with results skeleton"
```

---

## Task 9: 背景跑完 10 個實驗並回填結果

- [ ] **Step 1: 背景啟動驅動腳本**

```bash
cd regression && conda run -n nnMamba python scripts/run_all_rq.py
```
(以背景執行;完成後 `train_log/run_all_rq/summary.json` 會有 10 筆 `status: done` 與各自 UUID。)

- [ ] **Step 2: 監看進度**

Run:
```bash
cat regression/train_log/run_all_rq/summary.json
```
Expected: 逐步出現各 config 的 `done` 與 UUID;失敗者為 `failed` 但不中斷其餘。

- [ ] **Step 3: 回填 `rq_overview.md` 結果表**,由每個 UUID 的結果檔
  (`regression/figures/<task>/<uuid>/...` 的 metrics/`results.json`)取 Acc/AUC 填入,取代 `TBD`。

- [ ] **Step 4: Commit**

```bash
git add regression/docs/rq_overview.md regression/train_log/run_all_rq/summary.json
git commit -m "docs: backfill RQ1/2/3 results from completed runs"
```

---

## Self-Review 檢查結果

- **Spec coverage:** 10 個 config(Task 2–6)、66 世代含 RQ1 新建(Task 1)、tapct_s_3d fusion(各 config)、統一協定(Global Constraints + 模板)、驅動腳本(Task 7)、rq_overview 文件(Task 8)、跑完回填(Task 9)——spec 各節都有對應 Task。
- **Placeholder scan:** 結果表的 `TBD` 是刻意佔位(等訓練結果),非計畫缺漏;其餘皆具體指令/程式碼。
- **Type/命名一致:** config 檔名、manifest 輸出路徑、`run_all_rq.py` 的 `CONFIGS` 清單三處檔名一致(10 個)。
- **已知例外**已寫入:RQ2b=61(排除灰區)、迴歸 aug 欄位不生效(既有行為)。
