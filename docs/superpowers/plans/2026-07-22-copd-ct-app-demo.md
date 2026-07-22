# COPD CT App — Gradio Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the self-contained hospital-facing `copd-ct-app` repo: a Gradio demo that takes a DICOM series, runs the 5-member Normal-vs-Abnormal ensemble, shows the probability, and captures the CT + patient number into a staging area.

**Architecture:** A thin Gradio shell (`app.py`) over a UI-agnostic core. The single public entry `predict_and_capture(dicom_dir)` orchestrates DICOM→NIfTI → frozen preprocess → 5-member soft-vote ensemble → staging capture, returning a pure-data `PredictionResult`. The repo is self-contained (no nnMamba import): the model architecture and the preprocessing are vendored copies. The preprocessing is the frozen data contract, verified bit-for-bit against nnMamba's training preprocessing by a consistency test.

**Tech Stack:** Python 3.10+, PyTorch (cu124), mamba-ssm (CUDA-only), SimpleITK (DICOM), nibabel + scikit-image (preprocess), Gradio (UI), Docker (CUDA base) for delivery.

## Global Constraints

- **NVIDIA GPU mandatory.** mamba-ssm's selective-scan kernel is CUDA-only; there is no CPU path. All model code runs on `cuda`.
- **Self-contained repo.** No `import` from nnMamba / `regression`. Model architecture and preprocessing are vendored copies inside `copd-ct-app`.
- **Preprocessing is the frozen data contract.** `core/preprocess.py` must stay bit-for-bit identical to nnMamba `regression/data/dataset.py` `load_ct`: clip → resize `(112, 136, 112)` order=1, preserve_range, anti_aliasing → z-score (1/99 percentile clip, mean/std). Any change requires re-verifying against training.
- **Real patient number as filename** (de-identification is an off-by-default toggle, interface only).
- **Inference and capture are decoupled.** A capture failure must never prevent returning a prediction.
- **Tests use the project's inline runner convention, not pytest.** Each test file ends with `if __name__ == "__main__":` calling its test functions. Run with the `nnMamba` conda env: `conda activate nnMamba && python <test_file>`.
- **Checkpoint format:** `torch.load` returns a dict; weights live under `ckpt["state_dict"]`.
- **Model build kwargs for `hybrid_mamba_attention`:** `in_channels=1, num_classes=2, base_channels=32, depths=(3,3,3), head_hidden_dim=256, dropout=0.3, attn_heads=8, attn_layers=1, attn_mlp_ratio=2.0, attn_dropout=0.1` (from `config.normal_v_abnormal.imageonly.aug5.ensemble.yaml`).
- **Image size** `(112, 136, 112)`, **intensity window** `(-1000.0, 400.0)`, **input_normalization** `"zscore"`.

**Repo location:** `~/Research/copd-ct-app/` (new git repo, separate from nnMamba).

**Scope note:** This plan is the hospital demo app only. The research-side scripts (`train_production_ensemble.py`, `label_backfill.py`, `package_release.py`) are a separate follow-up plan. For this plan's tests, checkpoints are synthetic (randomly-initialized, saved in the real format). Running the real demo needs 5 real checkpoints placed in `models/current/`, produced by the follow-up plan or copied from an existing ensemble run.

---

## File Structure

```
copd-ct-app/
├── app.py                       # Gradio shell (Task 9)
├── config.yaml                  # paths + toggles (Task 1)
├── core/
│   ├── __init__.py
│   ├── config.py                # load config.yaml -> AppConfig (Task 1)
│   ├── preprocess.py            # FROZEN load_ct (Task 2)
│   ├── model.py                 # vendored HybridMambaAttentionRegressor + build (Task 3)
│   ├── ensemble.py              # load 5 checkpoints, soft-vote (Task 4)
│   ├── dicom_io.py              # DICOM series -> NIfTI (Task 5)
│   ├── staging.py               # capture writes + capture_log (Task 6)
│   ├── gradcam.py               # Grad-CAM heatmap (Task 8)
│   └── api.py                   # predict_and_capture + PredictionResult (Task 7)
├── models/current/              # 5 checkpoints + metrics.json (runtime, git-ignored)
├── staging/                     # incoming/ + capture_log.jsonl (runtime, git-ignored)
├── tests/
│   ├── test_preprocess_consistency.py
│   ├── test_model.py
│   ├── test_ensemble.py
│   ├── test_dicom_io.py
│   ├── test_staging.py
│   └── test_api.py
├── environment.yml              # dev env (Task 10)
├── Dockerfile                   # CUDA base (Task 10)
├── .gitignore
└── README.md                    # (Task 10)
```

---

### Task 1: Repo scaffold + config loader ✅ DONE

**Files:**
- Create: `~/Research/copd-ct-app/.gitignore`
- Create: `~/Research/copd-ct-app/config.yaml`
- Create: `~/Research/copd-ct-app/core/__init__.py` (empty)
- Create: `~/Research/copd-ct-app/core/config.py`
- Test: `~/Research/copd-ct-app/tests/test_config.py`

**Interfaces:**
- Produces: `AppConfig` dataclass with fields `models_dir: Path`, `staging_dir: Path`, `image_size: tuple[int,int,int]`, `intensity_window: tuple[float,float]`, `input_normalization: str`, `num_classes: int`, `class_names: list[str]`, `show_disclaimer: bool`, `deidentify: bool`, `gradcam_enabled: bool`; and `load_config(path: str | Path) -> AppConfig`.

- [ ] **Step 1: Initialize the repo**

```bash
mkdir -p ~/Research/copd-ct-app/core ~/Research/copd-ct-app/tests
cd ~/Research/copd-ct-app && git init
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
models/
staging/
*.nii.gz
.DS_Store
```

- [ ] **Step 3: Write `config.yaml`**

```yaml
# copd-ct-app runtime config
models_dir: ./models/current
staging_dir: ./staging
image_size: [112, 136, 112]
intensity_window: [-1000.0, 400.0]
input_normalization: zscore
num_classes: 2
class_names: [Abnormal, Normal]   # class_index 0 = Abnormal, 1 = Normal (matches training manifest)
show_disclaimer: true
deidentify: false
gradcam_enabled: true
```

- [ ] **Step 4: Write the failing test** — `tests/test_config.py`

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config


def test_load_config_defaults():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg.image_size == (112, 136, 112)
    assert cfg.intensity_window == (-1000.0, 400.0)
    assert cfg.input_normalization == "zscore"
    assert cfg.num_classes == 2
    assert cfg.class_names == ["Abnormal", "Normal"]
    assert cfg.show_disclaimer is True
    assert cfg.deidentify is False
    assert cfg.gradcam_enabled is True
    print("test_load_config_defaults PASS")


if __name__ == "__main__":
    test_load_config_defaults()
```

- [ ] **Step 5: Run test to verify it fails**

Run: `conda activate nnMamba && cd ~/Research/copd-ct-app && python tests/test_config.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.config'`

- [ ] **Step 6: Write `core/config.py`**

```python
"""Load and validate copd-ct-app runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AppConfig:
    models_dir: Path
    staging_dir: Path
    image_size: tuple[int, int, int]
    intensity_window: tuple[float, float]
    input_normalization: str
    num_classes: int
    class_names: list[str]
    show_disclaimer: bool
    deidentify: bool
    gradcam_enabled: bool


def load_config(path: str | Path) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text())
    return AppConfig(
        models_dir=Path(data["models_dir"]),
        staging_dir=Path(data["staging_dir"]),
        image_size=tuple(int(v) for v in data["image_size"]),
        intensity_window=(
            float(data["intensity_window"][0]),
            float(data["intensity_window"][1]),
        ),
        input_normalization=str(data["input_normalization"]),
        num_classes=int(data["num_classes"]),
        class_names=list(data["class_names"]),
        show_disclaimer=bool(data["show_disclaimer"]),
        deidentify=bool(data["deidentify"]),
        gradcam_enabled=bool(data["gradcam_enabled"]),
    )
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd ~/Research/copd-ct-app && python tests/test_config.py`
Expected: `test_load_config_defaults PASS`

- [ ] **Step 8: Commit**

```bash
cd ~/Research/copd-ct-app
git add .gitignore config.yaml core/__init__.py core/config.py tests/test_config.py
git commit -m "feat: repo scaffold and config loader"
```

---

### Task 2: Frozen preprocessing + consistency test ✅ DONE

**Files:**
- Create: `~/Research/copd-ct-app/core/preprocess.py`
- Test: `~/Research/copd-ct-app/tests/test_preprocess_consistency.py`

**Interfaces:**
- Produces: `load_ct(path, image_size, intensity_window=None, input_normalization="zscore") -> np.ndarray` returning a `(1, D, H, W)` float32 array; `PREPROCESS_HASH: str` (sha256 of the frozen source of the three functions).

- [ ] **Step 1: Write `core/preprocess.py`** (verbatim copy of the nnMamba `load_ct` pipeline — this is the frozen data contract)

```python
"""FROZEN CT preprocessing — bit-for-bit copy of nnMamba regression/data/dataset.py.

Do not edit without re-verifying against training via package_release.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from skimage import transform

InputNormalization = Literal["zscore", "none"]


def _resize_volume(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if volume.shape == target_shape:
        return volume
    return transform.resize(
        volume,
        target_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )


def _normalize_volume(
    volume: np.ndarray,
    intensity_window: tuple[float, float] | None = None,
    input_normalization: InputNormalization = "zscore",
) -> np.ndarray:
    if intensity_window is not None:
        lo, hi = intensity_window
        volume = np.clip(volume, lo, hi)
    if input_normalization == "none":
        return volume
    if input_normalization != "zscore":
        raise ValueError(f"Unsupported input_normalization: {input_normalization}")
    lo, hi = np.percentile(volume, [1, 99])
    if hi > lo:
        volume = np.clip(volume, lo, hi)
    mean = float(volume.mean())
    std = float(volume.std())
    if std < 1e-6:
        std = 1.0
    return (volume - mean) / std


def load_ct(
    path: str | Path,
    image_size: tuple[int, int, int],
    intensity_window: tuple[float, float] | None = None,
    input_normalization: InputNormalization = "zscore",
) -> np.ndarray:
    path = Path(path)
    volume = nib.load(str(path)).get_fdata().astype(np.float32)
    if volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={volume.shape} for {path}")
    volume = _resize_volume(volume, image_size).astype(np.float32)
    volume = _normalize_volume(
        volume,
        intensity_window=intensity_window,
        input_normalization=input_normalization,
    )
    return np.expand_dims(volume, axis=0).astype(np.float32)


def _compute_hash() -> str:
    import inspect

    src = "".join(
        inspect.getsource(fn) for fn in (_resize_volume, _normalize_volume, load_ct)
    )
    return hashlib.sha256(src.encode()).hexdigest()


PREPROCESS_HASH = _compute_hash()
```

- [ ] **Step 2: Write the failing consistency test** — `tests/test_preprocess_consistency.py`

This test proves the frozen copy produces identical output to nnMamba's training `load_ct` on a random volume. It imports the training function directly (dev machine only; the hospital never runs this).

```python
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.preprocess import load_ct as frozen_load_ct

# Import the training reference from nnMamba (dev machine has it as a sibling repo).
NNMAMBA_REGRESSION = Path.home() / "Research" / "nnMamba" / "regression"
sys.path.insert(0, str(NNMAMBA_REGRESSION))
from data.dataset import load_ct as training_load_ct  # noqa: E402


def _write_random_nifti(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    vol = (rng.random((90, 100, 80)).astype(np.float32) * 1400.0) - 1000.0
    p = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), str(p))
    return p


def test_frozen_matches_training():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = _write_random_nifti(Path(d))
        kw = dict(intensity_window=(-1000.0, 400.0), input_normalization="zscore")
        a = frozen_load_ct(p, (112, 136, 112), **kw)
        b = training_load_ct(p, (112, 136, 112), **kw)
        assert a.shape == (1, 112, 136, 112)
        assert np.array_equal(a, b), "frozen preprocess drifted from training"
    print("test_frozen_matches_training PASS")


if __name__ == "__main__":
    test_frozen_matches_training()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/Research/copd-ct-app && python tests/test_preprocess_consistency.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.preprocess'` (before Step 1 is saved) or an import error. If Step 1 is already saved, temporarily rename a constant to force a mismatch, confirm it fails, then restore.

- [ ] **Step 4: Confirm implementation is correct**

`core/preprocess.py` from Step 1 is the implementation; no further code needed.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/Research/copd-ct-app && python tests/test_preprocess_consistency.py`
Expected: `test_frozen_matches_training PASS`

- [ ] **Step 6: Commit**

```bash
cd ~/Research/copd-ct-app
git add core/preprocess.py tests/test_preprocess_consistency.py
git commit -m "feat: frozen preprocessing with training-consistency test"
```

---

### Task 3: Vendored model architecture + builder ✅ DONE

**Files:**
- Create: `~/Research/copd-ct-app/core/model.py`
- Test: `~/Research/copd-ct-app/tests/test_model.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `build_ensemble_member(num_classes: int = 2) -> torch.nn.Module` returning a `HybridMambaAttentionRegressor` on CPU with the locked hyperparameters; the class `HybridMambaAttentionRegressor` with attribute `attention_layers` (Grad-CAM target) and `forward(x: Tensor[B,1,D,H,W]) -> Tensor[B,num_classes]`.

- [ ] **Step 1: Write `core/model.py`** (merge of nnMamba `mamba_regressor.py` blocks + `hybrid_mamba_attention_regressor.py`, trimmed to what the hybrid model needs)

```python
"""Vendored HybridMambaAttentionRegressor (from nnMamba regression networks).

Architecture only — stable. Requires mamba-ssm (CUDA).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba


def conv3x3(in_c: int, out_c: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_c: int, out_c: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, bias=False)


def norm3d(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ResidualMambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 8, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.pre_norm = norm3d(dim)
        self.pre_proj = conv1x1(dim, dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.post_norm = norm3d(dim)
        self.post_proj = conv1x1(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_proj(self.pre_norm(x))
        b, c = x.shape[:2]
        spatial = x.shape[2:]
        tokens = x.reshape(b, c, -1).transpose(1, 2)
        tokens = self.mamba(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, *spatial)
        x = self.post_proj(self.post_norm(x))
        return self.act(x + residual)


class DownsampleStage(nn.Module):
    def __init__(self, in_c: int, out_c: int, depth: int, stride: int = 2):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Sequential(conv3x3(in_c, out_c, stride=stride), norm3d(out_c), nn.GELU())
        ]
        for _ in range(depth):
            layers.append(ResidualMambaBlock(out_c))
        self.stage = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


def _resolve_attention_heads(dim: int, requested: int) -> int:
    heads = max(1, min(int(requested), int(dim)))
    while heads > 1 and dim % heads != 0:
        heads -= 1
    return heads


class HybridAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden = max(dim, int(dim * mlp_ratio))
        self.pre_norm = norm3d(dim)
        self.pos_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.token_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=_resolve_attention_heads(dim, num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(float(dropout))
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(hidden, dim), nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        x = x + self.pos_conv(x)
        b, c = x.shape[:2]
        spatial = x.shape[2:]
        tokens = x.reshape(b, c, -1).transpose(1, 2)
        attn_input = self.token_norm(tokens)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + self.attn_dropout(attn_out)
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        return residual + tokens.transpose(1, 2).reshape(b, c, *spatial)


class HybridMambaAttentionRegressor(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        depths: tuple[int, int, int] = (1, 1, 1),
        head_hidden_dim: int = 128,
        dropout: float = 0.2,
        attn_heads: int = 8,
        attn_layers: int = 1,
        attn_mlp_ratio: float = 2.0,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=4, padding=3, bias=False),
            norm3d(base_channels), nn.GELU(),
        )
        self.stage1 = DownsampleStage(base_channels, base_channels, depths[0], stride=1)
        self.stage2 = DownsampleStage(base_channels, base_channels * 2, depths[1], stride=2)
        self.stage3 = DownsampleStage(base_channels * 2, base_channels * 4, depths[2], stride=2)
        self.attention_layers = nn.Sequential(
            *[
                HybridAttentionBlock(
                    dim=base_channels * 4, num_heads=attn_heads,
                    mlp_ratio=attn_mlp_ratio, dropout=attn_dropout,
                )
                for _ in range(max(1, int(attn_layers)))
            ]
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        feature_dim = base_channels + base_channels * 2 + base_channels * 4 * 2
        head_hidden_dim = max(int(head_hidden_dim), feature_dim // 2, base_channels * 4)
        head_mid_dim = max(head_hidden_dim // 2, base_channels * 4)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, head_hidden_dim), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(head_hidden_dim, head_mid_dim), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(head_mid_dim, int(num_classes)),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stage1(self.stem(x))
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.attention_layers(x3)
        f1 = self.pool(x1).flatten(1)
        f2 = self.pool(x2).flatten(1)
        f3 = self.pool(x3).flatten(1)
        f4 = self.pool(x4).flatten(1)
        return torch.cat([f1, f2, f3, f4], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        output = self.head(features)
        return output.squeeze(-1) if output.shape[-1] == 1 else output


def build_ensemble_member(num_classes: int = 2) -> HybridMambaAttentionRegressor:
    """Build one ensemble member with the locked Normal-vs-Abnormal hyperparameters."""
    return HybridMambaAttentionRegressor(
        in_channels=1,
        num_classes=num_classes,
        base_channels=32,
        depths=(3, 3, 3),
        head_hidden_dim=256,
        dropout=0.3,
        attn_heads=8,
        attn_layers=1,
        attn_mlp_ratio=2.0,
        attn_dropout=0.1,
    )
```

- [ ] **Step 2: Write the failing test** — `tests/test_model.py`

```python
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.model import build_ensemble_member


def test_forward_shape_and_params():
    model = build_ensemble_member(num_classes=2).cuda().eval()
    n = sum(p.numel() for p in model.parameters())
    assert 1.0e6 < n < 1.3e6, f"unexpected param count {n}"
    x = torch.randn(2, 1, 112, 136, 112, device="cuda")
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)
    assert hasattr(model, "attention_layers")
    print(f"test_forward_shape_and_params PASS ({n/1e6:.2f}M params)")


if __name__ == "__main__":
    test_forward_shape_and_params()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd ~/Research/copd-ct-app && python tests/test_model.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.model'`

- [ ] **Step 4: (implementation is Step 1) Run test to verify it passes**

Run: `cd ~/Research/copd-ct-app && python tests/test_model.py`
Expected: `test_forward_shape_and_params PASS (1.15M params)`

- [ ] **Step 5: Commit**

```bash
cd ~/Research/copd-ct-app
git add core/model.py tests/test_model.py
git commit -m "feat: vendored hybrid mamba-attention model + builder"
```

---

### Task 4: Ensemble loader + soft-vote ✅ DONE

**Files:**
- Create: `~/Research/copd-ct-app/core/ensemble.py`
- Test: `~/Research/copd-ct-app/tests/test_ensemble.py`

**Interfaces:**
- Consumes: `build_ensemble_member` (Task 3); `load_ct` (Task 2).
- Produces: `class Ensemble` with `Ensemble.from_dir(models_dir: Path, num_classes: int = 2, device: str = "cuda") -> Ensemble` (loads every `*.pth` under `models_dir`, requires ≥1) and `predict_proba(volume: np.ndarray) -> np.ndarray` taking a `(1, D, H, W)` array, returning a `(num_classes,)` mean-softmax probability vector.

- [ ] **Step 1: Write the failing test** — `tests/test_ensemble.py`

```python
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.model import build_ensemble_member
from core.ensemble import Ensemble


def _make_fake_checkpoints(dir_: Path, n: int = 3) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        torch.manual_seed(i)
        m = build_ensemble_member(num_classes=2)
        torch.save({"state_dict": m.state_dict(), "fold": i}, dir_ / f"member_{i}.pth")


def test_predict_proba_shape_and_normalized():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        mdir = Path(d) / "current"
        _make_fake_checkpoints(mdir, n=3)
        ens = Ensemble.from_dir(mdir, num_classes=2, device="cuda")
        vol = np.random.randn(1, 112, 136, 112).astype(np.float32)
        proba = ens.predict_proba(vol)
        assert proba.shape == (2,)
        assert abs(float(proba.sum()) - 1.0) < 1e-4
        assert ens.num_members == 3
    print("test_predict_proba_shape_and_normalized PASS")


if __name__ == "__main__":
    test_predict_proba_shape_and_normalized()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Research/copd-ct-app && python tests/test_ensemble.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.ensemble'`

- [ ] **Step 3: Write `core/ensemble.py`**

```python
"""Load the 5-member ensemble and soft-vote."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.model import build_ensemble_member


class Ensemble:
    def __init__(self, members: list[torch.nn.Module], device: str):
        if not members:
            raise ValueError("Ensemble requires at least one member checkpoint.")
        self.members = members
        self.device = device

    @property
    def num_members(self) -> int:
        return len(self.members)

    @classmethod
    def from_dir(cls, models_dir: Path, num_classes: int = 2, device: str = "cuda") -> "Ensemble":
        models_dir = Path(models_dir)
        ckpts = sorted(models_dir.glob("*.pth"))
        if not ckpts:
            raise FileNotFoundError(f"No .pth checkpoints found in {models_dir}")
        members = []
        for ckpt_path in ckpts:
            payload = torch.load(ckpt_path, map_location=device)
            state = payload["state_dict"] if "state_dict" in payload else payload
            model = build_ensemble_member(num_classes=num_classes).to(device).eval()
            model.load_state_dict(state)
            members.append(model)
        return cls(members, device)

    @torch.no_grad()
    def predict_proba(self, volume: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(volume)).float().unsqueeze(0).to(self.device)
        probs = []
        for model in self.members:
            logits = model(x)
            probs.append(F.softmax(logits, dim=1))
        mean = torch.stack(probs, dim=0).mean(dim=0)
        return mean.squeeze(0).cpu().numpy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Research/copd-ct-app && python tests/test_ensemble.py`
Expected: `test_predict_proba_shape_and_normalized PASS`

- [ ] **Step 5: Commit**

```bash
cd ~/Research/copd-ct-app
git add core/ensemble.py tests/test_ensemble.py
git commit -m "feat: ensemble loader with soft-vote"
```

---

### Task 5: DICOM series → NIfTI

**Files:**
- Create: `~/Research/copd-ct-app/core/dicom_io.py`
- Test: `~/Research/copd-ct-app/tests/test_dicom_io.py`

**Interfaces:**
- Produces: `dicom_series_to_nifti(dicom_dir: str | Path, out_path: str | Path) -> DicomResult` where `DicomResult` is a dataclass with `nifti_path: Path`, `patient_id: str`, `series_uid: str`, `num_slices: int`. Raises `DicomError` on: no DICOM found, multiple series (ambiguous), or unreadable data.

- [ ] **Step 1: Write the failing test** — `tests/test_dicom_io.py` (synthesizes a small single-series DICOM stack with SimpleITK)

```python
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.dicom_io import dicom_series_to_nifti, DicomError


def _write_dicom_series(dir_: Path, patient_id: str = "TEST123", n_slices: int = 8) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    series_uid = "1.2.826.0.1.3680043.2.1125.1.111111111111111111111111111111"
    for i in range(n_slices):
        arr = (np.random.rand(32, 32).astype(np.float32) * 400).astype(np.int16)
        img = sitk.GetImageFromArray(arr)
        img.SetMetaData("0010|0020", patient_id)          # PatientID
        img.SetMetaData("0020|000e", series_uid)          # SeriesInstanceUID
        img.SetMetaData("0020|0032", f"0\\0\\{i}")        # ImagePositionPatient
        img.SetMetaData("0008|0060", "CT")                # Modality
        img.SetMetaData("0028|0100", "16")                # BitsAllocated
        writer.SetFileName(str(dir_ / f"slice_{i:03d}.dcm"))
        writer.Execute(img)


def test_converts_single_series():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "dicom"
        _write_dicom_series(src, patient_id="TEST123", n_slices=8)
        out = Path(d) / "out.nii.gz"
        result = dicom_series_to_nifti(src, out)
        assert result.nifti_path.exists()
        assert result.patient_id == "TEST123"
        assert result.num_slices == 8
    print("test_converts_single_series PASS")


def test_empty_dir_raises():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        try:
            dicom_series_to_nifti(Path(d), Path(d) / "out.nii.gz")
        except DicomError:
            print("test_empty_dir_raises PASS")
            return
        raise AssertionError("expected DicomError for empty dir")


if __name__ == "__main__":
    test_converts_single_series()
    test_empty_dir_raises()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Research/copd-ct-app && python tests/test_dicom_io.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.dicom_io'`

- [ ] **Step 3: Write `core/dicom_io.py`**

```python
"""DICOM series -> NIfTI conversion with PHI/series handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk


class DicomError(Exception):
    """Raised when a DICOM folder cannot be converted unambiguously."""


@dataclass(frozen=True)
class DicomResult:
    nifti_path: Path
    patient_id: str
    series_uid: str
    num_slices: int


def dicom_series_to_nifti(dicom_dir: str | Path, out_path: str | Path) -> DicomResult:
    dicom_dir = Path(dicom_dir)
    out_path = Path(out_path)
    if not dicom_dir.is_dir():
        raise DicomError(f"Not a directory: {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise DicomError(f"No DICOM series found in {dicom_dir}")
    if len(series_ids) > 1:
        raise DicomError(
            f"Multiple DICOM series in {dicom_dir}: {series_ids}. "
            "Provide a folder with a single series."
        )

    series_uid = series_ids[0]
    files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_uid)
    if not files:
        raise DicomError(f"Series {series_uid} has no files.")
    reader.SetFileNames(files)
    try:
        image = reader.Execute()  # SimpleITK applies RescaleSlope/Intercept + orders slices
    except RuntimeError as exc:
        raise DicomError(f"Failed to read series: {exc}") from exc

    # PatientID from the first slice header.
    meta = sitk.ImageFileReader()
    meta.SetFileName(files[0])
    meta.LoadPrivateTagsOn()
    meta.ReadImageInformation()
    patient_id = meta.GetMetaData("0010|0020").strip() if meta.HasMetaDataKey("0010|0020") else "UNKNOWN"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out_path))
    return DicomResult(
        nifti_path=out_path,
        patient_id=patient_id or "UNKNOWN",
        series_uid=series_uid,
        num_slices=len(files),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Research/copd-ct-app && python tests/test_dicom_io.py`
Expected: `test_converts_single_series PASS` and `test_empty_dir_raises PASS`

- [ ] **Step 5: Commit**

```bash
cd ~/Research/copd-ct-app
git add core/dicom_io.py tests/test_dicom_io.py
git commit -m "feat: DICOM series to NIfTI conversion"
```

---

### Task 6: Staging (capture writes + log)

**Files:**
- Create: `~/Research/copd-ct-app/core/staging.py`
- Test: `~/Research/copd-ct-app/tests/test_staging.py`

**Interfaces:**
- Produces: `class Staging` with `__init__(self, staging_dir: Path, deidentify: bool = False)`, and `capture(self, nifti_path: Path, patient_id: str, series_uid: str, prediction: dict) -> str | None`. `capture` copies the NIfTI into `incoming/` as `{patient_id}_{timestamp}.nii.gz`, appends a JSON line to `capture_log.jsonl`, and returns the stored relative path (or `None` on failure — never raises). All filesystem access is inside this class so the backend is swappable.

- [ ] **Step 1: Write the failing test** — `tests/test_staging.py`

```python
import sys
import json
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.staging import Staging


def _tmp_nifti(p: Path) -> Path:
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), str(p))
    return p


def test_capture_writes_file_and_log():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = _tmp_nifti(d / "src.nii.gz")
        staging = Staging(d / "staging", deidentify=False)
        rel = staging.capture(src, "PID42", "1.2.3", {"prob_abnormal": 0.7, "pred": "Abnormal"})
        assert rel is not None
        assert (d / "staging" / rel).exists()
        assert rel.startswith("incoming/PID42_")
        log = (d / "staging" / "capture_log.jsonl").read_text().strip().splitlines()
        assert len(log) == 1
        rec = json.loads(log[0])
        assert rec["patient_id"] == "PID42"
        assert rec["label"] is None
        assert rec["prediction"]["pred"] == "Abnormal"
    print("test_capture_writes_file_and_log PASS")


def test_capture_missing_source_returns_none():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        staging = Staging(d / "staging", deidentify=False)
        rel = staging.capture(d / "nope.nii.gz", "PID1", "1.2", {"pred": "Normal"})
        assert rel is None  # failure must not raise
    print("test_capture_missing_source_returns_none PASS")


if __name__ == "__main__":
    test_capture_writes_file_and_log()
    test_capture_missing_source_returns_none()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Research/copd-ct-app && python tests/test_staging.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.staging'`

- [ ] **Step 3: Write `core/staging.py`**

```python
"""Staging: capture CT + metadata. All storage access is centralized here so the
backend (local FS now, object storage later) is swappable in one place."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path


class Staging:
    def __init__(self, staging_dir: Path, deidentify: bool = False):
        self.staging_dir = Path(staging_dir)
        self.incoming = self.staging_dir / "incoming"
        self.log_path = self.staging_dir / "capture_log.jsonl"
        self.deidentify = bool(deidentify)

    def _dest_name(self, patient_id: str, timestamp: str) -> str:
        if self.deidentify:
            # Interface kept; real de-id (study code + mapping table) is future work.
            raise NotImplementedError("de-identification not yet enabled")
        return f"{patient_id}_{timestamp}.nii.gz"

    def capture(self, nifti_path: Path, patient_id: str, series_uid: str, prediction: dict) -> str | None:
        try:
            nifti_path = Path(nifti_path)
            if not nifti_path.exists():
                raise FileNotFoundError(nifti_path)
            self.incoming.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dest_name = self._dest_name(patient_id, timestamp)
            dest = self.incoming / dest_name
            shutil.copyfile(nifti_path, dest)
            rel = f"incoming/{dest_name}"
            record = {
                "patient_id": patient_id,
                "nifti": rel,
                "captured_at": timestamp,
                "series_uid": series_uid,
                "prediction": prediction,
                "label": None,
            }
            with self.log_path.open("a") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            return rel
        except Exception as exc:  # capture must never break inference
            self._log_error(patient_id, str(exc))
            return None

    def _log_error(self, patient_id: str, message: str) -> None:
        try:
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as fh:
                fh.write(json.dumps({"patient_id": patient_id, "error": message}) + "\n")
        except Exception:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Research/copd-ct-app && python tests/test_staging.py`
Expected: both PASS lines.

- [ ] **Step 5: Commit**

```bash
cd ~/Research/copd-ct-app
git add core/staging.py tests/test_staging.py
git commit -m "feat: staging capture with append-only log"
```

---

### Task 7: Public API — predict_and_capture

**Files:**
- Create: `~/Research/copd-ct-app/core/api.py`
- Test: `~/Research/copd-ct-app/tests/test_api.py`

**Interfaces:**
- Consumes: `AppConfig`/`load_config` (Task 1), `load_ct` (Task 2), `Ensemble` (Task 4), `dicom_series_to_nifti`/`DicomError` (Task 5), `Staging` (Task 6).
- Produces: `PredictionResult` dataclass (`patient_id: str`, `prob_abnormal: float`, `predicted_label: str`, `staging_rel: str | None`, `gradcam_png: bytes | None`, `error: str | None`); `class Predictor` with `__init__(self, config: AppConfig, ensemble: Ensemble)` and `predict_and_capture(self, dicom_dir, *, capture=True) -> PredictionResult`. `predicted_label`/`prob_abnormal` use `class_index 0 = Abnormal`.

- [ ] **Step 1: Write the failing test** — `tests/test_api.py` (uses fake checkpoints + synthetic DICOM from earlier helpers)

```python
import sys
from pathlib import Path

import numpy as np
import torch
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config
from core.model import build_ensemble_member
from core.ensemble import Ensemble
from core.api import Predictor, PredictionResult


def _fake_ckpts(dir_: Path, n=3):
    dir_.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        torch.manual_seed(i)
        m = build_ensemble_member(2)
        torch.save({"state_dict": m.state_dict()}, dir_ / f"m{i}.pth")


def _dicom_series(dir_: Path, patient_id="P1", n=8):
    dir_.mkdir(parents=True, exist_ok=True)
    w = sitk.ImageFileWriter(); w.KeepOriginalImageUIDOn()
    uid = "1.2.826.0.1.3680043.2.1125.1.222222222222222222222222222222"
    for i in range(n):
        arr = (np.random.rand(32, 32).astype(np.float32) * 400).astype(np.int16)
        img = sitk.GetImageFromArray(arr)
        img.SetMetaData("0010|0020", patient_id)
        img.SetMetaData("0020|000e", uid)
        img.SetMetaData("0020|0032", f"0\\0\\{i}")
        img.SetMetaData("0008|0060", "CT")
        img.SetMetaData("0028|0100", "16")
        w.SetFileName(str(dir_ / f"s{i:03d}.dcm")); w.Execute(img)


def test_predict_and_capture_end_to_end():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _fake_ckpts(d / "models", 3)
        _dicom_series(d / "dicom", "P1", 8)
        # minimal config
        cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
        object.__setattr__(cfg, "models_dir", d / "models")
        object.__setattr__(cfg, "staging_dir", d / "staging")
        object.__setattr__(cfg, "gradcam_enabled", False)
        ens = Ensemble.from_dir(cfg.models_dir, num_classes=2, device="cuda")
        predictor = Predictor(cfg, ens)
        res = predictor.predict_and_capture(d / "dicom", capture=True)
        assert isinstance(res, PredictionResult)
        assert res.error is None
        assert res.patient_id == "P1"
        assert 0.0 <= res.prob_abnormal <= 1.0
        assert res.predicted_label in ("Abnormal", "Normal")
        assert res.staging_rel is not None
    print("test_predict_and_capture_end_to_end PASS")


def test_bad_dicom_returns_error_result():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _fake_ckpts(d / "models", 2)
        cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
        object.__setattr__(cfg, "models_dir", d / "models")
        object.__setattr__(cfg, "staging_dir", d / "staging")
        object.__setattr__(cfg, "gradcam_enabled", False)
        ens = Ensemble.from_dir(cfg.models_dir, num_classes=2, device="cuda")
        res = Predictor(cfg, ens).predict_and_capture(d / "empty", capture=True)
        assert res.error is not None
        assert res.staging_rel is None
    print("test_bad_dicom_returns_error_result PASS")


if __name__ == "__main__":
    test_predict_and_capture_end_to_end()
    test_bad_dicom_returns_error_result()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Research/copd-ct-app && python tests/test_api.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.api'`

- [ ] **Step 3: Write `core/api.py`**

```python
"""Public entry point: DICOM -> prediction + capture. UI-agnostic."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from core.config import AppConfig
from core.dicom_io import dicom_series_to_nifti, DicomError
from core.ensemble import Ensemble
from core.preprocess import load_ct
from core.staging import Staging

ABNORMAL_INDEX = 0  # class_index 0 = Abnormal (training manifest)


@dataclass
class PredictionResult:
    patient_id: str
    prob_abnormal: float
    predicted_label: str
    staging_rel: str | None
    gradcam_png: bytes | None
    error: str | None


class Predictor:
    def __init__(self, config: AppConfig, ensemble: Ensemble):
        self.config = config
        self.ensemble = ensemble
        self.staging = Staging(config.staging_dir, deidentify=config.deidentify)

    def predict_and_capture(self, dicom_dir, *, capture: bool = True) -> PredictionResult:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                nifti = Path(tmp) / "converted.nii.gz"
                dcm = dicom_series_to_nifti(dicom_dir, nifti)
                volume = load_ct(
                    nifti,
                    self.config.image_size,
                    intensity_window=self.config.intensity_window,
                    input_normalization=self.config.input_normalization,
                )
                proba = self.ensemble.predict_proba(volume)
                prob_abnormal = float(proba[ABNORMAL_INDEX])
                pred_idx = int(proba.argmax())
                predicted_label = self.config.class_names[pred_idx]

                staging_rel = None
                if capture:
                    staging_rel = self.staging.capture(
                        nifti,
                        dcm.patient_id,
                        dcm.series_uid,
                        {"prob_abnormal": prob_abnormal, "pred": predicted_label},
                    )
                return PredictionResult(
                    patient_id=dcm.patient_id,
                    prob_abnormal=prob_abnormal,
                    predicted_label=predicted_label,
                    staging_rel=staging_rel,
                    gradcam_png=None,
                    error=None,
                )
        except DicomError as exc:
            return PredictionResult("UNKNOWN", 0.0, "", None, None, f"DICOM error: {exc}")
        except Exception as exc:
            return PredictionResult("UNKNOWN", 0.0, "", None, None, f"Prediction failed: {exc}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Research/copd-ct-app && python tests/test_api.py`
Expected: both PASS lines.

- [ ] **Step 5: Commit**

```bash
cd ~/Research/copd-ct-app
git add core/api.py tests/test_api.py
git commit -m "feat: predict_and_capture public API"
```

---

### Task 8: Grad-CAM heatmap

**Files:**
- Create: `~/Research/copd-ct-app/core/gradcam.py`
- Modify: `~/Research/copd-ct-app/core/api.py` (wire gradcam into the result when enabled)
- Test: `~/Research/copd-ct-app/tests/test_gradcam.py`

**Interfaces:**
- Consumes: a single ensemble member (`torch.nn.Module` with `attention_layers`), the preprocessed `volume`.
- Produces: `gradcam_overlay_png(model, volume, device, target_class=ABNORMAL_INDEX) -> bytes` returning PNG bytes of a mid-axial CT slice with the CAM overlaid.

- [ ] **Step 1: Write the failing test** — `tests/test_gradcam.py`

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.model import build_ensemble_member
from core.gradcam import gradcam_overlay_png


def test_gradcam_returns_png_bytes():
    model = build_ensemble_member(2).cuda().eval()
    vol = np.random.randn(1, 112, 136, 112).astype(np.float32)
    png = gradcam_overlay_png(model, vol, device="cuda", target_class=0)
    assert isinstance(png, (bytes, bytearray))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
    print("test_gradcam_returns_png_bytes PASS")


if __name__ == "__main__":
    test_gradcam_returns_png_bytes()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Research/copd-ct-app && python tests/test_gradcam.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.gradcam'`

- [ ] **Step 3: Write `core/gradcam.py`**

```python
"""Grad-CAM overlay for one ensemble member (target layer: attention_layers)."""

from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def gradcam_overlay_png(model, volume: np.ndarray, device: str = "cuda", target_class: int = 0) -> bytes:
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(volume)).float().unsqueeze(0).to(device)
    x.requires_grad_(True)

    activations = {}
    gradients = {}
    target = model.attention_layers

    def fwd_hook(_m, _i, out):
        activations["v"] = out

    def bwd_hook(_m, _gi, go):
        gradients["v"] = go[0]

    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)
    try:
        logits = model(x)
        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        score.backward()
        act = activations["v"]           # [1, C, d, h, w]
        grad = gradients["v"]            # [1, C, d, h, w]
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # [1,1,d,h,w]
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    finally:
        h1.remove()
        h2.remove()

    vol = volume[0]
    mid = vol.shape[0] // 2
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(vol[mid], cmap="gray")
    ax.imshow(cam[mid], cmap="jet", alpha=0.4)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/Research/copd-ct-app && python tests/test_gradcam.py`
Expected: `test_gradcam_returns_png_bytes PASS`

- [ ] **Step 5: Wire Grad-CAM into `core/api.py`**

In `core/api.py`, add the import at the top:

```python
from core.gradcam import gradcam_overlay_png
```

Then, inside `predict_and_capture`, replace the `gradcam_png=None,` line in the success `PredictionResult` with a computed value. Add this block right after `predicted_label = self.config.class_names[pred_idx]`:

```python
                gradcam_png = None
                if self.config.gradcam_enabled:
                    try:
                        gradcam_png = gradcam_overlay_png(
                            self.ensemble.members[0], volume,
                            device=self.ensemble.device, target_class=ABNORMAL_INDEX,
                        )
                    except Exception:
                        gradcam_png = None
```

And change the success return's `gradcam_png=None,` to `gradcam_png=gradcam_png,`.

- [ ] **Step 6: Run the API test again to confirm no regression**

Run: `cd ~/Research/copd-ct-app && python tests/test_api.py`
Expected: both PASS lines (gradcam disabled in that test, so behavior unchanged).

- [ ] **Step 7: Commit**

```bash
cd ~/Research/copd-ct-app
git add core/gradcam.py core/api.py tests/test_gradcam.py
git commit -m "feat: Grad-CAM overlay wired into prediction"
```

---

### Task 9: Gradio app shell

**Files:**
- Create: `~/Research/copd-ct-app/app.py`

**Interfaces:**
- Consumes: `load_config`, `Ensemble.from_dir`, `Predictor` (Tasks 1/4/7).
- Produces: a runnable Gradio app; no automated test (validated manually — it is a throwaway demo shell, do not polish).

- [ ] **Step 1: Write `app.py`**

```python
"""Gradio demo shell — THROWAWAY. Swapped for the real web app in phase 2.
Keep this thin: all logic lives in core/."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import gradio as gr

from core.api import Predictor
from core.config import load_config
from core.ensemble import Ensemble

CONFIG_PATH = Path(__file__).parent / "config.yaml"
DISCLAIMER = "研究用途、非診斷 (Research use only — not for diagnosis)."

cfg = load_config(CONFIG_PATH)
ensemble = Ensemble.from_dir(cfg.models_dir, num_classes=cfg.num_classes, device="cuda")
predictor = Predictor(cfg, ensemble)


def _extract_to_dir(file_paths: list[str], workdir: Path) -> Path:
    dcm_dir = workdir / "dicom"
    dcm_dir.mkdir(parents=True, exist_ok=True)
    for fp in file_paths:
        fp = Path(fp)
        if fp.suffix.lower() == ".zip":
            with zipfile.ZipFile(fp) as zf:
                zf.extractall(dcm_dir)
        else:
            (dcm_dir / fp.name).write_bytes(fp.read_bytes())
    # If the zip made a single subfolder, descend into it.
    entries = [p for p in dcm_dir.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dcm_dir


def run(files):
    if not files:
        return "請上傳 DICOM 資料夾或 zip。", None
    with tempfile.TemporaryDirectory() as tmp:
        dcm_dir = _extract_to_dir([f.name for f in files], Path(tmp))
        res = predictor.predict_and_capture(dcm_dir, capture=True)
    if res.error:
        return f"❌ {res.error}", None
    lines = [
        f"病人號碼: {res.patient_id}",
        f"Abnormal 機率: {res.prob_abnormal * 100:.1f}%",
        f"預測: {res.predicted_label}",
        f"已擷取: {res.staging_rel}",
    ]
    if cfg.show_disclaimer:
        lines.append("")
        lines.append(DISCLAIMER)
    img = None
    if res.gradcam_png:
        img_path = Path(tempfile.mkstemp(suffix=".png")[1])
        img_path.write_bytes(res.gradcam_png)
        img = str(img_path)
    return "\n".join(lines), img


with gr.Blocks(title="COPD CT — Normal vs Abnormal") as demo:
    gr.Markdown("## COPD CT 分類 (Normal vs Abnormal) — Demo")
    files = gr.File(file_count="multiple", label="上傳 DICOM (資料夾檔案或 .zip)")
    btn = gr.Button("分析")
    out_text = gr.Textbox(label="結果", lines=8)
    out_img = gr.Image(label="Grad-CAM", type="filepath")
    btn.click(run, inputs=files, outputs=[out_text, out_img])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

- [ ] **Step 2: Manual smoke test (requires real checkpoints in `models/current/`)**

Place 5 real checkpoints (or, for a code-path smoke test, run `tests/test_ensemble.py`'s helper to drop fake ones into `models/current/`), then:

Run: `cd ~/Research/copd-ct-app && python app.py`
Expected: Gradio starts on `http://0.0.0.0:7860` with no import errors. Upload a DICOM zip → a result string + Grad-CAM image appears. (Prediction values are meaningless with fake checkpoints; this only verifies the wiring.)

- [ ] **Step 3: Commit**

```bash
cd ~/Research/copd-ct-app
git add app.py
git commit -m "feat: Gradio demo shell"
```

---

### Task 10: Packaging — Docker, environment, README

**Files:**
- Create: `~/Research/copd-ct-app/environment.yml`
- Create: `~/Research/copd-ct-app/Dockerfile`
- Create: `~/Research/copd-ct-app/README.md`

**Interfaces:** none (delivery artifacts).

- [ ] **Step 1: Write `environment.yml`** (dev env; mirrors the working `nnMamba` env)

```yaml
name: copd-ct-app
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - torch==2.5.1
      - mamba-ssm
      - causal-conv1d
      - SimpleITK
      - nibabel
      - scikit-image
      - gradio
      - matplotlib
      - pyyaml
```

- [ ] **Step 2: Write `Dockerfile`** (CUDA base; mamba-ssm baked in)

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir \
    causal-conv1d mamba-ssm SimpleITK nibabel scikit-image gradio matplotlib pyyaml

COPY core/ /app/core/
COPY app.py config.yaml /app/

# models/ and staging/ are mounted at runtime as volumes.
EXPOSE 7860
CMD ["python", "app.py"]
```

- [ ] **Step 3: Write `README.md`**

````markdown
# copd-ct-app

Hospital-facing demo: 3D CT Normal-vs-Abnormal (COPD) classifier with data capture.

## Requirements

- **NVIDIA GPU is mandatory** (mamba-ssm is CUDA-only; no CPU path).
- Docker with `nvidia-container-toolkit`, or a conda env from `environment.yml`.

## Run with Docker

```bash
docker build -t copd-ct-app .
docker run --gpus all -p 7860:7860 \
  -v $PWD/models:/app/models \
  -v $PWD/staging:/app/staging \
  copd-ct-app
```

Open http://localhost:7860, upload a DICOM series (folder files or a `.zip`).

## Run with conda (dev)

```bash
conda env create -f environment.yml
conda activate copd-ct-app
python app.py
```

## Models

Place 5 ensemble checkpoints in `models/current/` (`*.pth`, each a dict with a
`state_dict` key). These are produced research-side by `train_production_ensemble.py`
and shipped via `package_release.py`. Swapping versions = replace `models/current/`
and restart.

## Data capture

Each prediction copies the CT into `staging/incoming/{patient_id}_{timestamp}.nii.gz`
and appends a line to `staging/capture_log.jsonl` (label pending). The researcher
collects `staging/` periodically for offline labeling.

## Notes

- The Gradio UI is a throwaway demo; the production web app replaces it (phase 2).
- Honest performance reference: ~0.73 Acc / 0.80 AUC (nested CV). Do not quote 0.833.
- "研究用途、非診斷" disclaimer is shown by default (config `show_disclaimer`).
````

- [ ] **Step 4: Verify the full test suite passes**

Run:
```bash
cd ~/Research/copd-ct-app && conda activate nnMamba
for t in tests/test_config.py tests/test_preprocess_consistency.py tests/test_model.py \
         tests/test_ensemble.py tests/test_dicom_io.py tests/test_staging.py \
         tests/test_api.py tests/test_gradcam.py; do
  echo "== $t =="; python "$t" || break
done
```
Expected: every test prints its `PASS` line.

- [ ] **Step 5: Commit**

```bash
cd ~/Research/copd-ct-app
git add environment.yml Dockerfile README.md
git commit -m "chore: Docker, environment, and README"
```

---

## Follow-up (separate plan — not this one)

Research-side scripts in nnMamba: `train_production_ensemble.py` (retrain 5 members on all data → `release/<date>/`), `label_backfill.py` (collected staging → match PFT → dataset), `package_release.py` (bit-for-bit preprocess check + bundle). These feed `models/current/` for this app but are built and run on the research machine.

## Self-Review Notes

- **Spec coverage:** §3 architecture (Tasks 1–9), §4 inference flow (Tasks 2/4/5/7), §5 staging (Task 6), §8 Grad-CAM/errors/packaging/testing (Tasks 8/7/10), roadmap Docker (Task 10). Research-side §6/§7 retraining + backfill are deferred to the follow-up plan by scope decision. Consistency test (§1/§8) is Task 2.
- **De-identification** interface present (Task 6, off by default) per spec decision.
- **Class-index convention** (0 = Abnormal) fixed in Tasks 1/4/7 and used consistently.
- **No pytest** — all tests use the inline-runner convention per project memory.
