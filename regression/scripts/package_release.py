"""Bundle a production release for copd-ct-app, blocking on preprocessing drift.

The hospital repo ships a FROZEN copy of the CT preprocessing. This script is
the gate that proves that frozen copy still produces bit-for-bit identical
output to the training preprocessing before any release leaves the research
machine. A failed check blocks the release.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import load_ct as training_load_ct  # noqa: E402

IMAGE_SIZE = (112, 136, 112)
INTENSITY_WINDOW = (-1000.0, 400.0)
INPUT_NORMALIZATION = "zscore"


def _load_frozen_preprocess(app_repo: Path):
    """Import the hospital repo's frozen preprocess module from its file path."""
    path = Path(app_repo) / "core" / "preprocess.py"
    if not path.exists():
        raise FileNotFoundError(f"Frozen preprocess not found: {path}")
    spec = importlib.util.spec_from_file_location("frozen_preprocess", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_preprocess_matches(app_repo: Path) -> tuple[bool, str]:
    """Return (True, "") iff the frozen preprocess matches training bit-for-bit."""
    try:
        frozen = _load_frozen_preprocess(app_repo)
    except Exception as exc:
        return False, f"could not load frozen preprocess: {type(exc).__name__}: {exc}"

    rng = np.random.default_rng(0)
    volume = (rng.random((90, 100, 80)).astype(np.float32) * 1400.0) - 1000.0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "vol.nii.gz"
        nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), str(p))
        kwargs = dict(intensity_window=INTENSITY_WINDOW, input_normalization=INPUT_NORMALIZATION)
        try:
            a = frozen.load_ct(p, IMAGE_SIZE, **kwargs)
        except Exception as exc:
            return False, f"frozen load_ct raised: {type(exc).__name__}: {exc}"
        b = training_load_ct(p, IMAGE_SIZE, **kwargs)

    if a.shape != b.shape:
        return False, f"shape mismatch: frozen {a.shape} vs training {b.shape}"
    if not np.array_equal(a, b):
        diff = float(np.abs(a - b).max())
        return False, f"value mismatch: max abs diff {diff}"
    return True, ""


def bundle_release(release_dir: Path, app_repo: Path, dest: Path) -> Path:
    """Copy checkpoints, metrics, and the frozen preprocess into a release bundle."""
    release_dir = Path(release_dir)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    members = sorted(release_dir.glob("member_*.pth"))
    if not members:
        raise FileNotFoundError(f"No member_*.pth in {release_dir}")
    for m in members:
        shutil.copyfile(m, dest / m.name)

    metrics = release_dir / "metrics.json"
    if metrics.exists():
        shutil.copyfile(metrics, dest / "metrics.json")

    preprocess_src = Path(app_repo) / "core" / "preprocess.py"
    shutil.copyfile(preprocess_src, dest / "preprocess.py")
    digest = hashlib.sha256(preprocess_src.read_bytes()).hexdigest()
    (dest / "PREPROCESS_HASH").write_text(digest + "\n")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a production release for copd-ct-app")
    parser.add_argument("--release", required=True, help="Dir containing member_*.pth and metrics.json")
    parser.add_argument("--app-repo", default=str(Path.home() / "Research" / "copd-ct-app"))
    parser.add_argument("--dest", required=True, help="Output bundle dir")
    args = parser.parse_args()

    ok, reason = check_preprocess_matches(Path(args.app_repo))
    if not ok:
        print(f"RELEASE BLOCKED — preprocessing drift detected:\n  {reason}")
        raise SystemExit(1)
    print("Preprocessing check: frozen copy matches training bit-for-bit.")

    out = bundle_release(Path(args.release), Path(args.app_repo), Path(args.dest))
    print(f"Release bundled: {out}")
    print("Ship this dir to the hospital and point models/current at it, then restart the app.")


if __name__ == "__main__":
    main()
