"""Bundle a production release for copd-ct-app, blocking on preprocessing drift.

The hospital repo ships a FROZEN copy of the CT preprocessing. This script is
the gate that proves that frozen copy still produces bit-for-bit identical
output to the training preprocessing before any release leaves the research
machine. A failed check blocks the release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


DENSITY_BUILDER = (
    Path(__file__).resolve().parents[1]
    / "experiments" / "ratio5_20260918" / "run_1257_maskfree.py"
)


def check_density_matches(app_repo: Path) -> tuple[bool, str]:
    """Return (True, "") iff the app's emphysema channel matches the training one.

    Runs the real builder — run_1257_maskfree.maskfree(), the function that wrote the
    channel the 2-channel checkpoints were trained on — over a synthetic volume, with
    its output directory redirected to a temporary one, and compares the uint8 array
    against the app's frozen maskfree_density. Bit-for-bit or nothing: the ways this
    channel goes wrong (thresholding after the resize, dropping the uint8 step, using
    clipped HU) all produce a plausible array that silently shifts every prediction.

    Bundles without a density channel are not affected; main() only calls this when the
    release being packaged declares in_channels=2.
    """
    import importlib.util

    if not DENSITY_BUILDER.exists():
        return False, f"density builder not found: {DENSITY_BUILDER}"
    path = Path(app_repo) / "core" / "preprocess_density.py"
    if not path.exists():
        return False, f"app has no core/preprocess_density.py: {path}"

    spec = importlib.util.spec_from_file_location("frozen_density", path)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)

    spec_b = importlib.util.spec_from_file_location("maskfree_builder", DENSITY_BUILDER)
    builder = importlib.util.module_from_spec(spec_b)
    spec_b.loader.exec_module(builder)

    rng = np.random.default_rng(0)
    # Deliberately spans both thresholds the builder writes, with a thin emphysema slab
    # that only survives if the threshold is applied before the resize.
    volume = (rng.random((90, 100, 80)).astype(np.float32) * 1400.0) - 1000.0
    volume[:6] = -1000.0
    volume[6:10] = -930.0

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        nii = d / "vol.nii.gz"
        nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), str(nii))
        original = builder.DENSITY
        builder.DENSITY = d
        try:
            pid, err = builder.maskfree(("synthetic", nii))
        except Exception as exc:
            return False, f"training builder raised: {type(exc).__name__}: {exc}"
        finally:
            builder.DENSITY = original
        if err:
            return False, f"training builder failed: {err}"
        theirs = np.load(d / "synthetic.npy")[0]

    mine = frozen.maskfree_density(volume, tuple(theirs.shape))
    mine_u8 = np.rint(mine[0] * 255.0).astype(np.uint8)
    if mine_u8.shape != theirs.shape:
        return False, f"shape mismatch: app {mine_u8.shape} vs training {theirs.shape}"
    if not np.array_equal(mine_u8, theirs):
        worst = int(np.abs(mine_u8.astype(int) - theirs.astype(int)).max())
        differing = int((mine_u8 != theirs).sum())
        return False, (
            f"density channel differs: {differing} voxels, max |delta| {worst}/255"
        )
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
    # Hash the LF-normalised bytes. The app repo is checked out on both Linux and
    # Windows, and git rewrites line endings on the Windows checkout, so hashing the
    # raw bytes makes this record depend on which machine packaged the release -- a
    # later repackage on the other platform reads as preprocessing drift when nothing
    # about the code has changed. The real gate is check_preprocess_matches(), which
    # compares the produced arrays; this digest is provenance and must be stable.
    digest = hashlib.sha256(
        preprocess_src.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
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

    # A 2-channel release depends on a second frozen function, so it needs a second
    # gate. Read the release's own metrics.json rather than a flag: the bundle already
    # has to declare its shape for the app, and one source of truth is enough.
    try:
        declared = json.loads(
            (Path(args.release) / "metrics.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        declared = {}
    if int(declared.get("in_channels", 1)) > 1:
        ok, reason = check_density_matches(Path(args.app_repo))
        if not ok:
            print(f"RELEASE BLOCKED - density channel drift detected:\n  {reason}")
            raise SystemExit(1)
        print("Density check: frozen copy matches the training builder bit-for-bit.")

    out = bundle_release(Path(args.release), Path(args.app_repo), Path(args.dest))
    print(f"Release bundled: {out}")
    print("Ship this dir to the hospital and point models/current at it, then restart the app.")


if __name__ == "__main__":
    main()
