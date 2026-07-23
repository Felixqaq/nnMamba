"""Inline-runner tests for scripts/package_release.py."""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.package_release import check_preprocess_matches, bundle_release

APP_REPO = Path.home() / "Research" / "copd-ct-app"


def test_check_passes_on_current_frozen_preprocess():
    ok, reason = check_preprocess_matches(APP_REPO)
    assert ok, f"expected frozen preprocess to match training, got: {reason}"
    print("test_check_passes_on_current_frozen_preprocess PASS")


def test_check_fails_on_tampered_preprocess():
    with tempfile.TemporaryDirectory() as d:
        fake_repo = Path(d) / "copd-ct-app"
        (fake_repo / "core").mkdir(parents=True)
        src = (APP_REPO / "core" / "preprocess.py").read_text()
        # Tamper: change the resize interpolation order, which alters output values.
        tampered = src.replace("order=1,", "order=0,")
        assert tampered != src, "tamper failed to modify the source"
        (fake_repo / "core" / "preprocess.py").write_text(tampered)

        ok, reason = check_preprocess_matches(fake_repo)
        assert not ok, "tampered preprocess must be rejected"
        assert reason
    print("test_check_fails_on_tampered_preprocess PASS")


def test_bundle_copies_members_metrics_and_preprocess():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        release = d / "release"
        release.mkdir()
        for i in (1, 2):
            torch.save({"state_dict": {"w": torch.zeros(2)}, "seed": i}, release / f"member_{i}.pth")
        (release / "metrics.json").write_text(json.dumps({"held_out": False}))

        dest = d / "bundle"
        out = bundle_release(release, APP_REPO, dest)
        assert (out / "member_1.pth").exists()
        assert (out / "member_2.pth").exists()
        assert (out / "metrics.json").exists()
        assert (out / "preprocess.py").exists()
        assert (out / "PREPROCESS_HASH").exists()
        assert len((out / "PREPROCESS_HASH").read_text().strip()) == 64
    print("test_bundle_copies_members_metrics_and_preprocess PASS")


if __name__ == "__main__":
    test_check_passes_on_current_frozen_preprocess()
    test_check_fails_on_tampered_preprocess()
    test_bundle_copies_members_metrics_and_preprocess()
