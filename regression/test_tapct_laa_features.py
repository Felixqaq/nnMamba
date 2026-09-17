"""Smoke tests for frozen TAP-CT plus LAA late-fusion features."""

from __future__ import annotations

import numpy as np

from scripts.extract_laa_tapct_features import extract_feature_vectors


def test_feature_shapes_and_reconstructed_laa() -> None:
    stored = np.zeros((2, 8, 8, 8), dtype=np.uint8)
    stored[1, 2:6, 2:6, 2:6] = 255
    stored[0, 2:4, 2:6, 2:6] = 255
    density, density_names, combined, combined_names = extract_feature_vectors(
        stored, grid_size=2
    )
    assert density.shape == (19,)
    assert combined.shape == (48,)
    assert len(density_names) == len(density)
    assert len(combined_names) == len(combined)
    assert np.isfinite(combined).all()
    global_index = combined_names.index("laa950_global_ratio")
    core_index = combined_names.index("laa950_core_ratio")
    np.testing.assert_allclose(combined[global_index], 0.5, atol=1e-6)
    np.testing.assert_allclose(combined[core_index], 0.5, atol=1e-6)


if __name__ == "__main__":
    test_feature_shapes_and_reconstructed_laa()
    print("TapCT LAA feature smoke tests passed")
