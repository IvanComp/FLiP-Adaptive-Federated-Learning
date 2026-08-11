from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from prostate_flip.patterns import label_jsd, random_minority_oversample  # noqa: E402


def test_jsd_is_zero_for_balanced_binary_labels() -> None:
    assert label_jsd(np.array([0, 1, 0, 1])) == 0.0


def test_tabular_hdh_balances_the_minority_class() -> None:
    features = np.arange(40, dtype=float).reshape(10, 4)
    labels = np.array([0] * 8 + [1] * 2)
    balanced_features, balanced_labels = random_minority_oversample(
        features, labels, np.random.default_rng(1234)
    )
    assert balanced_features.shape == (16, 4)
    assert np.bincount(balanced_labels).tolist() == [8, 8]
