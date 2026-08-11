from __future__ import annotations

import pickle
import time
import zlib
from dataclasses import dataclass

import numpy as np


def label_jsd(labels: np.ndarray) -> float:
    counts = np.bincount(labels.astype(int), minlength=2).astype(float)
    probabilities = counts / counts.sum()
    uniform = np.array([0.5, 0.5], dtype=float)
    midpoint = 0.5 * (probabilities + uniform)

    def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
        nonzero = left > 0
        return float(np.sum(left[nonzero] * np.log2(left[nonzero] / right[nonzero])))

    return 0.5 * kl_divergence(probabilities, midpoint) + 0.5 * kl_divergence(
        uniform, midpoint
    )


def random_minority_oversample(
    features: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2 or counts[0] == counts[1]:
        return features.copy(), labels.copy()

    target_size = int(counts.max())
    sampled_indices: list[np.ndarray] = []
    for class_value, class_count in zip(classes, counts):
        indices = np.flatnonzero(labels == class_value)
        if class_count < target_size:
            extra = rng.choice(
                indices, size=target_size - int(class_count), replace=True
            )
            indices = np.concatenate([indices, extra])
        sampled_indices.append(indices)

    combined = np.concatenate(sampled_indices)
    rng.shuffle(combined)
    return features[combined].copy(), labels[combined].copy()


@dataclass(frozen=True)
class WirePayload:
    values: tuple[np.ndarray, np.ndarray]
    wire_bytes: int
    raw_bytes: int
    processing_seconds: float


def transmit_parameters(
    coefficient: np.ndarray,
    intercept: np.ndarray,
    compressed: bool,
    compression_level: int,
) -> WirePayload:
    started = time.perf_counter()
    raw = pickle.dumps((coefficient, intercept), protocol=pickle.HIGHEST_PROTOCOL)
    if compressed:
        wire = zlib.compress(raw, level=compression_level)
        restored = zlib.decompress(wire)
    else:
        wire = raw
        restored = wire
    restored_coefficient, restored_intercept = pickle.loads(restored)
    processing_seconds = time.perf_counter() - started
    return WirePayload(
        values=(restored_coefficient, restored_intercept),
        wire_bytes=len(wire),
        raw_bytes=len(raw),
        processing_seconds=processing_seconds,
    )
