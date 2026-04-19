from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

FIELD_NAMES = ["density", "pressure", "velocity_x", "velocity_y"]


@dataclass
class HiddenTargets:
    sample_ids: np.ndarray
    targets: np.ndarray


def load_hidden_targets() -> HiddenTargets:
    """
    Load the hidden test targets used by the evaluator.

    Expected arrays in the hidden target bundle:
    - sample_ids: shape (N,)
    - targets: shape (N, 1, 128, 384, 4)

    TODO:
    Replace this loader with your finalized hidden test data pipeline.
    """
    private_test_path = os.environ.get("PRIVATE_TEST_PATH")
    if not private_test_path:
        raise RuntimeError("PRIVATE_TEST_PATH is not configured.")

    with np.load(private_test_path, allow_pickle=False) as data:
        sample_ids = np.asarray(data["sample_ids"]).astype(str)
        targets = np.asarray(data["targets"], dtype=np.float32)

    if targets.ndim != 5 or tuple(targets.shape[1:]) != (1, 128, 384, 4):
        raise RuntimeError("Hidden targets must have shape (N, 1, 128, 384, 4).")

    if len(sample_ids) != targets.shape[0]:
        raise RuntimeError("Hidden targets sample_ids and targets are misaligned.")

    return HiddenTargets(sample_ids=sample_ids, targets=targets)


def compute_field_vrmse(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute VRMSE per field.

    predictions and targets must have shape (N, 1, 128, 384, 4).
    """
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")

    # Aggregate over sample, time, and spatial dimensions. Keep field dimension.
    reduce_axes = (0, 1, 2, 3)
    mse = np.mean((predictions - targets) ** 2, axis=reduce_axes)
    variance = np.var(targets, axis=reduce_axes)
    variance = np.maximum(variance, 1e-12)
    return np.sqrt(mse / variance)


def summarize_scores(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    field_vrmse = compute_field_vrmse(predictions, targets)
    scores = {f"{name}_vrmse": float(value) for name, value in zip(FIELD_NAMES, field_vrmse)}
    scores["avg_vrmse"] = float(field_vrmse.mean())
    return scores
