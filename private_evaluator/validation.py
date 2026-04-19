import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np

EXPECTED_TASK = "turbulent_radiative_layer_2D_1step"
EXPECTED_SHAPE_TAIL = (1, 128, 384, 4)
REQUIRED_ARCHIVE_FILES = ["predictions.npz", "submission.json"]
REQUIRED_MANIFEST_KEYS = [
    "task_name",
    "model_name",
    "team_name",
    "method_name",
    "authors",
    "contact_email",
    "description",
    "is_external_data_used",
]


@dataclass
class ValidatedSubmission:
    manifest: dict[str, Any]
    sample_ids: np.ndarray
    predictions: np.ndarray


def validate_submission_zip(zip_bytes: bytes, expected_sample_ids: np.ndarray) -> ValidatedSubmission:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = sorted(zf.namelist())
        if names != REQUIRED_ARCHIVE_FILES:
            raise ValueError(
                "Archive must contain exactly two root files: submission.json and predictions.npz."
            )

        with zf.open("submission.json") as f:
            manifest = json.load(f)

        with zf.open("predictions.npz") as f:
            npz_bytes = f.read()

    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            raise ValueError(f"submission.json is missing required key: {key}")

    if manifest["task_name"] != EXPECTED_TASK:
        raise ValueError(f"task_name must be '{EXPECTED_TASK}'.")

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        if "sample_ids" not in data or "predictions" not in data:
            raise ValueError("predictions.npz must contain 'sample_ids' and 'predictions'.")
        sample_ids = np.asarray(data["sample_ids"])
        predictions = np.asarray(data["predictions"])

    if sample_ids.ndim != 1:
        raise ValueError("sample_ids must be a 1D array.")

    if predictions.ndim != 5:
        raise ValueError("predictions must have shape (N, 1, 128, 384, 4).")

    if tuple(predictions.shape[1:]) != EXPECTED_SHAPE_TAIL:
        raise ValueError(
            f"predictions must have shape (N, {EXPECTED_SHAPE_TAIL[0]}, "
            f"{EXPECTED_SHAPE_TAIL[1]}, {EXPECTED_SHAPE_TAIL[2]}, {EXPECTED_SHAPE_TAIL[3]})."
        )

    if len(sample_ids) != predictions.shape[0]:
        raise ValueError("sample_ids length must match predictions.shape[0].")

    if not np.issubdtype(predictions.dtype, np.number):
        raise ValueError("predictions must be numeric.")

    if not np.isfinite(predictions).all():
        raise ValueError("predictions contains NaN or Inf.")

    sample_ids = sample_ids.astype(str)
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("sample_ids must be unique.")

    expected_sample_ids = expected_sample_ids.astype(str)
    if sample_ids.shape != expected_sample_ids.shape:
        raise ValueError("sample_ids length does not match the hidden test set.")

    if not np.array_equal(sample_ids, expected_sample_ids):
        raise ValueError("sample_ids do not match the benchmark hidden test order.")

    return ValidatedSubmission(
        manifest=manifest,
        sample_ids=sample_ids,
        predictions=predictions.astype(np.float32, copy=False),
    )
