import hashlib
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

API = HfApi()

SUBMISSIONS_REPO = os.environ.get("SUBMISSIONS_REPO", "your-org/the-well-submissions")
RESULTS_REPO = os.environ.get("RESULTS_REPO", "your-org/the-well-results")
HF_TOKEN = os.environ.get("HF_TOKEN")
MAX_SUBMISSION_MB = int(os.environ.get("MAX_SUBMISSION_MB", "200"))

EXPECTED_TASK = "turbulent_radiative_layer_2D_1step"
RESULT_COLUMNS = [
    "rank",
    "model_name",
    "team_name",
    "avg_vrmse",
    "density_vrmse",
    "pressure_vrmse",
    "velocity_x_vrmse",
    "velocity_y_vrmse",
    "submitted_at",
    "status",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value.strip())
    return cleaned[:80] or "submission"


def _read_submission_manifest(zip_bytes: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = sorted(zf.namelist())
        if names != ["predictions.npz", "submission.json"]:
            raise ValueError(
                "The zip must contain exactly two root files: submission.json and predictions.npz."
            )
        with zf.open("submission.json") as f:
            manifest = json.load(f)
    if manifest.get("task_name") != EXPECTED_TASK:
        raise ValueError(f"task_name must be '{EXPECTED_TASK}'.")
    if not str(manifest.get("model_name", "")).strip():
        raise ValueError("submission.json must include a non-empty model_name.")
    if not str(manifest.get("team_name", "")).strip():
        raise ValueError("submission.json must include a non-empty team_name.")
    return manifest


def submit_zip(zip_file) -> str:
    if zip_file is None:
        return "Please upload a submission `.zip` file."

    local_path = zip_file.name
    if not local_path.lower().endswith(".zip"):
        return "Invalid file type. Please upload a `.zip` file."

    file_size = os.path.getsize(local_path)
    if file_size > MAX_SUBMISSION_MB * 1024 * 1024:
        return f"Submission too large. Limit is {MAX_SUBMISSION_MB} MB."

    with open(local_path, "rb") as f:
        zip_bytes = f.read()

    try:
        manifest = _read_submission_manifest(zip_bytes)
    except Exception as exc:
        return f"Submission rejected: {exc}"

    submitted_at = _utc_now_iso()
    base_name = _safe_slug(manifest["model_name"])
    submission_id = f"{base_name}_{submitted_at}".replace(":", "-")
    sha256 = hashlib.sha256(zip_bytes).hexdigest()

    package_path = f"packages/{submission_id}.zip"
    metadata_path = f"metadata/{submission_id}.json"

    metadata = {
        "submission_id": submission_id,
        "task_name": manifest["task_name"],
        "model_name": manifest["model_name"],
        "team_name": manifest["team_name"],
        "method_name": manifest.get("method_name", ""),
        "submitted_at": submitted_at,
        "package_path": package_path,
        "sha256": sha256,
        "status": "pending",
    }

    API.upload_file(
        path_or_fileobj=zip_bytes,
        path_in_repo=package_path,
        repo_id=SUBMISSIONS_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    API.upload_file(
        path_or_fileobj=json.dumps(metadata, indent=2).encode("utf-8"),
        path_in_repo=metadata_path,
        repo_id=SUBMISSIONS_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    return (
        f"Submission received: `{submission_id}`\n\n"
        "It was uploaded to the submissions dataset and will appear on the leaderboard "
        "after the private evaluator processes it."
    )


def _download_json_records(repo_id: str, prefix: str) -> list[dict[str, Any]]:
    files = [
        path
        for path in API.list_repo_files(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN)
        if path.startswith(prefix) and path.endswith(".json")
    ]
    records = []
    for path in files:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=path,
            token=HF_TOKEN,
        )
        with open(local_path, "r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def load_results_dataframe() -> pd.DataFrame:
    try:
        records = _download_json_records(RESULTS_REPO, "results/")
    except Exception:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    if not records:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    df = pd.DataFrame.from_records(records)
    if "status" in df.columns:
        df = df[df["status"] == "succeeded"].copy()
    if df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    for column in [
        "avg_vrmse",
        "density_vrmse",
        "pressure_vrmse",
        "velocity_x_vrmse",
        "velocity_y_vrmse",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("avg_vrmse", ascending=True).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    for column in RESULT_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[RESULT_COLUMNS]
