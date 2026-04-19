import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from evaluation import load_hidden_targets, summarize_scores
from validation import validate_submission_zip

API = HfApi()

SUBMISSIONS_REPO = os.environ.get("SUBMISSIONS_REPO", "your-org/the-well-submissions")
RESULTS_REPO = os.environ.get("RESULTS_REPO", "your-org/the-well-results")
HF_TOKEN = os.environ.get("HF_TOKEN")
POLL_INTERVAL_SECONDS = int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))
TASK_NAME = "turbulent_radiative_layer_2D_1step"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def list_json_files(repo_id: str, prefix: str) -> list[str]:
    return [
        path
        for path in API.list_repo_files(repo_id=repo_id, repo_type="dataset", token=HF_TOKEN)
        if path.startswith(prefix) and path.endswith(".json")
    ]


def extract_id(path: str, prefix: str) -> str:
    return path.removeprefix(prefix).removesuffix(".json")


def get_pending_submission_ids() -> list[str]:
    metadata_files = list_json_files(SUBMISSIONS_REPO, "metadata/")
    result_files = list_json_files(RESULTS_REPO, "results/")

    submitted_ids = {extract_id(path, "metadata/") for path in metadata_files}
    finished_ids = {extract_id(path, "results/") for path in result_files}

    pending = sorted(submitted_ids - finished_ids)
    return pending


def download_submission_metadata(submission_id: str) -> dict[str, Any]:
    local_path = hf_hub_download(
        repo_id=SUBMISSIONS_REPO,
        repo_type="dataset",
        filename=f"metadata/{submission_id}.json",
        token=HF_TOKEN,
    )
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_submission_package(package_path: str) -> bytes:
    local_path = hf_hub_download(
        repo_id=SUBMISSIONS_REPO,
        repo_type="dataset",
        filename=package_path,
        token=HF_TOKEN,
    )
    with open(local_path, "rb") as f:
        return f.read()


def upload_result_record(submission_id: str, record: dict[str, Any]) -> None:
    API.upload_file(
        path_or_fileobj=json.dumps(record, indent=2).encode("utf-8"),
        path_in_repo=f"results/{submission_id}.json",
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )


def build_failure_record(metadata: dict[str, Any], error_message: str) -> dict[str, Any]:
    return {
        "submission_id": metadata["submission_id"],
        "task_name": metadata.get("task_name", TASK_NAME),
        "status": "failed",
        "created_at": utc_now_iso(),
        "submitted_at": metadata.get("submitted_at"),
        "model_name": metadata.get("model_name"),
        "team_name": metadata.get("team_name"),
        "method_name": metadata.get("method_name"),
        "avg_vrmse": None,
        "density_vrmse": None,
        "pressure_vrmse": None,
        "velocity_x_vrmse": None,
        "velocity_y_vrmse": None,
        "n_samples": None,
        "error_message": error_message[:1000],
    }


def build_success_record(metadata: dict[str, Any], manifest: dict[str, Any], scores: dict[str, float], n_samples: int) -> dict[str, Any]:
    return {
        "submission_id": metadata["submission_id"],
        "task_name": manifest["task_name"],
        "status": "succeeded",
        "created_at": utc_now_iso(),
        "submitted_at": metadata.get("submitted_at"),
        "model_name": manifest.get("model_name"),
        "team_name": manifest.get("team_name"),
        "method_name": manifest.get("method_name"),
        "avg_vrmse": scores["avg_vrmse"],
        "density_vrmse": scores["density_vrmse"],
        "pressure_vrmse": scores["pressure_vrmse"],
        "velocity_x_vrmse": scores["velocity_x_vrmse"],
        "velocity_y_vrmse": scores["velocity_y_vrmse"],
        "n_samples": int(n_samples),
        "error_message": None,
    }


def process_submission(submission_id: str) -> None:
    metadata = download_submission_metadata(submission_id)
    package_bytes = download_submission_package(metadata["package_path"])

    try:
        hidden = load_hidden_targets()
        validated = validate_submission_zip(package_bytes, expected_sample_ids=hidden.sample_ids)
        scores = summarize_scores(validated.predictions, hidden.targets)
        record = build_success_record(
            metadata=metadata,
            manifest=validated.manifest,
            scores=scores,
            n_samples=len(validated.sample_ids),
        )
    except Exception as exc:
        record = build_failure_record(metadata, str(exc))

    record["package_sha256"] = hashlib.sha256(package_bytes).hexdigest()
    upload_result_record(submission_id, record)


def main() -> None:
    print("Evaluator started")
    while True:
        pending_ids = get_pending_submission_ids()
        if pending_ids:
            print(f"Found {len(pending_ids)} pending submissions")
            for submission_id in pending_ids:
                print(f"Processing {submission_id}")
                process_submission(submission_id)
        else:
            print(f"[{utc_now_iso()}] No pending submissions")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
