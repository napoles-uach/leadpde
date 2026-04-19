# Private Evaluator Space

Private evaluator for the The Well benchmark MVP.

Responsibilities:

- detect pending submissions,
- validate archive contents,
- load hidden targets,
- compute VRMSE,
- write a final result record.

Required Space secrets:

- `HF_TOKEN`
- `SUBMISSIONS_REPO`
- `RESULTS_REPO`

Recommended Space variables:

- `POLL_INTERVAL_SECONDS`
- `PRIVATE_TEST_PATH`

Important:

- keep this Space private,
- keep hidden target tensors private,
- make sure participant-visible test inputs use the exact same `sample_ids`.
