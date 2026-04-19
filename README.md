# The Well Benchmark MVP

MVP benchmark for `turbulent_radiative_layer_2D` on Hugging Face.

This project implements a small but practical benchmark setup with:

- a public leaderboard Space,
- a private evaluator Space,
- a private submissions dataset,
- a results dataset,
- and a first benchmark protocol for 1-step forecasting.

## Task

- Dataset: `turbulent_radiative_layer_2D`
- Task name: `turbulent_radiative_layer_2D_1step`
- Input horizon: `n_steps_input = 4`
- Output horizon: `n_steps_output = 1`
- Primary metric: `avg_vrmse`

`avg_vrmse` is the mean VRMSE across the four output fields:

- `density`
- `pressure`
- `velocity_x`
- `velocity_y`

Lower is better.

## Architecture

The benchmark is split into four Hugging Face repos:

1. Public leaderboard Space
   - accepts `.zip` submissions,
   - writes packages and metadata to the submissions dataset,
   - reads results from the results dataset,
   - displays the leaderboard.

2. Private evaluator Space
   - polls for pending submissions,
   - validates format and tensor shape,
   - loads the private test targets,
   - computes VRMSE,
   - writes either a success or failure result record.

3. Private submissions dataset
   - stores uploaded submission archives,
   - stores immutable submission metadata records.

4. Results dataset
   - stores one result JSON per submission,
   - may be public if you want a transparent leaderboard.

## Proposed Hugging Face repos

Replace these with your own org/user names:

- Public Space: `your-org/the-well-leaderboard`
- Private Space: `your-org/the-well-evaluator`
- Private dataset: `your-org/the-well-submissions`
- Results dataset: `your-org/the-well-results`

## Repository layout

```text
the_well_leaderboard_mvp/
├── README.md
├── docs/
│   ├── benchmark_protocol.md
│   └── submission_format.md
├── public_space/
│   ├── README.md
│   ├── app.py
│   ├── about.py
│   ├── requirements.txt
│   └── utils.py
└── private_evaluator/
    ├── README.md
    ├── evaluation.py
    ├── main.py
    ├── requirements.txt
    └── validation.py
```

## Dataset repo layout

### Submissions dataset

```text
the-well-submissions/
├── metadata/
│   ├── <submission_id>.json
│   └── ...
└── packages/
    ├── <submission_id>.zip
    └── ...
```

### Results dataset

```text
the-well-results/
└── results/
    ├── <submission_id>.json
    └── ...
```

Each result JSON should share the same schema, even for failed runs.

## Deployment

### 1. Create the datasets

```bash
huggingface-cli repo create the-well-submissions --type dataset --private
huggingface-cli repo create the-well-results --type dataset
```

### 2. Create the Spaces

- Create a public Gradio Space and upload `public_space/`.
- Create a private Space and upload `private_evaluator/`.

### 3. Add Space secrets

Add these secrets to both Spaces:

- `HF_TOKEN`
- `SUBMISSIONS_REPO`
- `RESULTS_REPO`

Add these to the evaluator Space:

- `PRIVATE_TEST_PATH` or another path/config you use to load hidden targets
- any extra auth/config needed for private data access

### 4. Provide hidden evaluation targets

The evaluator needs hidden test targets aligned to the benchmark submission order.

Important:

- participants must have access to the test inputs,
- participants must not have access to the test targets,
- the evaluator must load targets in exactly the same sample order expected by `sample_ids`.

## Suggested benchmark protocol

See:

- [docs/benchmark_protocol.md](docs/benchmark_protocol.md)
- [docs/submission_format.md](docs/submission_format.md)

## Current status

This repo is a starter MVP. It is intentionally incomplete in a few places:

- the hidden test target loader is left as a `TODO`,
- the exact hidden test split generation procedure is described but not yet materialized,
- duplicate submission policy and rate limiting are not enforced yet,
- model cards and authenticated attribution are optional for v1.

## Recommended next step

Implement the hidden test bundle first:

1. define the exact test sample order,
2. export public test inputs for participants,
3. export private test targets for the evaluator,
4. freeze `sample_ids`,
5. then test one end-to-end submission locally.
