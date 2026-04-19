# Benchmark Protocol

## 1. Task definition

- Task name: `turbulent_radiative_layer_2D_1step`
- Dataset family: The Well
- Dataset instance: `turbulent_radiative_layer_2D`
- Forecast setting: one-step forecasting
- Inputs: `4` previous time steps
- Targets: `1` next time step

Each benchmark example corresponds to one sliding window:

- input tensor shape: `(4, 128, 384, 4)`
- target tensor shape: `(1, 128, 384, 4)`

Field order is fixed:

1. `density`
2. `pressure`
3. `velocity_x`
4. `velocity_y`

The benchmark assumes The Well conventions as shown in the tutorial:

- `input_fields` contains time-varying inputs,
- `output_fields` contains future targets,
- fields are flattened into the last dimension.

## 2. Public benchmark assets

Participants should receive:

- benchmark task description,
- submission template,
- list of `sample_ids` for the hidden test inputs,
- public test input tensors or a reproducible loader for them.

Participants should not receive:

- hidden target tensors,
- evaluator internals that expose the target values.

## 3. Submission package

Each submission is one `.zip` archive containing exactly:

- `submission.json`
- `predictions.npz`

See the dedicated format spec in
[submission_format.md](/Users/napolesd/Documents/New project/the_well_leaderboard_mvp/docs/submission_format.md).

## 4. Evaluation procedure

For each valid submission:

1. read `submission.json`,
2. load `predictions.npz`,
3. verify task name and tensor shape,
4. verify `sample_ids` match the benchmark hidden test set exactly,
5. load hidden targets,
6. compute VRMSE per field,
7. compute `avg_vrmse` as the arithmetic mean across fields,
8. write a result record to the results dataset.

Lower scores are better.

## 5. Metric definition

Primary leaderboard metric:

- `avg_vrmse`

Secondary metrics:

- `density_vrmse`
- `pressure_vrmse`
- `velocity_x_vrmse`
- `velocity_y_vrmse`

For this MVP, VRMSE is computed per field over all submitted hidden-test examples.

Conceptually:

- compute squared error between prediction and target,
- average over sample/time/space dimensions,
- divide by target variance for the same field,
- take the square root.

The final leaderboard score is the mean of the four field VRMSE values.

## 6. Leaderboard fields

Each result record should contain at least:

- `submission_id`
- `task_name`
- `status`
- `created_at`
- `submitted_at`
- `model_name`
- `team_name`
- `method_name`
- `avg_vrmse`
- `density_vrmse`
- `pressure_vrmse`
- `velocity_x_vrmse`
- `velocity_y_vrmse`
- `n_samples`
- `error_message`

`status` should be one of:

- `pending`
- `running`
- `succeeded`
- `failed`

For this MVP, the evaluator writes only final records, so stored result statuses will usually be:

- `succeeded`
- `failed`

## 7. Invalid submission cases

A submission should fail if any of the following is true:

- archive is not a `.zip`,
- required files are missing,
- extra unexpected files are present,
- `submission.json` is invalid JSON,
- `task_name` is not `turbulent_radiative_layer_2D_1step`,
- `predictions` array is missing,
- `sample_ids` array is missing,
- tensor shape is not exactly `(N, 1, 128, 384, 4)`,
- `sample_ids` length does not match `N`,
- `sample_ids` are duplicated,
- `sample_ids` do not match the evaluator's hidden test order,
- prediction tensor contains NaN or Inf,
- dtype is not numeric,
- archive exceeds the configured file size limit.

## 8. Failure reporting

Failed submissions should still generate a result record with:

- `status = "failed"`
- a short `error_message`
- metric fields left as `null`

This keeps the process auditable and helps users fix formatting issues.

## 9. Benchmark policy suggestions

For v1, keep policy minimal:

- allow multiple submissions,
- show best score per `model_name` on the public board if desired,
- keep all result records in the dataset,
- do not accept external labels or target leakage.

Later, you can add:

- daily submission caps,
- authenticated attribution,
- hidden final test split,
- separate validation and blind test leaderboards.
