TITLE = "# The Well Leaderboard MVP"

INTRO_TEXT = """
Benchmark task: `turbulent_radiative_layer_2D_1step`

This leaderboard evaluates one-step forecasting on The Well dataset
`turbulent_radiative_layer_2D` using:

- `n_steps_input = 4`
- `n_steps_output = 1`
- primary metric: `avg_vrmse`

Lower scores are better.
"""

SUBMISSION_TEXT = """
## Submission format

Upload a `.zip` file containing exactly:

- `submission.json`
- `predictions.npz`

Expected tensor shape in `predictions.npz`:

`(N, 1, 128, 384, 4)`

Field order:

1. `density`
2. `pressure`
3. `velocity_x`
4. `velocity_y`
"""

ABOUT_TEXT = """
## Benchmark overview

This is an MVP benchmark for a single The Well task:
`turbulent_radiative_layer_2D_1step`.

The leaderboard Space is public. The evaluator Space is private and holds
the evaluation logic and hidden targets.

Submissions are validated and then scored with VRMSE. The public ranking uses
the average VRMSE across fields.

## Notes

- Lower `avg_vrmse` is better.
- Invalid submissions are recorded with a failure message.
- This benchmark is intended to validate protocol and infrastructure first.
"""
