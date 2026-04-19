# Submission Format

Each submission must be a `.zip` file containing exactly two files at the archive root:

- `submission.json`
- `predictions.npz`

No nested directories.

## `submission.json`

Required keys:

```json
{
  "task_name": "turbulent_radiative_layer_2D_1step",
  "model_name": "My FNO Baseline",
  "team_name": "Example Lab",
  "method_name": "Fourier Neural Operator",
  "framework": "PyTorch",
  "authors": ["Alice Example", "Bob Example"],
  "contact_email": "alice@example.com",
  "description": "Short description of the method.",
  "is_external_data_used": false
}
```

Required rules:

- `task_name` must equal `turbulent_radiative_layer_2D_1step`
- `model_name` must be a non-empty string
- `team_name` must be a non-empty string
- `method_name` must be a non-empty string
- `authors` must be a non-empty list of strings
- `contact_email` must be a non-empty string
- `is_external_data_used` must be a boolean

Optional keys:

- `paper_url`
- `code_url`
- `hf_model_url`

## `predictions.npz`

Required arrays:

- `sample_ids`
- `predictions`

### `sample_ids`

- type: string array
- shape: `(N,)`
- values must match the benchmark hidden test identifiers exactly and in order

Example:

```python
sample_ids = np.array(["sample_000000", "sample_000001"])
```

### `predictions`

- type: numeric array
- recommended dtype: `float32`
- required shape: `(N, 1, 128, 384, 4)`

Axis meaning:

- `N`: number of test samples
- `1`: one output step
- `128`: spatial dimension `Lx`
- `384`: spatial dimension `Ly`
- `4`: fields in fixed order

Field order:

1. `density`
2. `pressure`
3. `velocity_x`
4. `velocity_y`

Validation rules:

- must be finite numeric values,
- no NaN,
- no Inf,
- first dimension must match `len(sample_ids)`,
- shape must match the hidden test split size.

## Example save code

```python
import json
import numpy as np

submission = {
    "task_name": "turbulent_radiative_layer_2D_1step",
    "model_name": "My FNO Baseline",
    "team_name": "Example Lab",
    "method_name": "Fourier Neural Operator",
    "framework": "PyTorch",
    "authors": ["Alice Example"],
    "contact_email": "alice@example.com",
    "description": "1-step forecast baseline on The Well.",
    "is_external_data_used": False,
}

with open("submission.json", "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2)

np.savez_compressed(
    "predictions.npz",
    sample_ids=np.asarray(sample_ids),
    predictions=np.asarray(predictions, dtype=np.float32),
)
```

Then create the zip:

```bash
zip submission.zip submission.json predictions.npz
```
