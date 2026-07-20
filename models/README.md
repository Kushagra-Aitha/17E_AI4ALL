# NHAMCS Streamlit Deploy Artifacts

| File | Purpose |
|---|---|
| `nhamcs_model1.joblib` | Demographics + access model bundle |
| `nhamcs_model2.joblib` | + clinical features model bundle |
| `nhamcs_model1_features.json` | Ordered feature list (41 cols) |
| `nhamcs_model2_features.json` | Ordered feature list (49 cols) |
| `sample_input_model1.json` | One encoded row that predicts successfully |
| `sample_input_model2.json` | One encoded row that predicts successfully |
| `metrics.json` | Test R² / MAE / RMSE for display |

## Joblib bundle keys

- `model` — `sklearn.linear_model.LinearRegression`
- `scaler` — `StandardScaler` fitted on `numeric_columns` only
- `feature_columns` — ordered list; pass columns in this order
- `numeric_columns` — columns that must be scaled before predict
- `target` — `"log_wait_time"`
- `metrics_test` — test-set metrics

## Prediction (Wait-Time Prediction tab)

```python
import json
import joblib
import numpy as np
import pandas as pd

bundle = joblib.load("models/nhamcs_model2.joblib")
row = json.load(open("models/sample_input_model2.json"))

X = pd.DataFrame([row])[bundle["feature_columns"]]
X[bundle["numeric_columns"]] = bundle["scaler"].transform(X[bundle["numeric_columns"]])

log_wait = bundle["model"].predict(X)[0]
wait_min = float(np.expm1(log_wait))  # minutes
```

## Display metrics (test set)

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Model 1 | 0.074 | 31.5 min | 57.6 min |
| Model 2 | 0.078 | 31.4 min | 57.5 min |

Regenerate with:

```bash
python scripts/export_deploy_artifacts.py
```
