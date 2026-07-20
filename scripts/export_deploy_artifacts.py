#!/usr/bin/env python3
"""
Export Streamlit-ready NHAMCS deployment artifacts.

Writes to models/:
  nhamcs_model1.joblib          — dict with model, scaler, feature_columns, ...
  nhamcs_model2.joblib
  nhamcs_model1_features.json   — ordered feature column list
  nhamcs_model2_features.json
  sample_input_model1.json      — one encoded row that predicts successfully
  sample_input_model2.json
  metrics.json                  — test R² / MAE / RMSE for display

Prediction contract (Streamlit):
  bundle = joblib.load("models/nhamcs_model2.joblib")
  X = pd.DataFrame([row])[bundle["feature_columns"]]
  X[bundle["numeric_columns"]] = bundle["scaler"].transform(X[bundle["numeric_columns"]])
  log_wait = bundle["model"].predict(X)[0]
  wait_min = float(np.expm1(log_wait))
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

TARGET = "log_wait_time"
DROP_COLS = ["wait_time_min", "log_wait_time"]
NUMERIC_M1 = ["age_years", "survey_year"]
NUMERIC_M2 = ["age_years", "survey_year", "pulse", "systolic_bp"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    y_min = np.expm1(y_true)
    y_pred_min = np.expm1(y_pred)
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 6),
        "mae_log": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "rmse_log": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 6),
        "mae_minutes": round(float(mean_absolute_error(y_min, y_pred_min)), 2),
        "rmse_minutes": round(float(mean_squared_error(y_min, y_pred_min) ** 0.5), 2),
    }


def train_bundle(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    model_name: str,
) -> tuple[dict, dict, dict]:
    """Fit model+scaler; return (joblib_bundle, metrics, sample_row)."""
    feature_columns = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    model = LinearRegression().fit(X_train, y_train)
    metrics = compute_metrics(y_test, model.predict(X_test))

    # Prefer a typical adult walk-in row for Streamlit demos
    sample_pool = X.loc[X_test.index]
    adult = sample_pool[
        (sample_pool["age_years"] >= 18) & (sample_pool["age_years"] <= 65)
    ]
    if "arrival_mode_Ambulance" in adult.columns:
        preferred = adult[adult["arrival_mode_Ambulance"] == 0]
        sample_pool = preferred if len(preferred) else adult
    elif len(adult):
        sample_pool = adult
    sample_idx = sample_pool.index[0]
    sample_row = {col: _to_jsonable(X.loc[sample_idx, col]) for col in feature_columns}

    # Sanity-check: same pipeline Streamlit should use
    X_sample = pd.DataFrame([sample_row])[feature_columns]
    X_sample[numeric_cols] = scaler.transform(X_sample[numeric_cols])
    log_pred = float(model.predict(X_sample)[0])
    wait_pred = float(np.expm1(log_pred))

    bundle = {
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "numeric_columns": list(numeric_cols),
        "target": TARGET,
        "target_transform": "log1p",  # predict log_wait; back-transform with expm1
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "metrics_test": metrics,
        "sample_prediction": {
            "log_wait_time": round(log_pred, 6),
            "wait_time_min": round(wait_pred, 2),
        },
    }
    return bundle, metrics, sample_row


def _to_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading encoded datasets...")
    m1 = pd.read_csv(PROCESSED / "nhamcs_model1_encoded.csv")
    m2 = pd.read_csv(PROCESSED / "nhamcs_model2_encoded.csv")
    y = m1[TARGET]
    X1 = m1.drop(columns=DROP_COLS)
    X2 = m2.drop(columns=DROP_COLS)
    print(f"  Model 1: {X1.shape[1]} features, {len(y):,} rows")
    print(f"  Model 2: {X2.shape[1]} features")

    print("\nTraining Model 1...")
    bundle1, metrics1, sample1 = train_bundle(X1, y, NUMERIC_M1, "nhamcs_model1")
    print("Training Model 2...")
    bundle2, metrics2, sample2 = train_bundle(X2, y, NUMERIC_M2, "nhamcs_model2")

    joblib.dump(bundle1, MODELS_DIR / "nhamcs_model1.joblib")
    joblib.dump(bundle2, MODELS_DIR / "nhamcs_model2.joblib")

    (MODELS_DIR / "nhamcs_model1_features.json").write_text(
        json.dumps(bundle1["feature_columns"], indent=2)
    )
    (MODELS_DIR / "nhamcs_model2_features.json").write_text(
        json.dumps(bundle2["feature_columns"], indent=2)
    )

    (MODELS_DIR / "sample_input_model1.json").write_text(
        json.dumps(sample1, indent=2)
    )
    (MODELS_DIR / "sample_input_model2.json").write_text(
        json.dumps(sample2, indent=2)
    )

    metrics_out = {
        "model1_test": metrics1,
        "model2_test": metrics2,
        "display": {
            "model1": {
                "r2": metrics1["r2"],
                "mae_minutes": metrics1["mae_minutes"],
                "rmse_minutes": metrics1["rmse_minutes"],
            },
            "model2": {
                "r2": metrics2["r2"],
                "mae_minutes": metrics2["mae_minutes"],
                "rmse_minutes": metrics2["rmse_minutes"],
            },
        },
        "notes": (
            "Models predict log(wait_time_min + 1). "
            "Convert predictions with np.expm1(pred) to get minutes. "
            "Display metrics use minutes MAE/RMSE on the test set."
        ),
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    # End-to-end verification using saved joblibs
    print("\nVerifying saved joblibs with sample inputs...")
    for name, sample_path in [
        ("nhamcs_model1.joblib", "sample_input_model1.json"),
        ("nhamcs_model2.joblib", "sample_input_model2.json"),
    ]:
        b = joblib.load(MODELS_DIR / name)
        row = json.loads((MODELS_DIR / sample_path).read_text())
        X = pd.DataFrame([row])[b["feature_columns"]]
        X[b["numeric_columns"]] = b["scaler"].transform(X[b["numeric_columns"]])
        log_pred = float(b["model"].predict(X)[0])
        wait_pred = float(np.expm1(log_pred))
        print(
            f"  {name}: {len(b['feature_columns'])} features → "
            f"log={log_pred:.4f}, wait≈{wait_pred:.1f} min"
        )

    print("\n=== Deploy artifacts written to models/ ===")
    for p in sorted(MODELS_DIR.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
