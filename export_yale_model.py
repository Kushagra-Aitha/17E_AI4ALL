"""Export the baseline Yale admission-risk model for Streamlit deployment."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "yale_clean_triage.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "yale_admission_model.joblib"
FEATURES_PATH = MODELS_DIR / "yale_features.json"
METRICS_PATH = MODELS_DIR / "yale_metrics.json"
TARGET = "disposition_admit"
RANDOM_STATE = 42


def build_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    cc_columns: list[str],
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    cc_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
            ("cc", cc_pipeline, cc_columns),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def metric_row(split: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "specificity": round(float(specificity), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' was not found.")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    cc_columns = [column for column in X.columns if column.startswith("cc_")]
    numeric_columns = [
        column
        for column in X.select_dtypes(include=[np.number]).columns
        if column not in cc_columns
    ]
    categorical_columns = list(X.select_dtypes(include=["object", "category"]).columns)

    assigned_columns = set(cc_columns + numeric_columns + categorical_columns)
    unassigned_columns = [column for column in X.columns if column not in assigned_columns]
    if unassigned_columns:
        raise ValueError(f"Columns were not assigned to a preprocessing group: {unassigned_columns}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline(numeric_columns, categorical_columns, cc_columns)
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Baseline Logistic Regression",
        "target": TARGET,
        "dataset_shape": list(df.shape),
        "predictor_count": X.shape[1],
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "column_groups": {
            "numeric": numeric_columns,
            "categorical": categorical_columns,
            "chief_complaint": cc_columns,
        },
        "train": metric_row("train", y_train, y_train_pred),
        "test": metric_row("test", y_test, y_test_pred),
        "test_roc_auc": round(float(roc_auc_score(y_test, y_test_proba)), 6),
    }

    feature_info = {
        "target": TARGET,
        "features": list(X.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "cc_columns": cc_columns,
        "default_note": (
            "For Streamlit inputs, initialize all feature columns, fill hidden fields "
            "with safe defaults, and set the selected cc_* column to 1."
        ),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_info, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved features: {FEATURES_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Model size: {MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB")
    print(
        "Test metrics: "
        f"accuracy={metrics['test']['accuracy']:.6f}, "
        f"recall={metrics['test']['recall']:.6f}, "
        f"precision={metrics['test']['precision']:.6f}, "
        f"roc_auc={metrics['test_roc_auc']:.6f}"
    )


if __name__ == "__main__":
    main()
