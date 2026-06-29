"""Fit and evaluate a triage-time Logistic Regression model for Yale admission risk."""

from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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
RESULTS_PATH = BASE_DIR / "yale_logistic_results.txt"
TARGET = "disposition_admit"
RANDOM_STATE = 42


class Tee:
    """Write printed output to both the terminal and the results file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def clean_feature_name(name):
    """Remove ColumnTransformer prefixes while preserving one-hot category names."""
    for prefix in ("numeric__", "categorical__", "cc__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def print_class_balance(label, values):
    counts = values.value_counts().sort_index()
    percentages = values.value_counts(normalize=True).sort_index().mul(100)
    print(f"{label} class balance:")
    for value in counts.index:
        print(f"  {int(value)}: {counts[value]:,} ({percentages[value]:.2f}%)")


def run_analysis():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' was not found.")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    print("Yale triage-time Logistic Regression")
    print("======================================")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of predictors: {X.shape[1]}")
    print_class_balance("Full dataset", y)

    missing_counts = X.isna().sum().sort_values(ascending=False)
    missing_percentages = X.isna().mean().mul(100).sort_values(ascending=False)
    top_missing = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_percent": missing_percentages,
        }
    ).head(15)
    print("\nTop missing predictor columns:")
    print(top_missing.to_string(formatters={"missing_percent": "{:.2f}%".format}))

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

    print("\nColumn groups:")
    print(f"  Chief complaint (cc_*) columns: {len(cc_columns)}")
    print(f"  Numeric columns: {len(numeric_columns)}")
    print(f"  Categorical columns: {len(categorical_columns)}")

    # A stratified split preserves the admission/discharge ratio in both datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print_class_balance("Train", y_train)
    print_class_balance("Test", y_test)

    # Predictor rows are retained and missing values are imputed because dropping every
    # row with any missing predictor would discard substantial data and could bias the sample.
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

    # Logistic Regression fits a binary outcome such as admission (1) versus discharge (0).
    # Balanced class weights reduce the tendency to favor the more common discharge class.
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("\nFitting Logistic Regression...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)
    matrix = confusion_matrix(y_test, y_pred)

    print("\nEvaluation metrics:")
    print(f"  Accuracy: {accuracy:.6f}")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall: {recall:.6f}")
    print(f"  F1-score: {f1:.6f}")
    print(f"  ROC-AUC: {roc_auc:.6f}")

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_names = [clean_feature_name(name) for name in feature_names]
    coefficients = pipeline.named_steps["model"].coef_[0]

    if len(feature_names) != len(coefficients):
        raise RuntimeError(
            f"Feature-name count ({len(feature_names)}) does not match coefficient count "
            f"({len(coefficients)})."
        )

    coefficient_table = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    ).sort_values("coefficient", ascending=False)

    print("\nTop 20 positive coefficients (higher admission odds):")
    print(coefficient_table.head(20).to_string(index=False, float_format="{:.6f}".format))

    print("\nTop 20 negative coefficients (lower admission odds):")
    print(
        coefficient_table.tail(20)
        .sort_values("coefficient", ascending=True)
        .to_string(index=False, float_format="{:.6f}".format)
    )

    # Coefficients describe adjusted associations with admission disposition. They do
    # not establish that a feature causes admission or directly measures clinical severity.
    print("\nInterpretation note:")
    print(
        "Positive coefficients are associated with higher modeled admission odds and "
        "negative coefficients with lower modeled admission odds, holding other model "
        "features constant. These are associations, not causal effects. Admission is an "
        "ED disposition influenced by clinical and system factors, not a pure measure of severity."
    )


def main():
    with RESULTS_PATH.open("w", encoding="utf-8") as results_file:
        tee = Tee(sys.stdout, results_file)
        with redirect_stdout(tee):
            run_analysis()


if __name__ == "__main__":
    main()
