"""Smoke-test the exported Yale admission-risk model artifact."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "yale_admission_model.joblib"
FEATURES_PATH = MODELS_DIR / "yale_features.json"


def make_default_input(feature_info: dict) -> pd.DataFrame:
    features = feature_info["features"]
    cc_columns = set(feature_info["cc_columns"])

    row = {}
    for column in features:
        if column in cc_columns:
            row[column] = 0
        else:
            row[column] = pd.NA

    # Streamlit-facing simple inputs.
    row["age"] = 62
    row["esi"] = 2
    row["triage_vital_hr"] = 104
    row["triage_vital_sbp"] = 145
    row["triage_vital_o2"] = 96
    row["cc_chestpain"] = 1

    # Hidden/default fields used to satisfy the full training schema.
    row["gender"] = "Female"
    row["race"] = "White or Caucasian"
    row["ethnicity"] = "Non-Hispanic"
    row["insurance_status"] = "Commercial"
    row["arrivalmode"] = "Car"
    row["arrivalmonth"] = "January"
    row["arrivalday"] = "Monday"
    row["arrivalhour_bin"] = "11-14"
    row["triage_vital_dbp"] = 85
    row["triage_vital_rr"] = 18
    row["triage_vital_o2_device"] = 0
    row["triage_vital_temp"] = 98.6
    row["n_edvisits"] = 0
    row["n_admissions"] = 0

    missing = [column for column in features if column not in row]
    if missing:
        raise ValueError(f"Sample row is missing expected feature columns: {missing}")

    return pd.DataFrame([row], columns=features)


def main() -> None:
    model = joblib.load(MODEL_PATH)
    feature_info = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    sample = make_default_input(feature_info)

    prediction = int(model.predict(sample)[0])
    admission_probability = float(model.predict_proba(sample)[0, 1])
    label = "Admit" if prediction == 1 else "Discharge"

    print(f"Admission probability: {admission_probability:.6f}")
    print(f"Prediction: {label} ({prediction})")


if __name__ == "__main__":
    main()
