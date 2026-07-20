"""Streamlit demo for the emergency department wait-time and admission models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_json(filename: str):
    return json.loads((MODELS_DIR / filename).read_text(encoding="utf-8"))


@st.cache_resource
def load_artifacts():
    return {
        "nhamcs_model1": joblib.load(MODELS_DIR / "nhamcs_model1.joblib"),
        "nhamcs_model2": joblib.load(MODELS_DIR / "nhamcs_model2.joblib"),
        "nhamcs_features1": load_json("nhamcs_model1_features.json"),
        "nhamcs_features2": load_json("nhamcs_model2_features.json"),
        "nhamcs_sample1": load_json("sample_input_model1.json"),
        "nhamcs_sample2": load_json("sample_input_model2.json"),
        "nhamcs_metrics": load_json("nhamcs_metrics.json"),
        "yale_model": joblib.load(MODELS_DIR / "yale_admission_model.joblib"),
        "yale_features": load_json("yale_features.json"),
        "yale_metrics": load_json("yale_metrics.json"),
    }


def keep_tab_selected(tab_name: str) -> None:
    st.session_state["active_tab"] = tab_name


def set_one_hot(row: dict, prefix: str, selection: str) -> None:
    columns = [column for column in row if column.startswith(prefix)]
    for column in columns:
        row[column] = 0
    selected_column = f"{prefix}{selection}"
    if selected_column in row:
        row[selected_column] = 1


def make_nhamcs_row(
    sample: dict,
    age: int,
    payment_type: str,
    arrival_mode: str,
    triage_acuity: str,
    pulse: int,
    systolic_bp: int,
) -> dict:
    row = sample.copy()
    row["age_years"] = age
    set_one_hot(row, "payment_type_", payment_type)
    set_one_hot(row, "arrival_mode_", arrival_mode)
    set_one_hot(row, "triage_acuity_", triage_acuity)

    if "pulse" in row:
        row["pulse"] = pulse
        row["pulse_missing"] = 0
    if "systolic_bp" in row:
        row["systolic_bp"] = systolic_bp
        row["systolic_bp_missing"] = 0
    return row


def predict_wait_minutes(bundle: dict, features: list[str], row: dict) -> float:
    if bundle.get("target") != "log_wait_time" or bundle.get("target_transform") != "log1p":
        raise ValueError("Unexpected NHAMCS target format; expected log1p wait time.")

    frame = pd.DataFrame([row], columns=features)
    numeric_columns = bundle["numeric_columns"]
    frame[numeric_columns] = bundle["scaler"].transform(frame[numeric_columns])
    log_wait = float(bundle["model"].predict(frame)[0])
    return max(0.0, float(np.expm1(log_wait)))


def complaint_label(column: str) -> str:
    return column.removeprefix("cc_").replace("_", " ").title()


def make_yale_row(feature_info: dict, values: dict) -> pd.DataFrame:
    features = feature_info["features"]
    cc_columns = feature_info["cc_columns"]
    row = {column: (0 if column in cc_columns else pd.NA) for column in features}

    row.update(
        {
            "age": values["age"],
            "esi": values["esi"],
            "triage_vital_hr": values["heart_rate"],
            "triage_vital_sbp": values["systolic_bp"],
            "triage_vital_o2": values["oxygen_saturation"],
            "gender": "Female",
            "race": "White or Caucasian",
            "ethnicity": "Non-Hispanic",
            "insurance_status": "Commercial",
            "arrivalmode": "Car",
            "arrivalmonth": "January",
            "arrivalday": "Monday",
            "arrivalhour_bin": "11-14",
            "triage_vital_dbp": 80,
            "triage_vital_rr": 18,
            "triage_vital_o2_device": 0,
            "triage_vital_temp": 98.6,
            "n_edvisits": 0,
            "n_admissions": 0,
        }
    )
    selected_complaint = values["chief_complaint"]
    if selected_complaint in row:
        row[selected_complaint] = 1
    return pd.DataFrame([row], columns=features)


def nhamcs_metrics_table(metrics: dict) -> pd.DataFrame:
    display = metrics["display"]
    return pd.DataFrame(
        [
            {
                "Model": "Model 1",
                "R-squared": display["model1"]["r2"],
                "MAE (minutes)": display["model1"]["mae_minutes"],
                "RMSE (minutes)": display["model1"]["rmse_minutes"],
            },
            {
                "Model": "Model 2",
                "R-squared": display["model2"]["r2"],
                "MAE (minutes)": display["model2"]["mae_minutes"],
                "RMSE (minutes)": display["model2"]["rmse_minutes"],
            },
        ]
    )


def yale_metrics_table(metrics: dict) -> pd.DataFrame:
    test = metrics["test"]
    return pd.DataFrame(
        [
            {
                "Accuracy": test["accuracy"],
                "Recall": test["recall"],
                "Precision": test["precision"],
                "F1": test["f1"],
                "ROC-AUC": metrics["test_roc_auc"],
            }
        ]
    )


st.set_page_config(page_title="ED Model Explorer", page_icon="+", layout="wide")
st.title("Drivers of Emergency Department Wait Times")

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error(f"Model artifacts could not be loaded: {exc}")
    st.stop()

tab_names = ["Home", "Wait-Time Prediction", "Admission Risk", "Monitoring"]
home_tab, wait_tab, admission_tab, monitoring_tab = st.tabs(
    tab_names,
    default=st.session_state.get("active_tab", "Home"),
)

with home_tab:
    st.subheader("Project Overview")
    st.write(
        "To what extent are emergency department wait times and hospital admission risk "
        "explained by patient symptoms, clinical urgency, and triage-time factors, and do "
        "demographic factors remain associated after accounting for medical need?"
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**NHAMCS wait-time analysis**")
        st.write(
            "National emergency department survey records are used for two Linear Regression "
            "models. Model 1 uses demographics and access factors; Model 2 adds clinical "
            "urgency and vital signs."
        )
    with right:
        st.markdown("**Yale admission-risk analysis**")
        st.write(
            "Yale emergency department visits are used for Logistic Regression with 219 "
            "arrival and triage-time predictors. Admission is an outcome, not a pure measure "
            "of clinical severity."
        )
    st.warning(
        "Research demonstration only. Predictions are not medical advice and must not be used "
        "for clinical decisions."
    )

with wait_tab:
    st.subheader("NHAMCS Wait-Time Prediction")
    st.caption("Model 1 uses access factors; Model 2 also uses triage acuity and vital signs.")

    with st.form("wait_time_form"):
        col1, col2, col3 = st.columns(3)
        age = col1.number_input("Age", min_value=0, max_value=100, value=31)
        payment_type = col2.selectbox(
            "Payment type",
            [
                "Private insurance / reference",
                "Medicaid/CHIP",
                "Medicare",
                "Self-pay",
                "No charge/Charity",
                "Worker's compensation",
                "Other",
            ],
        )
        arrival_mode = col3.selectbox(
            "Arrival mode",
            [
                "Personal transportation",
                "Ambulance",
                "Public service (nonambulance)",
                "Other / reference",
            ],
        )
        triage_acuity = col1.selectbox("Triage acuity", ["Moderate / reference", "High", "Low"])
        pulse = col2.number_input("Pulse", min_value=20, max_value=250, value=90)
        systolic_bp = col3.number_input(
            "Systolic blood pressure", min_value=50, max_value=260, value=125
        )
        predict_wait = st.form_submit_button(
            "Predict wait time",
            type="primary",
            on_click=keep_tab_selected,
            args=("Wait-Time Prediction",),
        )

    if predict_wait:
        row1 = make_nhamcs_row(
            artifacts["nhamcs_sample1"],
            age,
            payment_type,
            arrival_mode,
            triage_acuity,
            pulse,
            systolic_bp,
        )
        row2 = make_nhamcs_row(
            artifacts["nhamcs_sample2"],
            age,
            payment_type,
            arrival_mode,
            triage_acuity,
            pulse,
            systolic_bp,
        )
        wait1 = predict_wait_minutes(
            artifacts["nhamcs_model1"], artifacts["nhamcs_features1"], row1
        )
        wait2 = predict_wait_minutes(
            artifacts["nhamcs_model2"], artifacts["nhamcs_features2"], row2
        )
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Model 1 predicted wait", f"{wait1:.1f} min")
        metric2.metric("Model 2 predicted wait", f"{wait2:.1f} min")
        metric3.metric("Model 2 - Model 1", f"{wait2 - wait1:+.1f} min")

    st.markdown("**Test-set metrics**")
    st.dataframe(nhamcs_metrics_table(artifacts["nhamcs_metrics"]), hide_index=True)

with admission_tab:
    st.subheader("Yale Admission Risk")
    st.caption("Admission disposition predicted from information available at arrival or triage.")
    complaint_columns = artifacts["yale_features"]["cc_columns"]
    complaint_lookup = {complaint_label(column): column for column in complaint_columns}

    with st.form("admission_form"):
        col1, col2, col3 = st.columns(3)
        yale_age = col1.number_input("Age", min_value=18, max_value=100, value=62, key="yale_age")
        esi = col2.selectbox("ESI level", [1, 2, 3, 4, 5], index=1)
        complaint_name = col3.selectbox("Chief complaint", sorted(complaint_lookup))
        heart_rate = col1.number_input("Heart rate", min_value=30, max_value=250, value=104)
        yale_sbp = col2.number_input(
            "Systolic blood pressure", min_value=50, max_value=260, value=145, key="yale_sbp"
        )
        oxygen_saturation = col3.number_input(
            "Oxygen saturation", min_value=50, max_value=100, value=96
        )
        predict_admission = st.form_submit_button(
            "Estimate admission risk",
            type="primary",
            on_click=keep_tab_selected,
            args=("Admission Risk",),
        )

    if predict_admission:
        yale_row = make_yale_row(
            artifacts["yale_features"],
            {
                "age": yale_age,
                "esi": esi,
                "chief_complaint": complaint_lookup[complaint_name],
                "heart_rate": heart_rate,
                "systolic_bp": yale_sbp,
                "oxygen_saturation": oxygen_saturation,
            },
        )
        prediction = int(artifacts["yale_model"].predict(yale_row)[0])
        probability = float(artifacts["yale_model"].predict_proba(yale_row)[0, 1])
        result1, result2 = st.columns(2)
        result1.metric("Admission probability", f"{probability:.1%}")
        result2.metric("Predicted disposition", "Admit" if prediction else "Discharge")

    st.markdown("**Test-set metrics**")
    st.dataframe(yale_metrics_table(artifacts["yale_metrics"]), hide_index=True)

with monitoring_tab:
    st.subheader("Monitoring and Governance")
    st.markdown("**Model drift**")
    st.write("Track changes in input distributions and prediction performance over time.")
    st.markdown("**COVID-era shift**")
    st.write("Evaluate NHAMCS performance separately before, during, and after COVID-era changes.")
    st.markdown("**Missing values**")
    st.write("Monitor missingness by feature and confirm that deployment inputs match training rules.")
    st.markdown("**Fairness and subgroup performance**")
    st.write("Compare errors across demographic groups without treating demographic associations as causal.")
    st.markdown("**Retraining plan**")
    st.write("Retrain after material drift, data-definition changes, or sustained performance decline.")
