"""NHAMCS Wait-Time Prediction tab — hospital CSV upload + batch predictions."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

TEMPLATE_PATH = Path(__file__).resolve().parent / "sample_wait_time_upload.csv"
DEMO_PATH = Path(__file__).resolve().parent / "demo_wait_time_upload.csv"

# Friendly CSV columns hospitals fill in (human-readable values).
FRIENDLY_COLUMNS = [
    "patient_id",
    "age_years",
    "survey_year",
    "sex",
    "race",
    "ethnicity",
    "residence",
    "visit_month",
    "visit_day_of_week",
    "payment_type",
    "arrival_mode",
    "triage_acuity",
    "pulse",
    "systolic_bp",
    "seen_72h",
]

# Allowed values (case-insensitive match after strip). Reference = all one-hots 0.
SEX_VALUES = {"female", "male"}
RACE_VALUES = {
    "white only",
    "black/african american only",
    "asian only",
    "american indian/alaska native only",
    "native hawaiian/oth pac isl only",
    "more than one race reported",
}
ETHNICITY_VALUES = {"not hispanic or latino", "hispanic or latino"}
RESIDENCE_VALUES = {
    "private residence",
    "homeless",
    "homeless/homeless shelter",
    "nursing home",
    "other",
    "other institution",
    "other residence",
}
MONTH_VALUES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
DAY_VALUES = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
PAYMENT_VALUES = {
    "private insurance",
    "medicaid/chip",
    "medicare",
    "self-pay",
    "no charge/charity",
    "worker's compensation",
    "other",
}
ARRIVAL_VALUES = {
    "walk-in/other",
    "ambulance",
    "personal transportation",
    "public service (nonambulance)",
}
TRIAGE_VALUES = {"medium", "high", "low"}
SEEN_72H_VALUES = {"yes", "no", "missing"}

RACE_TO_COL = {
    "american indian/alaska native only": "race_American Indian/Alaska Native Only",
    "asian only": "race_Asian Only",
    "black/african american only": "race_Black/African American Only",
    "more than one race reported": "race_More than one race reported",
    "native hawaiian/oth pac isl only": "race_Native Hawaiian/Oth Pac Isl Only",
}
RESIDENCE_TO_COL = {
    "homeless": "residence_Homeless",
    "homeless/homeless shelter": "residence_Homeless/homeless shelter",
    "nursing home": "residence_Nursing home",
    "other": "residence_Other",
    "other institution": "residence_Other institution",
    "other residence": "residence_Other residence",
}
PAYMENT_TO_COL = {
    "medicaid/chip": "payment_type_Medicaid/CHIP",
    "medicare": "payment_type_Medicare",
    "no charge/charity": "payment_type_No charge/Charity",
    "other": "payment_type_Other",
    "self-pay": "payment_type_Self-pay",
    "worker's compensation": "payment_type_Worker's compensation",
}
ARRIVAL_TO_COL = {
    "ambulance": "arrival_mode_Ambulance",
    "personal transportation": "arrival_mode_Personal transportation",
    "public service (nonambulance)": "arrival_mode_Public service (nonambulance)",
}


def _norm(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _title_month(value: str) -> str:
    return value.strip().capitalize()


def _title_day(value: str) -> str:
    return value.strip().capitalize()


def blank_encoded_row(feature_columns: list[str]) -> dict:
    return {col: 0 for col in feature_columns}


def encode_friendly_row(raw: dict, feature_columns: list[str]) -> dict:
    """Convert one hospital-friendly row into model feature space."""
    row = blank_encoded_row(feature_columns)

    age = float(raw.get("age_years", 0) or 0)
    year = float(raw.get("survey_year", 2015) or 2015)
    row["age_years"] = age
    row["survey_year"] = year

    sex = _norm(raw.get("sex", "female"))
    if sex == "male" and "sex_Male" in row:
        row["sex_Male"] = 1

    race = _norm(raw.get("race", "white only"))
    race_col = RACE_TO_COL.get(race)
    if race_col and race_col in row:
        row[race_col] = 1

    ethnicity = _norm(raw.get("ethnicity", "not hispanic or latino"))
    if ethnicity == "hispanic or latino" and "ethnicity_Hispanic or Latino" in row:
        row["ethnicity_Hispanic or Latino"] = 1

    residence = _norm(raw.get("residence", "private residence"))
    res_col = RESIDENCE_TO_COL.get(residence)
    if res_col and res_col in row:
        row[res_col] = 1

    month = _norm(raw.get("visit_month", "july"))
    if month != "july":
        month_col = f"visit_month_{_title_month(month)}"
        if month_col in row:
            row[month_col] = 1

    day = _norm(raw.get("visit_day_of_week", "wednesday"))
    if day != "wednesday":
        day_col = f"visit_day_of_week_{_title_day(day)}"
        if day_col in row:
            row[day_col] = 1

    payment = _norm(raw.get("payment_type", "private insurance"))
    pay_col = PAYMENT_TO_COL.get(payment)
    if pay_col and pay_col in row:
        row[pay_col] = 1

    arrival = _norm(raw.get("arrival_mode", "walk-in/other"))
    arr_col = ARRIVAL_TO_COL.get(arrival)
    if arr_col and arr_col in row:
        row[arr_col] = 1

    # Model 2 clinical fields
    if "triage_acuity_High" in row:
        triage = _norm(raw.get("triage_acuity", "medium"))
        if triage == "high":
            row["triage_acuity_High"] = 1
        elif triage == "low":
            row["triage_acuity_Low"] = 1

    if "pulse" in row:
        pulse_raw = raw.get("pulse")
        if pulse_raw is None or (isinstance(pulse_raw, float) and np.isnan(pulse_raw)) or str(pulse_raw).strip() == "":
            row["pulse"] = 88.0  # training median fallback
            row["pulse_missing"] = 1
        else:
            row["pulse"] = float(pulse_raw)
            row["pulse_missing"] = 0

    if "systolic_bp" in row:
        bp_raw = raw.get("systolic_bp")
        if bp_raw is None or (isinstance(bp_raw, float) and np.isnan(bp_raw)) or str(bp_raw).strip() == "":
            row["systolic_bp"] = 131.0
            row["systolic_bp_missing"] = 1
        else:
            row["systolic_bp"] = float(bp_raw)
            row["systolic_bp_missing"] = 0

    if "seen_72h_Yes" in row:
        seen = _norm(raw.get("seen_72h", "no"))
        if seen == "yes":
            row["seen_72h_Yes"] = 1
            row["seen_72h_missing"] = 0
        elif seen == "missing":
            row["seen_72h_Yes"] = 0
            row["seen_72h_missing"] = 1
        else:
            row["seen_72h_Yes"] = 0
            row["seen_72h_missing"] = 0

    return row


def looks_encoded(df: pd.DataFrame, feature_columns: list[str]) -> bool:
    return all(col in df.columns for col in feature_columns)


def validate_friendly(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = ["age_years", "survey_year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return errors

    checks = [
        ("sex", SEX_VALUES),
        ("race", RACE_VALUES),
        ("ethnicity", ETHNICITY_VALUES),
        ("residence", RESIDENCE_VALUES),
        ("visit_month", MONTH_VALUES),
        ("visit_day_of_week", DAY_VALUES),
        ("payment_type", PAYMENT_VALUES),
        ("arrival_mode", ARRIVAL_VALUES),
        ("triage_acuity", TRIAGE_VALUES),
        ("seen_72h", SEEN_72H_VALUES),
    ]
    for col, allowed in checks:
        if col not in df.columns:
            continue
        bad = sorted({_norm(v) for v in df[col].dropna().unique() if _norm(v) not in allowed})
        if bad:
            errors.append(f"Invalid `{col}` values: {bad[:5]}{'…' if len(bad) > 5 else ''}")
    return errors


def encode_dataframe(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if looks_encoded(df, feature_columns):
        return df[feature_columns].copy()

    rows = [encode_friendly_row(record, feature_columns) for record in df.to_dict(orient="records")]
    return pd.DataFrame(rows, columns=feature_columns)


def predict_batch(bundle: dict, features: list[str], encoded: pd.DataFrame) -> np.ndarray:
    if bundle.get("target") != "log_wait_time" or bundle.get("target_transform") != "log1p":
        raise ValueError("Unexpected NHAMCS target format; expected log1p wait time.")

    frame = encoded[features].copy()
    numeric_columns = bundle["numeric_columns"]
    if numeric_columns:
        frame[numeric_columns] = bundle["scaler"].transform(frame[numeric_columns])
    log_wait = bundle["model"].predict(frame)
    return np.maximum(0.0, np.expm1(log_wait))


def nhamcs_metrics_table(metrics: dict) -> pd.DataFrame:
    display = metrics["display"]
    return pd.DataFrame(
        [
            {
                "Model": "Model 1 (demographics + access)",
                "R²": display["model1"]["r2"],
                "MAE (min)": display["model1"]["mae_minutes"],
                "RMSE (min)": display["model1"]["rmse_minutes"],
            },
            {
                "Model": "Model 2 (+ clinical)",
                "R²": display["model2"]["r2"],
                "MAE (min)": display["model2"]["mae_minutes"],
                "RMSE (min)": display["model2"]["rmse_minutes"],
            },
        ]
    )


def build_template_csv() -> str:
    if TEMPLATE_PATH.exists():
        return TEMPLATE_PATH.read_text(encoding="utf-8")
    sample = pd.DataFrame(
        [
            {
                "patient_id": "P001",
                "age_years": 31,
                "survey_year": 2018,
                "sex": "Female",
                "race": "White Only",
                "ethnicity": "Not Hispanic or Latino",
                "residence": "Private residence",
                "visit_month": "July",
                "visit_day_of_week": "Wednesday",
                "payment_type": "Private insurance",
                "arrival_mode": "Walk-in/Other",
                "triage_acuity": "Medium",
                "pulse": 88,
                "systolic_bp": 125,
                "seen_72h": "No",
            },
            {
                "patient_id": "P002",
                "age_years": 54,
                "survey_year": 2019,
                "sex": "Male",
                "race": "Black/African American Only",
                "ethnicity": "Not Hispanic or Latino",
                "residence": "Private residence",
                "visit_month": "January",
                "visit_day_of_week": "Monday",
                "payment_type": "Medicaid/CHIP",
                "arrival_mode": "Ambulance",
                "triage_acuity": "High",
                "pulse": 102,
                "systolic_bp": 148,
                "seen_72h": "No",
            },
            {
                "patient_id": "P003",
                "age_years": 27,
                "survey_year": 2020,
                "sex": "Female",
                "race": "White Only",
                "ethnicity": "Hispanic or Latino",
                "residence": "Private residence",
                "visit_month": "September",
                "visit_day_of_week": "Saturday",
                "payment_type": "Self-pay",
                "arrival_mode": "Personal transportation",
                "triage_acuity": "Low",
                "pulse": 78,
                "systolic_bp": 118,
                "seen_72h": "Yes",
            },
        ]
    )
    return sample.to_csv(index=False)


def build_template_bytes() -> bytes:
    """Return template CSV as UTF-8 bytes (more reliable for st.download_button)."""
    text = build_template_csv()
    if not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")


def load_demo_dataframe() -> pd.DataFrame:
    if DEMO_PATH.exists():
        return pd.read_csv(DEMO_PATH)
    return pd.read_csv(StringIO(build_template_csv()))


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .wait-hero {
            padding: 1.1rem 1.25rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #0f2744 0%, #163a5f 55%, #1a4d6d 100%);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }
        .wait-hero h3 {
            margin: 0 0 0.35rem 0;
            color: #f4f7fb;
            font-weight: 650;
            letter-spacing: -0.01em;
        }
        .wait-hero p {
            margin: 0;
            color: #c5d4e4;
            font-size: 0.95rem;
            line-height: 1.45;
        }
        .wait-chip {
            display: inline-block;
            margin-top: 0.75rem;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            background: rgba(46, 196, 182, 0.15);
            color: #9eefe6;
            font-size: 0.78rem;
            border: 1px solid rgba(46, 196, 182, 0.35);
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 10px;
            padding: 0.55rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def wait_band_label(minutes: float) -> str:
    if minutes < 20:
        return "Shorter than typical"
    if minutes < 40:
        return "Around a typical ED wait"
    return "Longer than typical"


def _draw_compare_bars(labels: list[str], values: list[float], title: str):
    """Matplotlib bars — reliable in Streamlit even when Vega charts fail."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = ["#5B7C99", "#2EC4B6"][: len(values)]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_ylabel("Minutes")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 10)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def render_single_patient_charts(
    artifacts: dict,
    patient: dict,
    wait1: float,
    wait2: float,
    features1: list[str],
    features2: list[str],
) -> None:
    """Simple interactive charts for one patient estimate."""
    try:
        st.markdown("---")
        st.subheader("Visualize this estimate")
        st.write(
            f"Main estimate **{wait2:.0f} min** · Model 1 (access only) **{wait1:.0f} min**"
        )

        left, right = st.columns(2)
        with left:
            st.caption("Model 1 vs Model 2")
            fig1 = _draw_compare_bars(
                ["Access only", "Main estimate"],
                [float(wait1), float(wait2)],
                "Predicted wait (minutes)",
            )
            st.pyplot(fig1, clear_figure=True)

        with right:
            st.caption("Where this wait sits (0–60 min scale)")
            capped = min(max(float(wait2), 0.0), 60.0)
            st.progress(capped / 60.0)
            st.write(f"**{wait2:.0f} minutes** on a 0–60 scale")
            band = wait_band_label(wait2)
            if wait2 < 20:
                st.success(band)
            elif wait2 < 40:
                st.warning(band)
            else:
                st.error(band)
            st.caption("Rough bands: under 20 · 20–40 · over 40 minutes.")

        st.markdown("##### What if something changed?")
        st.caption("Change arrival or triage — the chart updates right away.")
        arrival_options = [
            "Walk-in/Other",
            "Ambulance",
            "Personal transportation",
            "Public service (nonambulance)",
        ]
        triage_options = ["High", "Medium", "Low"]
        s1, s2 = st.columns(2)
        with s1:
            alt_arrival = st.selectbox(
                "Arrival mode",
                arrival_options,
                index=(
                    arrival_options.index(patient["arrival_mode"])
                    if patient["arrival_mode"] in arrival_options
                    else 0
                ),
                key="viz_arrival_whatif",
            )
        with s2:
            alt_triage = st.selectbox(
                "Triage acuity",
                triage_options,
                index=(
                    triage_options.index(patient["triage_acuity"])
                    if patient["triage_acuity"] in triage_options
                    else 1
                ),
                key="viz_triage_whatif",
            )

        scenario = patient.copy()
        scenario["arrival_mode"] = alt_arrival
        scenario["triage_acuity"] = alt_triage
        try:
            enc2 = encode_dataframe(pd.DataFrame([scenario]), features2)
            scenario_wait = float(
                predict_batch(artifacts["nhamcs_model2"], features2, enc2)[0]
            )
        except Exception as exc:
            st.warning(f"What-if prediction failed: {exc}")
            scenario_wait = wait2

        fig2 = _draw_compare_bars(
            ["Your inputs", "What-if"],
            [float(wait2), float(scenario_wait)],
            "What-if comparison (Model 2)",
        )
        st.pyplot(fig2, clear_figure=True)

        delta = scenario_wait - wait2
        if abs(delta) < 0.5:
            st.info("This what-if is about the **same** as your original estimate.")
        elif delta < 0:
            st.info(
                f"What-if is **{abs(delta):.0f} min shorter** "
                f"({scenario_wait:.0f} vs {wait2:.0f} min)."
            )
        else:
            st.info(
                f"What-if is **{delta:.0f} min longer** "
                f"({scenario_wait:.0f} vs {wait2:.0f} min)."
            )
    except Exception as exc:
        st.error("Could not draw charts.")
        st.exception(exc)


def render_wait_time_tab(artifacts: dict, keep_tab_selected) -> None:
    """Single-patient wait estimate (primary) + optional hospital CSV batch."""
    _inject_styles()

    st.markdown(
        """
        <div class="wait-hero">
          <h3>How long might this patient wait?</h3>
          <p>
            Enter one patient’s arrival details. The model estimates minutes until
            first provider contact using NHAMCS survey patterns (research demo only).
          </p>
          <span class="wait-chip">Not medical advice · associations, not guarantees</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    features1 = artifacts["nhamcs_features1"]
    features2 = artifacts["nhamcs_features2"]

    # ------------------------------------------------------------------
    # Single patient form
    # ------------------------------------------------------------------
    st.markdown("#### Patient information")
    with st.form("single_patient_wait_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (years)", min_value=0, max_value=100, value=35)
        sex = c2.selectbox("Sex", ["Female", "Male"])
        survey_year = c3.number_input("Visit year", min_value=2007, max_value=2022, value=2019)

        c1, c2, c3 = st.columns(3)
        race = c1.selectbox(
            "Race",
            [
                "White Only",
                "Black/African American Only",
                "Asian Only",
                "American Indian/Alaska Native Only",
                "Native Hawaiian/Oth Pac Isl Only",
                "More than one race reported",
            ],
        )
        ethnicity = c2.selectbox(
            "Ethnicity",
            ["Not Hispanic or Latino", "Hispanic or Latino"],
        )
        payment_type = c3.selectbox(
            "Payment / insurance",
            [
                "Private insurance",
                "Medicaid/CHIP",
                "Medicare",
                "Self-pay",
                "No charge/Charity",
                "Worker's compensation",
                "Other",
            ],
        )

        c1, c2, c3 = st.columns(3)
        arrival_mode = c1.selectbox(
            "Arrival mode",
            [
                "Walk-in/Other",
                "Ambulance",
                "Personal transportation",
                "Public service (nonambulance)",
            ],
        )
        triage_acuity = c2.selectbox("Triage acuity", ["Medium", "High", "Low"])
        residence = c3.selectbox(
            "Residence",
            [
                "Private residence",
                "Nursing home",
                "Homeless",
                "Homeless/homeless shelter",
                "Other",
                "Other institution",
                "Other residence",
            ],
        )

        c1, c2, c3 = st.columns(3)
        visit_month = c1.selectbox(
            "Visit month",
            [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
            ],
            index=6,
        )
        visit_day = c2.selectbox(
            "Day of week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=2,
        )
        seen_72h = c3.selectbox("Seen in ED in last 72 hours?", ["No", "Yes", "Missing"])

        c1, c2, c3 = st.columns(3)
        pulse = c1.number_input("Pulse", min_value=20, max_value=250, value=88)
        systolic_bp = c2.number_input("Systolic blood pressure", min_value=50, max_value=260, value=125)
        c3.caption("Pulse & BP used by Model 2 (clinical).")

        submitted = st.form_submit_button(
            "Estimate wait time",
            type="primary",
            width="stretch",
            on_click=keep_tab_selected,
            args=("Wait-Time Prediction",),
        )

    if submitted:
        patient = {
            "patient_id": "single",
            "age_years": age,
            "survey_year": survey_year,
            "sex": sex,
            "race": race,
            "ethnicity": ethnicity,
            "residence": residence,
            "visit_month": visit_month,
            "visit_day_of_week": visit_day,
            "payment_type": payment_type,
            "arrival_mode": arrival_mode,
            "triage_acuity": triage_acuity,
            "pulse": pulse,
            "systolic_bp": systolic_bp,
            "seen_72h": seen_72h,
        }
        try:
            enc1 = encode_dataframe(pd.DataFrame([patient]), features1)
            enc2 = encode_dataframe(pd.DataFrame([patient]), features2)
            wait1 = float(predict_batch(artifacts["nhamcs_model1"], features1, enc1)[0])
            wait2 = float(predict_batch(artifacts["nhamcs_model2"], features2, enc2)[0])
            st.session_state["single_wait"] = {
                "patient": patient,
                "wait1": wait1,
                "wait2": wait2,
            }
        except Exception as exc:
            st.error(f"Could not estimate wait time: {exc}")

    single = st.session_state.get("single_wait")
    if single:
        wait1 = single["wait1"]
        wait2 = single["wait2"]
        st.markdown("---")
        st.markdown("#### Estimated wait")

        st.success(
            f"### About **{wait2:.0f} minutes**\n\n"
            "Estimated time until first provider contact "
            "(using triage acuity and vitals, plus demographics & access)."
        )
        st.caption(
            "Research estimate from national ED survey data — not a guarantee of real wait time, "
            "and not medical advice."
        )

        render_single_patient_charts(
            artifacts,
            single["patient"],
            wait1,
            wait2,
            features1,
            features2,
        )

        with st.expander("Why might you see two different times? (research comparison)"):
            st.markdown(
                """
**Simple takeaway**

| Number | What it means for you |
|---|---|
| **Main estimate (Model 2)** — shown above | Best guess of wait using **everything we know**: who the patient is *and* how urgent they look (triage + vitals). **Use this one.** |
| **Model 1** — access/demographics only | What the wait might look like if we **ignored** triage and vitals. Useful for research, not the number to quote. |
| **Difference** | How much clinical urgency changed the estimate. Small difference → acuity didn’t move the needle much. Larger difference → triage/vitals mattered for this case. |

**Why we keep both**

Our research asks: *do people wait longer mainly because they’re sicker, or do factors like insurance and race still matter?*  
Comparing Model 1 → Model 2 helps answer that. For a demo patient, you only need the **main estimate**.

**For this patient**
                """
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Model 1 (ignored clinical urgency)", f"{wait1:.0f} min")
            c2.metric("Model 2 (main estimate)", f"{wait2:.0f} min")
            c3.metric("How much clinical info changed it", f"{wait2 - wait1:+.0f} min")

        with st.expander("Patient inputs used"):
            p = single["patient"]
            st.write(
                f"- Age {p['age_years']}, {p['sex']}, {p['race']}, {p['ethnicity']}\n"
                f"- {p['payment_type']} · {p['arrival_mode']} · triage {p['triage_acuity']}\n"
                f"- Pulse {p['pulse']}, BP {p['systolic_bp']} · {p['visit_month']} {p['survey_year']}"
            )

    with st.expander("Model test performance (for reference)"):
        st.dataframe(
            nhamcs_metrics_table(artifacts["nhamcs_metrics"]),
            hide_index=True,
            width="stretch",
        )
        st.caption("Typical error is ~31 minutes MAE — use as a rough guide, not an exact clock.")

    # ------------------------------------------------------------------
    # Optional batch CSV (collapsed)
    # ------------------------------------------------------------------
    with st.expander("Optional: upload many patients (CSV)", expanded=False):
        st.caption("For a hospital batch file. Most demos only need the form above.")
        template_bytes = build_template_bytes()
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download CSV template",
                data=template_bytes,
                file_name="nhamcs_wait_time_upload_template.csv",
                mime="text/csv",
                width="stretch",
                key="wait_template_download",
            )
        with d2:

            def _load_demo_visits() -> None:
                keep_tab_selected("Wait-Time Prediction")
                st.session_state["wait_raw_df"] = load_demo_dataframe()
                st.session_state["wait_upload_key"] = "demo_builtin"
                st.session_state.pop("wait_preds", None)

            st.button(
                "Load demo CSV",
                width="stretch",
                key="wait_load_demo",
                on_click=_load_demo_visits,
            )

        uploaded = st.file_uploader(
            "CSV of ED visits",
            type=["csv"],
            key="wait_file_uploader",
        )

        raw_df = None
        if uploaded is not None:
            try:
                raw_df = pd.read_csv(uploaded)
                upload_key = f"{uploaded.name}:{uploaded.size}"
                if st.session_state.get("wait_upload_key") != upload_key:
                    st.session_state["wait_upload_key"] = upload_key
                    st.session_state["wait_raw_df"] = raw_df
                    st.session_state.pop("wait_preds", None)
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
                raw_df = None
        elif "wait_raw_df" in st.session_state:
            raw_df = st.session_state["wait_raw_df"]

        if raw_df is not None and not raw_df.empty:
            st.dataframe(raw_df.head(5), width="stretch", hide_index=True)
            if not looks_encoded(raw_df, features2):
                problems = validate_friendly(raw_df)
                if problems:
                    for problem in problems:
                        st.error(problem)
                    return

            if st.button(
                "Run batch predictions",
                type="primary",
                key="wait_run_preds",
                on_click=keep_tab_selected,
                args=("Wait-Time Prediction",),
            ):
                enc1 = encode_dataframe(raw_df, features1)
                enc2 = encode_dataframe(raw_df, features2)
                pred1 = predict_batch(artifacts["nhamcs_model1"], features1, enc1)
                pred2 = predict_batch(artifacts["nhamcs_model2"], features2, enc2)
                results = raw_df.copy()
                if "patient_id" not in results.columns:
                    results.insert(0, "patient_id", [f"row_{i+1}" for i in range(len(results))])
                results["pred_wait_model1_min"] = np.round(pred1, 1)
                results["pred_wait_model2_min"] = np.round(pred2, 1)
                st.session_state["wait_preds"] = results

            results = st.session_state.get("wait_preds")
            if results is not None:
                st.dataframe(
                    results[
                        [
                            c
                            for c in [
                                "patient_id",
                                "pred_wait_model1_min",
                                "pred_wait_model2_min",
                            ]
                            if c in results.columns
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )
                st.download_button(
                    "Download predictions",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="nhamcs_wait_time_predictions.csv",
                    mime="text/csv",
                    key="wait_preds_download",
                )
