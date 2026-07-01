#!/usr/bin/env python3
"""
NHAMCS ED Data Cleaning & Preprocessing Pipeline.

Outputs (all in data/processed/):
  nhamcs_cleaned_2007_2022.csv   cleaned dataset before encoding
  nhamcs_model1_encoded.csv      demographic + access features + targets
  nhamcs_model2_encoded.csv      demographic + access + clinical features + targets
  preprocessing_report.json          run metadata and row/column counts

Design decisions encoded here (see EDA notes for rationale):
  - Only years 2007-2022: consistent WAITTIME field and 5-level triage coding
  - wait_time_min clamped to [0, 480]; rows with wait_time_min > 480 are dropped
  - log_wait_time = log(wait_time_min + 1) is the regression target
  - pulse == 0 and systolic_bp == 0 are treated as missing (recording artifacts)
  - Missing indicators added BEFORE imputation for pulse, systolic_bp, seen_72h
  - Triage collapsed to 3-level acuity (High/Medium/Low) to bridge survey eras
  - Model 1 = demographic + access variables only (measures total disparity)
  - Model 2 = Model 1 + clinical variables (measures residual disparity)
  - survey_year included in both models as a numeric temporal control
  - length_of_visit_min, admit_hospital, disposition excluded (post-visit outcomes)
  - pain_scale excluded (71% missing, only 2011+; use in a dedicated sub-analysis)
  - StandardScaler is NOT applied here; apply it inside the modeling pipeline
    to avoid data leakage when doing cross-validation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_GLOB = "NHAMCS_ED_*_decoded.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

MODEL_YEAR_START = 2007
MODEL_YEAR_END = 2022
WAIT_TIME_CLAMP = 480  # minutes; ~8 hours

MISSING_TOKENS = {
    "",
    " ",
    "blank",
    "not applicable",
    "not applicable.",
    "unknown",
    "-9",
    "no information",
    "item could not be located",
    "all sources of payment are blank",
    "visit occurred in esa that does not conduct nursing triage",
    "no triage for this visit but esa does conduct triage",
}

YEAR_COLUMN_MAP: dict[str, list[str]] = {
    "wait_time_min": ["WAITTIME__"],
    "age_years": ["AGE__"],
    "sex": ["SEX__"],
    "race": ["RACEUN__", "RACER__", "RACE__"],
    "ethnicity": ["ETHIM__", "ETHUN__", "ETHNIC__"],
    "triage_immediacy": ["IMMEDR__", "IMMED__", "URGENT__"],
    "arrival_mode": ["ARREMS__", "ARRIVE__"],
    "payment_type": ["PAYTYPER__", "PAYTYPE__"],
    "admit_hospital": ["ADMITHOS__"],
    "admit_observation": ["ADMITOBS__", "OBSHOS__", "OBSDIS__"],
    "length_of_visit_min": ["LOV__"],
    "pulse": ["PULSE__"],
    "systolic_bp": ["BPSYS__"],
    "pain_scale": ["PAINSCALE__", "PAIN__"],
    "seen_72h": ["SEEN72__"],
    "visit_month": ["VMONTH__"],
    "visit_day_of_week": ["VDAYR__"],
    "residence": ["RESIDNCE__"],
}

STANDARD_COLS = list(YEAR_COLUMN_MAP.keys()) + ["survey_year"]

# Triage labels from all survey eras collapsed to 3-level acuity
TRIAGE_ACUITY_MAP: dict[str, str] = {
    "immediate": "High",
    "emergent": "High",
    "urgent/emergent": "High",
    "urgent": "Medium",
    "semi-urgent": "Medium",
    "semiurgent": "Medium",
    "1-14 min": "Medium",
    "less than 15 minutes": "Medium",
    "15-60 min": "Medium",
    "15- 60 minutes": "Medium",
    "nonurgent": "Low",
    "non-urgent": "Low",
    "not urgent": "Low",
    "not an emergency": "Low",
    ">1 hour - 2 hours": "Low",
    ">1 hour": "Low",
    "no triage": "Low",
}

# Harmonize arrival_mode values from the binary ARREMS era (Yes/No) into the
# categorical era (Ambulance / Walk-in/Other). Both formats appear in 2007-2022.
ARRIVAL_MODE_HARMONIZE: dict[str, str] = {
    "yes": "Ambulance",
    "no": "Walk-in/Other",
}

# Collapse near-duplicate payment labels that arose from questionnaire changes
# across survey years into a single canonical value per payer class.
PAYMENT_TYPE_CONSOLIDATE: dict[str, str] = {
    "medicaid": "Medicaid/CHIP",
    "medicaid or chip": "Medicaid/CHIP",
    "medicaid or chip or other state-based program": "Medicaid/CHIP",
    "medicaid/schip": "Medicaid/CHIP",
    "no charge": "No charge/Charity",
    "no charge/charity": "No charge/Charity",
    # treat "all sources blank" as missing (slip through MISSING_TOKENS filter)
    "all sources for payment are blank": "__MISSING__",
}

# Reference categories dropped during one-hot encoding.
# Chosen as the most clinically/statistically natural baseline.
REFERENCE_CATEGORIES: dict[str, str] = {
    "triage_acuity": "Medium",
    "sex": "Female",
    "race": "White Only",
    "ethnicity": "Not Hispanic or Latino",
    "payment_type": "Private insurance",
    "arrival_mode": "Walk-in/Other",
    "residence": "Private residence",
    "visit_day_of_week": "Wednesday",
    "visit_month": "July",
    # seen_72h_No must be dropped here because seen_72h_No + seen_72h_Yes +
    # seen_72h_missing always sum to 1 — keeping all three is perfectly collinear.
    "seen_72h": "No",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_column(columns: list[str], prefixes: list[str]) -> str | None:
    for prefix in prefixes:
        p = prefix.lower()
        for col in columns:
            if col.lower().startswith(p):
                return col
    return None


def _extract_year(path: Path) -> int:
    m = re.search(r"NHAMCS_ED_(\d{4})_decoded\.csv$", path.name)
    if not m:
        raise ValueError(f"Cannot parse year from {path.name}")
    return int(m.group(1))


def _load_year_file(path: Path) -> pd.DataFrame:
    year = _extract_year(path)
    header = pd.read_csv(path, nrows=0)
    source_cols = list(header.columns)

    rename_map: dict[str, str] = {}
    for std_col, prefixes in YEAR_COLUMN_MAP.items():
        src = _find_column(source_cols, prefixes)
        if src:
            rename_map[src] = std_col

    if not rename_map:
        return pd.DataFrame(columns=STANDARD_COLS)

    df = pd.read_csv(path, usecols=list(rename_map.keys()), low_memory=False)
    df = df.rename(columns=rename_map)
    df["survey_year"] = year
    for col in STANDARD_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[STANDARD_COLS]


def load_raw(year_start: int = MODEL_YEAR_START, year_end: int = MODEL_YEAR_END) -> pd.DataFrame:
    files = sorted(ROOT.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No {DATA_GLOB} files found in {ROOT}")
    frames = [
        _load_year_file(f)
        for f in files
        if year_start <= _extract_year(f) <= year_end
    ]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({np.nan: "", "nan": "", "None": ""})
    )
    lower = cleaned.str.lower()
    invalid = lower.isin(MISSING_TOKENS) | lower.str.contains(
        r"^(?:blank|not applicable|unknown|no entry)", regex=True, na=False
    )
    cleaned = cleaned.mask(invalid, np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_age(series: pd.Series) -> pd.Series:
    numeric = _parse_numeric(series)
    category_map = {
        "under one year": 0.5,
        "under 15 years": 7,
        "15-24 years": 20,
        "25-44 years": 35,
        "45-64 years": 55,
        "65-74 years": 70,
        "75 years and over": 80,
    }
    mapped = series.astype(str).str.strip().str.lower().map(category_map)
    return numeric.fillna(mapped)


def _normalize_categorical(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    lower = out.str.lower()
    missing_mask = lower.isin(MISSING_TOKENS) | lower.str.contains(
        r"^(?:blank|not applicable|unknown|no entry)", regex=True, na=False
    )
    return out.mask(missing_mask, np.nan)


# ---------------------------------------------------------------------------
# Step 1: Basic cleaning
# ---------------------------------------------------------------------------

def clean_basic(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n_raw = len(df)

    df = df.drop_duplicates()
    n_dedup = len(df)

    df["age_years"] = _parse_age(df["age_years"])
    for col in ["wait_time_min", "length_of_visit_min", "pulse", "systolic_bp", "pain_scale"]:
        df[col] = _parse_numeric(df[col])

    cat_cols = [
        "sex", "race", "ethnicity", "triage_immediacy", "arrival_mode",
        "payment_type", "admit_hospital", "admit_observation",
        "seen_72h", "visit_month", "visit_day_of_week", "residence",
    ]
    for col in cat_cols:
        df[col] = _normalize_categorical(df[col])

    # Harmonize arrival_mode: binary Yes/No era → Ambulance/Walk-in/Other
    lower_arr = df["arrival_mode"].str.lower().str.strip()
    for token, canonical in ARRIVAL_MODE_HARMONIZE.items():
        df.loc[lower_arr == token, "arrival_mode"] = canonical

    # Consolidate payment_type near-duplicates across survey eras
    lower_pay = df["payment_type"].str.lower().str.strip()
    for token, canonical in PAYMENT_TYPE_CONSOLIDATE.items():
        mask = lower_pay == token
        if canonical == "__MISSING__":
            df.loc[mask, "payment_type"] = np.nan
        else:
            df.loc[mask, "payment_type"] = canonical

    # pulse = 0 and systolic_bp = 0 are recording artifacts, not real zeros
    df.loc[df["pulse"] == 0, "pulse"] = np.nan
    df.loc[df["systolic_bp"] == 0, "systolic_bp"] = np.nan

    # Keep only rows with a valid wait time
    df = df.dropna(subset=["wait_time_min"])
    n_valid_wait = len(df)

    # Clamp to [0, WAIT_TIME_CLAMP] — values > 480 min are retained in the
    # cleaned file but flagged; the encoded model files exclude them
    df["wait_time_clamped"] = df["wait_time_min"].clip(0, WAIT_TIME_CLAMP)

    meta = {
        "n_raw": n_raw,
        "n_after_dedup": n_dedup,
        "duplicates_removed": n_raw - n_dedup,
        "n_with_valid_wait_time": n_valid_wait,
    }
    return df, meta


# ---------------------------------------------------------------------------
# Step 2: Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log-transform target (shift by 1 so wait_time=0 maps to 0)
    df["log_wait_time"] = np.log1p(df["wait_time_clamped"])

    # 3-level triage acuity across all survey eras
    lower_triage = df["triage_immediacy"].str.lower().str.strip()
    df["triage_acuity"] = lower_triage.map(TRIAGE_ACUITY_MAP)

    # Missing indicators BEFORE imputation
    df["pulse_missing"] = df["pulse"].isna().astype(int)
    df["systolic_bp_missing"] = df["systolic_bp"].isna().astype(int)
    df["seen_72h_missing"] = df["seen_72h"].isna().astype(int)

    # Median imputation for vitals (medians from this training-range dataset)
    pulse_median = df["pulse"].median()
    bp_median = df["systolic_bp"].median()
    df["pulse"] = df["pulse"].fillna(pulse_median)
    df["systolic_bp"] = df["systolic_bp"].fillna(bp_median)

    return df, {"pulse_imputed_median": float(pulse_median), "bp_imputed_median": float(bp_median)}


# ---------------------------------------------------------------------------
# Step 3: One-hot encoding helpers
# ---------------------------------------------------------------------------

def _one_hot(df: pd.DataFrame, col: str, reference: str | None = None, prefix: str | None = None) -> pd.DataFrame:
    prefix = prefix or col
    dummies = pd.get_dummies(df[col], prefix=prefix, dtype=int)
    if reference is not None:
        ref_col = f"{prefix}_{reference}"
        dummies = dummies.drop(columns=[ref_col], errors="ignore")
    return pd.concat([df.drop(columns=[col]), dummies], axis=1)


def encode_for_model(df: pd.DataFrame, include_clinical: bool) -> pd.DataFrame:
    """
    Build a model-ready feature matrix.

    include_clinical=False  →  Model 1 (demographic + access)
    include_clinical=True   →  Model 2 (demographic + access + clinical)
    """
    # Only rows within the clamped wait-time range
    df = df[df["wait_time_min"] <= WAIT_TIME_CLAMP].copy()

    # Columns always kept as-is
    keep_numeric = ["age_years", "survey_year", "wait_time_min", "log_wait_time"]
    if include_clinical:
        keep_numeric += ["pulse", "systolic_bp", "pulse_missing", "systolic_bp_missing"]

    # Categorical columns to one-hot encode
    demographic_cats = ["sex", "race", "ethnicity", "residence"]
    access_cats = ["visit_month", "visit_day_of_week", "payment_type", "arrival_mode"]
    clinical_cats = ["triage_acuity", "seen_72h"]

    cats = demographic_cats + access_cats
    if include_clinical:
        cats += clinical_cats

    # Start with numeric targets + controls
    out = df[keep_numeric].copy()

    # Add seen_72h_missing only in Model 2
    if include_clinical:
        out["seen_72h_missing"] = df["seen_72h_missing"]

    # One-hot encode each categorical, dropping the reference category where defined
    temp = df[cats].copy()
    for col in cats:
        ref = REFERENCE_CATEGORIES.get(col)
        temp = _one_hot(temp, col, reference=ref)

    out = pd.concat([out, temp], axis=1)

    # Drop rows missing age (0.24% — negligible, complete-case on the one
    # remaining required demographic)
    out = out.dropna(subset=["age_years"])

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading NHAMCS ED files ({MODEL_YEAR_START}–{MODEL_YEAR_END})...")
    df_raw = load_raw()
    print(f"  {len(df_raw):,} rows loaded across {MODEL_YEAR_END - MODEL_YEAR_START + 1} years")

    print("Cleaning...")
    df_clean, clean_meta = clean_basic(df_raw)
    print(f"  {len(df_clean):,} rows after dedup + wait-time filter")

    print("Engineering features...")
    df_feat, impute_meta = engineer_features(df_clean)

    # Save cleaned pre-encoding dataset
    clean_path = PROCESSED_DIR / "nhamcs_cleaned_2007_2022.csv"
    df_feat.to_csv(clean_path, index=False)
    print(f"  Saved cleaned dataset → {clean_path.relative_to(ROOT)}")

    print("Encoding Model 1 (demographic + access)...")
    df_m1 = encode_for_model(df_feat, include_clinical=False)
    m1_path = PROCESSED_DIR / "nhamcs_model1_encoded.csv"
    df_m1.to_csv(m1_path, index=False)
    print(f"  {df_m1.shape[0]:,} rows × {df_m1.shape[1]} cols → {m1_path.relative_to(ROOT)}")

    print("Encoding Model 2 (demographic + access + clinical)...")
    df_m2 = encode_for_model(df_feat, include_clinical=True)
    m2_path = PROCESSED_DIR / "nhamcs_model2_encoded.csv"
    df_m2.to_csv(m2_path, index=False)
    print(f"  {df_m2.shape[0]:,} rows × {df_m2.shape[1]} cols → {m2_path.relative_to(ROOT)}")

    # Missing-value summary on the cleaned (pre-encoding) dataset
    key_cols = [
        "wait_time_min", "age_years", "sex", "race", "ethnicity",
        "triage_acuity", "pulse", "systolic_bp", "payment_type",
        "arrival_mode", "residence", "seen_72h",
    ]
    missing_summary = {
        col: {
            "n_missing": int(df_feat[col].isna().sum()),
            "pct_missing": round(100 * df_feat[col].isna().mean(), 2),
        }
        for col in key_cols
        if col in df_feat.columns
    }

    wait_desc = df_feat["wait_time_min"].describe()

    report = {
        "year_range": f"{MODEL_YEAR_START}–{MODEL_YEAR_END}",
        "wait_time_clamp_minutes": WAIT_TIME_CLAMP,
        "cleaning": clean_meta,
        "imputation": impute_meta,
        "cleaned_dataset": {
            "rows": len(df_feat),
            "cols": len(df_feat.columns),
            "path": str(clean_path.relative_to(ROOT)),
        },
        "model1_encoded": {
            "rows": df_m1.shape[0],
            "cols": df_m1.shape[1],
            "feature_columns": [c for c in df_m1.columns if c not in ("wait_time_min", "log_wait_time")],
            "path": str(m1_path.relative_to(ROOT)),
        },
        "model2_encoded": {
            "rows": df_m2.shape[0],
            "cols": df_m2.shape[1],
            "feature_columns": [c for c in df_m2.columns if c not in ("wait_time_min", "log_wait_time")],
            "path": str(m2_path.relative_to(ROOT)),
        },
        "wait_time_stats": {
            "mean": round(float(wait_desc["mean"]), 2),
            "median": float(wait_desc["50%"]),
            "std": round(float(wait_desc["std"]), 2),
            "min": float(wait_desc["min"]),
            "max": float(wait_desc["max"]),
            "n_over_clamp": int((df_feat["wait_time_min"] > WAIT_TIME_CLAMP).sum()),
        },
        "missing_summary_post_clean": missing_summary,
        "reference_categories_dropped": REFERENCE_CATEGORIES,
        "triage_acuity_mapping": TRIAGE_ACUITY_MAP,
        "modeling_notes": [
            "Apply StandardScaler to numeric columns INSIDE your CV pipeline to prevent leakage.",
            "Model 1 vs Model 2 coefficient comparison answers the equity research question.",
            "pain_scale excluded (71% missing); run a separate 2011-2022 sub-analysis if needed.",
            "length_of_visit_min excluded (post-visit outcome, not available at triage time).",
            "admit_hospital / admit_observation excluded (outcomes, not predictors).",
            "Zero values in pulse and systolic_bp were converted to NaN before imputation.",
        ],
    }

    report_path = PROCESSED_DIR / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report → {report_path.relative_to(ROOT)}")

    print("\n=== Preprocessing complete ===")
    print(f"  Model 1 features ({df_m1.shape[1] - 2}):  {[c for c in df_m1.columns if c not in ('wait_time_min','log_wait_time')]}")
    print(f"  Model 2 features ({df_m2.shape[1] - 2}):  {[c for c in df_m2.columns if c not in ('wait_time_min','log_wait_time')]}")


if __name__ == "__main__":
    main()
