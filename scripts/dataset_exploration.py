#!/usr/bin/env python3
"""
NHAMCS Emergency Department Dataset Exploration (1992-2022).

Combines all NHAMCS_ED_*_decoded.csv files, cleans data, generates
summary statistics, visualizations, and a markdown exploration report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_GLOB = "NHAMCS_ED_*_decoded.csv"
VIZ_DIR = ROOT / "outputs" / "visualizations"
DELIVERABLES_DIR = ROOT / "deliverables"
STATS_JSON = DELIVERABLES_DIR / "exploration_stats.json"

# Standardized column names used in the combined dataset
STANDARD_COLS = [
    "survey_year",
    "wait_time_min",
    "age_years",
    "sex",
    "race",
    "ethnicity",
    "triage_immediacy",
    "arrival_mode",
    "payment_type",
    "admit_hospital",
    "admit_observation",
    "disposition",
    "length_of_visit_min",
    "pulse",
    "systolic_bp",
    "pain_scale",
    "seen_72h",
    "visit_month",
    "visit_day_of_week",
    "residence",
]

# Columns used for ML covariate planning (excluding outcome and survey_year)
COVARIATE_COLS = [c for c in STANDARD_COLS if c not in ("survey_year", "wait_time_min")]

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

# Per-year column prefix resolution (first match wins)
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
    "disposition": ["ADISP__", "OTHERDIS__"],
    "length_of_visit_min": ["LOV__"],
    "pulse": ["PULSE__"],
    "systolic_bp": ["BPSYS__"],
    "pain_scale": ["PAINSCALE__", "PAIN__"],
    "seen_72h": ["SEEN72__"],
    "visit_month": ["VMONTH__"],
    "visit_day_of_week": ["VDAYR__"],
    "residence": ["RESIDNCE__"],
}


def find_column(columns: list[str], prefixes: list[str]) -> str | None:
    for prefix in prefixes:
        prefix_lower = prefix.lower()
        for col in columns:
            if col.lower().startswith(prefix_lower):
                return col
    return None


def extract_year(path: Path) -> int:
    match = re.search(r"NHAMCS_ED_(\d{4})_decoded\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot parse year from {path.name}")
    return int(match.group(1))


def parse_numeric(series: pd.Series) -> pd.Series:
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


def parse_age(series: pd.Series) -> pd.Series:
    numeric = parse_numeric(series)
    text = series.astype(str).str.strip().str.lower()
    # Map common NHAMCS age categories to midpoint approximations where needed
    category_map = {
        "under one year": 0.5,
        "under 15 years": 7,
        "15-24 years": 20,
        "25-44 years": 35,
        "45-64 years": 55,
        "65-74 years": 70,
        "75 years and over": 80,
    }
    mapped = text.map(category_map)
    return numeric.fillna(mapped)


def normalize_categorical(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    lower = out.str.lower()
    missing_mask = lower.isin(MISSING_TOKENS) | lower.str.contains(
        r"^(?:blank|not applicable|unknown|no entry)", regex=True, na=False
    )
    return out.mask(missing_mask, np.nan)


def load_year_file(path: Path) -> pd.DataFrame:
    year = extract_year(path)
    header = pd.read_csv(path, nrows=0)
    source_cols = list(header.columns)

    rename_map: dict[str, str] = {}
    for std_col, prefixes in YEAR_COLUMN_MAP.items():
        src = find_column(source_cols, prefixes)
        if src:
            rename_map[src] = std_col

    if not rename_map:
        return pd.DataFrame(columns=STANDARD_COLS)

    usecols = list(rename_map.keys())
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(columns=rename_map)
    df["survey_year"] = year

    for col in STANDARD_COLS:
        if col not in df.columns:
            df[col] = np.nan

    return df[STANDARD_COLS]


def load_all_years() -> tuple[pd.DataFrame, list[int]]:
    files = sorted(ROOT.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matching {DATA_GLOB} in {ROOT}")

    years = [extract_year(f) for f in files]
    frames = [load_year_file(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    return combined, years


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw_rows = len(df)
    raw_cols = len(df.columns)

    # Remove exact duplicate rows
    df = df.drop_duplicates()
    duplicates_removed = raw_rows - len(df)

    # Drop columns that are entirely empty
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty_cols)

    # Parse numeric fields
    for col in [
        "wait_time_min",
        "age_years",
        "length_of_visit_min",
        "pulse",
        "systolic_bp",
        "pain_scale",
    ]:
        if col == "age_years":
            df[col] = parse_age(df[col])
        else:
            df[col] = parse_numeric(df[col])

    # Normalize categoricals
    cat_cols = [
        "sex",
        "race",
        "ethnicity",
        "triage_immediacy",
        "arrival_mode",
        "payment_type",
        "admit_hospital",
        "admit_observation",
        "disposition",
        "seen_72h",
        "visit_month",
        "visit_day_of_week",
        "residence",
    ]
    for col in cat_cols:
        df[col] = normalize_categorical(df[col])

    # Harmonize triage labels across URGENT / IMMED / IMMEDR eras
    triage_map = {
        "immediate": "Immediate",
        "emergent": "Emergent",
        "urgent": "Urgent",
        "semi-urgent": "Semi-urgent",
        "semiurgent": "Semi-urgent",
        "nonurgent": "Nonurgent",
        "non-urgent": "Nonurgent",
        "not urgent": "Nonurgent",
        "not an emergency": "Nonurgent",
    }
    if "triage_immediacy" in df.columns:
        lower = df["triage_immediacy"].str.lower()
        df["triage_immediacy"] = lower.map(triage_map).fillna(df["triage_immediacy"])

    # Create derived disposition category for visualization
    def disposition_category(row: pd.Series) -> str | float:
        if pd.notna(row.get("admit_hospital")) and str(row["admit_hospital"]).lower().startswith("yes"):
            return "Admitted to hospital"
        if pd.notna(row.get("admit_observation")) and str(row["admit_observation"]).lower().startswith("yes"):
            return "Admitted to observation"
        if pd.notna(row.get("disposition")):
            return str(row["disposition"])
        return np.nan

    df["disposition_category"] = df.apply(disposition_category, axis=1)

    cleaning_meta = {
        "raw_rows": raw_rows,
        "raw_cols": raw_cols,
        "duplicates_removed": duplicates_removed,
        "empty_columns_removed": empty_cols,
        "final_rows": len(df),
        "final_cols": len(df.columns),
    }
    return df, cleaning_meta


def outlier_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        rows.append(
            {
                "column": col,
                "n_valid": int(series.shape[0]),
                "min": float(series.min()),
                "q1": float(q1),
                "median": float(series.median()),
                "q3": float(q3),
                "max": float(series.max()),
                "iqr_lower": float(lower),
                "iqr_upper": float(upper),
                "n_outliers_iqr": int(outliers.shape[0]),
                "pct_outliers_iqr": float(100 * outliers.shape[0] / series.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def missing_values_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    total_cells = len(df) * len(cols)
    total_missing = 0
    for col in cols:
        n_missing = int(df[col].isna().sum())
        total_missing += n_missing
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "n_missing": n_missing,
                "pct_missing": round(100 * n_missing / len(df), 2),
                "n_non_missing": int(df[col].notna().sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("pct_missing", ascending=False)
    return summary, total_missing, total_cells


def categorize_covariates() -> dict[str, list[str]]:
    return {
        "demographic": [
            "age_years",
            "sex",
            "race",
            "ethnicity",
            "residence",
            "visit_month",
            "visit_day_of_week",
        ],
        "clinical": [
            "triage_immediacy",
            "pulse",
            "systolic_bp",
            "pain_scale",
            "seen_72h",
            "length_of_visit_min",
            "admit_hospital",
            "admit_observation",
            "disposition",
            "disposition_category",
        ],
        "access": [
            "arrival_mode",
            "payment_type",
            "survey_year",
        ],
    }


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def save_wait_time_distribution(df: pd.DataFrame) -> Path:
    out = VIZ_DIR / "wait_time_distribution.png"
    wt = df["wait_time_min"].dropna()
    wt = wt[(wt >= 0) & (wt <= 240)]  # clinically plausible window for viz
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(wt, bins=48, color="#2E86AB", edgecolor="white", alpha=0.9)
    ax.set_title("Distribution of ED Wait Time to First Provider")
    ax.set_xlabel("Wait time (minutes)")
    ax.set_ylabel("Number of visits")
    ax.axvline(wt.median(), color="#A23B72", linestyle="--", label=f"Median = {wt.median():.0f} min")
    ax.legend()
    fig.savefig(out)
    plt.close(fig)
    return out


def save_disposition_bar_chart(df: pd.DataFrame) -> Path:
    out = VIZ_DIR / "target_disposition_bar_chart.png"
    counts = df["disposition_category"].dropna().value_counts().head(10)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, palette="viridis", legend=False, ax=ax)
    ax.set_title("Top ED Disposition Categories (All Years)")
    ax.set_xlabel("Number of visits")
    ax.set_ylabel("Disposition")
    fig.savefig(out)
    plt.close(fig)
    return out


def save_age_by_disposition_boxplot(df: pd.DataFrame) -> Path:
    out = VIZ_DIR / "age_by_disposition_boxplot.png"
    plot_df = df.dropna(subset=["age_years", "disposition_category"]).copy()
    top_disp = plot_df["disposition_category"].value_counts().head(6).index
    plot_df = plot_df[plot_df["disposition_category"].isin(top_disp)]
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(
        data=plot_df,
        x="disposition_category",
        y="age_years",
        hue="disposition_category",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title("Patient Age by ED Disposition")
    ax.set_xlabel("Disposition")
    ax.set_ylabel("Age (years)")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(out)
    plt.close(fig)
    return out


def save_age_pulse_scatter(df: pd.DataFrame) -> Path:
    out = VIZ_DIR / "age_vs_pulse_scatter.png"
    plot_df = df.dropna(subset=["age_years", "pulse"]).copy()
    plot_df = plot_df[(plot_df["pulse"] > 0) & (plot_df["pulse"] < 250)]
    plot_df = plot_df[(plot_df["age_years"] >= 0) & (plot_df["age_years"] <= 100)]
    if len(plot_df) > 8000:
        plot_df = plot_df.sample(8000, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(plot_df["age_years"], plot_df["pulse"], alpha=0.25, s=12, color="#3A7CA5")
    ax.set_title("Age vs. Initial Heart Rate")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Pulse (beats per minute)")
    fig.savefig(out)
    plt.close(fig)
    return out


def admitted_flag(df: pd.DataFrame) -> pd.Series:
    admitted = df["admit_hospital"].astype(str).str.lower().str.startswith("yes")
    obs = df["admit_observation"].astype(str).str.lower().str.startswith("yes")
    return (admitted | obs).map({True: "Admitted", False: "Not Admitted"})


def save_admit_rate_bar(df: pd.DataFrame, group_col: str, filename: str, title: str) -> Path:
    out = VIZ_DIR / filename
    plot_df = df.dropna(subset=[group_col, "admit_hospital"]).copy()
    plot_df["admitted"] = admitted_flag(plot_df).eq("Admitted").astype(float)
    rates = plot_df.groupby(group_col, observed=True)["admitted"].mean().sort_values(ascending=False)
    if len(rates) > 10:
        rates = rates.head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=rates.index, y=rates.values, hue=rates.index, palette="viridis", legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_ylabel("Admission rate")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(out)
    plt.close(fig)
    return out


def save_scatter_age_pulse_by_disposition(df: pd.DataFrame) -> Path:
    out = VIZ_DIR / "scatter_age_vs_pulse_by_disposition.png"
    plot_df = df.dropna(subset=["age_years", "pulse"]).copy()
    plot_df = plot_df[(plot_df["pulse"] > 0) & (plot_df["pulse"] < 250)]
    plot_df = plot_df[(plot_df["age_years"] >= 0) & (plot_df["age_years"] <= 100)]
    plot_df["disposition_group"] = admitted_flag(plot_df)
    if len(plot_df) > 8000:
        plot_df = plot_df.sample(8000, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=plot_df,
        x="age_years",
        y="pulse",
        hue="disposition_group",
        alpha=0.4,
        s=18,
        ax=ax,
    )
    ax.set_title("Age vs Heart Rate by Admission Status")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Pulse (beats per minute)")
    fig.savefig(out)
    plt.close(fig)
    return out


def save_correlation_matrix(df: pd.DataFrame) -> Path:
    out = VIZ_DIR / "correlation_matrix_selected_numeric.png"
    num_cols = [
        "wait_time_min",
        "age_years",
        "length_of_visit_min",
        "pulse",
        "systolic_bp",
        "pain_scale",
    ]
    corr = df[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix — Selected Numeric Variables")
    fig.savefig(out)
    plt.close(fig)
    return out


def save_wait_time_by_triage_bar(df: pd.DataFrame) -> Path:
    out = VIZ_DIR / "wait_time_by_triage_bar.png"
    plot_df = df.dropna(subset=["wait_time_min", "triage_immediacy"]).copy()
    plot_df = plot_df[(plot_df["wait_time_min"] >= 0) & (plot_df["wait_time_min"] <= 240)]
    order = ["Immediate", "Emergent", "Urgent", "Semi-urgent", "Nonurgent"]
    present = [o for o in order if o in plot_df["triage_immediacy"].unique()]
    means = (
        plot_df.groupby("triage_immediacy", observed=True)["wait_time_min"]
        .mean()
        .reindex(present)
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=means.index, y=means.values, hue=means.index, palette="RdYlGn_r", legend=False, ax=ax)
    ax.set_title("Mean Wait Time by Triage Immediacy")
    ax.set_xlabel("Triage category")
    ax.set_ylabel("Mean wait time (minutes)")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(out)
    plt.close(fig)
    return out


def save_wait_time_by_payment_bar(df: pd.DataFrame) -> Path:
    out = VIZ_DIR / "wait_time_by_payment_bar.png"
    plot_df = df.dropna(subset=["wait_time_min", "payment_type"]).copy()
    plot_df = plot_df[(plot_df["wait_time_min"] >= 0) & (plot_df["wait_time_min"] <= 240)]
    top_pay = plot_df["payment_type"].value_counts().head(8).index
    plot_df = plot_df[plot_df["payment_type"].isin(top_pay)]
    means = plot_df.groupby("payment_type", observed=True)["wait_time_min"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=means.values, y=means.index, hue=means.index, palette="Blues_d", legend=False, ax=ax)
    ax.set_title("Mean Wait Time by Expected Payment Source")
    ax.set_xlabel("Mean wait time (minutes)")
    ax.set_ylabel("Payment type")
    fig.savefig(out)
    plt.close(fig)
    return out


def save_admit_rate_by_demographics(df: pd.DataFrame) -> Path:
    out = VIZ_DIR / "admit_rate_by_sex_bar.png"
    plot_df = df.dropna(subset=["sex", "admit_hospital"]).copy()
    plot_df["admitted"] = plot_df["admit_hospital"].str.lower().str.startswith("yes").astype(int)
    rates = plot_df.groupby("sex", observed=True)["admitted"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=rates.index, y=rates.values, hue=rates.index, palette="muted", legend=False, ax=ax)
    ax.set_title("Hospital Admission Rate by Sex")
    ax.set_xlabel("Sex")
    ax.set_ylabel("Admission rate (%)")
    fig.savefig(out)
    plt.close(fig)
    return out


def save_yearly_wait_time_trend(df: pd.DataFrame) -> Path:
    out = VIZ_DIR / "wait_time_yearly_trend.png"
    plot_df = df.dropna(subset=["wait_time_min", "survey_year"]).copy()
    plot_df = plot_df[(plot_df["wait_time_min"] >= 0) & (plot_df["wait_time_min"] <= 240)]
    yearly = (
        plot_df.groupby("survey_year", observed=True)["wait_time_min"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly["survey_year"], yearly["mean"], marker="o", label="Mean wait time", color="#E76F51")
    ax.plot(yearly["survey_year"], yearly["median"], marker="s", label="Median wait time", color="#264653")
    ax.set_title("ED Wait Time Trends by Survey Year")
    ax.set_xlabel("Survey year")
    ax.set_ylabel("Wait time (minutes)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out)
    plt.close(fig)
    return out


def generate_visualizations(df: pd.DataFrame) -> list[str]:
    setup_plot_style()
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        save_wait_time_distribution(df),
        save_disposition_bar_chart(df),
        save_age_by_disposition_boxplot(df),
        save_scatter_age_pulse_by_disposition(df),
        save_correlation_matrix(df),
        save_wait_time_by_triage_bar(df),
        save_wait_time_by_payment_bar(df),
        save_admit_rate_bar(
            df,
            "triage_immediacy",
            "admit_rate_by_immediacy.png",
            "Admission Rate by Triage Immediacy",
        ),
        save_admit_rate_bar(
            df,
            "arrival_mode",
            "admit_rate_by_arrivalmode.png",
            "Admission Rate by Arrival Mode",
        ),
        save_admit_rate_bar(
            df,
            "payment_type",
            "admit_rate_by_insurance_status.png",
            "Admission Rate by Insurance / Payment Type",
        ),
        save_admit_rate_bar(df, "sex", "admit_rate_by_gender.png", "Admission Rate by Sex"),
        save_admit_rate_bar(df, "race", "admit_rate_by_race.png", "Admission Rate by Race"),
        save_admit_rate_bar(
            df,
            "ethnicity",
            "admit_rate_by_ethnicity.png",
            "Admission Rate by Ethnicity",
        ),
        save_yearly_wait_time_trend(df),
    ]
    return [str(p.relative_to(ROOT)) for p in paths]


def schema_notes(years: list[int]) -> str:
    return f"""
### Schema differences across survey years

NHAMCS ED questionnaires evolved substantially from 1992 to 2022. After stacking files,
the combined dataset uses **{len(STANDARD_COLS)} standardized fields** (plus derived
`disposition_category`). Key cross-year differences:

| Concept | Early years (1992–2000) | Middle years (2001–2011) | Recent years (2012–2022) |
|---|---|---|---|
| Wait time | `WAITTIME` (1997–2000 only) | `waittime`/`WAITTIME` (2003–2011) | `WAITTIME` (provider contact, 2012+) |
| Triage urgency | `URGENT` | `IMMED` (2001–2008) | `IMMEDR` (2009+) |
| Race | `RACE` | `RACE` / `RACEUN` | `RACEUN` |
| Payment | Individual payer flags; `PAYTYPE` | `PAYTYPE` | `PAYTYPER` |
| Arrival mode | `ARRIVE` | `ARRIVE` / `ARREMS` | `ARREMS` |
| Pain | `PAIN` (categorical) | `PAIN` | `PAINSCALE` (0–10) |
| Vitals | Limited / absent early | `PULSE`, `BPSYS` (2001+) | Full initial vitals |

**Wait time coverage:** `wait_time_min` is populated for survey years **1997–2000**,
**2003–2006** (lowercase field names in source files), and **2007–2022**
({len([y for y in years if (1997 <= y <= 2000) or (2003 <= y <= 2006) or (y >= 2007)])} of
{len(years)} files). Years **1992–1996**, **2001–2002**, and early files without a
`WAITTIME`/`waittime` field appear as missing for the outcome variable.

Files included: **{len(years)}** (`{years[0]}`–`{years[-1]}`).
"""


def write_report(
    df: pd.DataFrame,
    years: list[int],
    cleaning: dict,
    missing_df: pd.DataFrame,
    total_missing: int,
    total_cells: int,
    outlier_df: pd.DataFrame,
    covariate_groups: dict[str, list[str]],
    viz_paths: list[str],
) -> Path:
    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DELIVERABLES_DIR / "dataset_exploration.md"

    wt = df["wait_time_min"].dropna()
    wt_valid = wt[(wt >= 0) & (wt <= 480)]

    covariate_count = sum(len(v) for k, v in covariate_groups.items() if k != "access" or True)
    # Count planned predictors excluding survey_year from access for ML
    planned_predictors = [c for group, cols in covariate_groups.items() for c in cols if c != "survey_year"]

    missing_table_md = missing_df.to_markdown(index=False)
    outlier_table_md = outlier_df.to_markdown(index=False) if not outlier_df.empty else "_No numeric outliers computed._"

    covariate_sections = []
    for group, cols in covariate_groups.items():
        covariate_sections.append(f"**{group.title()} ({len(cols)}):** " + ", ".join(f"`{c}`" for c in cols))

    viz_section = "\n".join(
        f"![{Path(p).stem}](../{p})" for p in viz_paths
    )

    content = f"""# NHAMCS Emergency Department — Dataset Exploration Report

> **Dataset: NHAMCS Emergency Department (not Yale).**  
> This report summarizes the National Hospital Ambulatory Medical Care Survey (NHAMCS)
> Emergency Department component, combining all decoded annual files available in the workspace.

## 1. Dataset Overview

| Item | Value |
|---|---|
| Source | NHAMCS Emergency Department public-use files (decoded CSV) |
| Files combined | **{len(years)}** (`NHAMCS_ED_{years[0]}_decoded.csv` – `NHAMCS_ED_{years[-1]}_decoded.csv`) |
| Survey years | {years[0]}–{years[-1]} |
| Raw rows (stacked) | {cleaning['raw_rows']:,} |
| Duplicates removed | {cleaning['duplicates_removed']:,} |
| Empty columns removed | {len(cleaning['empty_columns_removed'])} ({', '.join(cleaning['empty_columns_removed']) or 'none'}) |
| **Final rows** | **{cleaning['final_rows']:,}** |
| **Final columns** | **{cleaning['final_cols']}** (standardized + derived) |

### Combining approach

1. Load each annual `NHAMCS_ED_YYYY_decoded.csv` with year-specific column resolution.
2. Rename fields to a common schema (`wait_time_min`, `triage_immediacy`, etc.).
3. Append a `survey_year` column when stacking.
4. Remove exact duplicate rows and wholly empty columns.
5. Parse numeric outcomes/predictors; normalize categorical missing tokens (`Blank`, `Not Applicable`, etc.).

{schema_notes(years)}

## 2. Missing Values

| Metric | Value |
|---|---|
| Total cells (key columns) | {total_cells:,} |
| Total missing values | {total_missing:,} |
| Overall missing rate | {100 * total_missing / total_cells:.2f}% |

### Per-column missingness (project-relevant columns)

{missing_table_md}

## 3. Outlier Analysis (IQR method, numeric columns)

Outliers defined as values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR.

{outlier_table_md}

## 4. Covariates / Predictors

**Total planned predictor fields: {len(planned_predictors)}** (excluding dependent variable `wait_time_min`).

{chr(10).join(covariate_sections)}

These fields support the research question: *Are Emergency Department wait times determined by medical need?*
Triage immediacy (`triage_immediacy`), vitals, pain, and admission/disposition variables proxy medical acuity;
demographics and access variables capture non-clinical factors.

## 5. Dependent Variable (Wait-Time ML Project)

| Field | Description |
|---|---|
| **`wait_time_min`** | Wait time to first provider contact (minutes). Mapped from NHAMCS `WAITTIME` across years. |
| Non-missing observations | {int(wt.shape[0]):,} ({100 * wt.shape[0] / len(df):.1f}% of combined rows) |
| Summary stats range (0–480 min) | {int(wt_valid.shape[0]):,} visits |
| Median (0–480 min) | {wt_valid.median():.0f} minutes |
| Mean (0–480 min) | {wt_valid.mean():.1f} minutes |
| Range (all non-missing) | {wt.min():.0f} – {wt.max():.0f} minutes |

**Secondary outcomes for exploratory analysis:**
- `admit_hospital`, `admit_observation`, `disposition` / `disposition_category` — ED disposition and admission
- `length_of_visit_min` — total ED length of stay

## 6. Visualizations

{viz_section}

---

*Generated by `scripts/dataset_exploration.py`. Re-run to reproduce statistics and figures.*
"""
    report_path.write_text(content)
    return report_path


def main() -> None:
    print("Loading all NHAMCS ED files...")
    df, years = load_all_years()
    print(f"  Loaded {len(years)} files, {len(df):,} rows")

    print("Cleaning dataset...")
    df, cleaning = clean_dataset(df)

    key_cols = [c for c in STANDARD_COLS if c in df.columns]
    missing_df, total_missing, total_cells = missing_values_table(df, key_cols)

    numeric_for_outliers = [
        c
        for c in ["wait_time_min", "age_years", "length_of_visit_min", "pulse", "systolic_bp", "pain_scale"]
        if c in df.columns
    ]
    outlier_df = outlier_summary(df, numeric_for_outliers)

    covariate_groups = categorize_covariates()

    print("Generating visualizations...")
    viz_paths = generate_visualizations(df)

    print("Writing report...")
    report_path = write_report(
        df,
        years,
        cleaning,
        missing_df,
        total_missing,
        total_cells,
        outlier_df,
        covariate_groups,
        viz_paths,
    )

    stats = {
        "files": len(years),
        "years": years,
        "final_rows": cleaning["final_rows"],
        "final_cols": cleaning["final_cols"],
        "duplicates_removed": cleaning["duplicates_removed"],
        "dependent_variable": "wait_time_min",
        "wait_time_valid_n": int(df["wait_time_min"].notna().sum()),
        "covariate_count": sum(len(v) for v in covariate_groups.values()) - 1,  # exclude survey_year double count
        "visualizations": viz_paths,
        "report": str(report_path.relative_to(ROOT)),
    }
    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
    STATS_JSON.write_text(json.dumps(stats, indent=2))

    print("\n=== Summary ===")
    print(json.dumps(stats, indent=2))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
