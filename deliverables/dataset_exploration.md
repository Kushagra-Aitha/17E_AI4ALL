# NHAMCS Emergency Department — Dataset Exploration Report

> **Dataset: NHAMCS Emergency Department (not Yale).**  
> This report summarizes the National Hospital Ambulatory Medical Care Survey (NHAMCS)
> Emergency Department component, combining all decoded annual files available in the workspace.

## 1. Dataset Overview

| Item | Value |
|---|---|
| Source | NHAMCS Emergency Department public-use files (decoded CSV) |
| Files combined | **16** (`NHAMCS_ED_2007_decoded.csv` – `NHAMCS_ED_2022_decoded.csv`) |
| Survey years | 2007–2022 |
| Raw rows (stacked) | 392,761 |
| Duplicates removed | 163 |
| Empty columns removed | 0 (none) |
| **Final rows** | **392,598** |
| **Final columns** | **21** (standardized + derived) |

### Combining approach

1. Load each annual `NHAMCS_ED_YYYY_decoded.csv` with year-specific column resolution.
2. Rename fields to a common schema (`wait_time_min`, `triage_immediacy`, etc.).
3. Append a `survey_year` column when stacking.
4. Remove exact duplicate rows and wholly empty columns.
5. Parse numeric outcomes/predictors; normalize categorical missing tokens (`Blank`, `Not Applicable`, etc.).


### Schema differences across survey years

The 16 files available (`2007`–`2022`) span one NHAMCS questionnaire
transition, around 2009. After stacking files, the combined dataset uses
**20 standardized fields** (plus derived `disposition_category`). Key
cross-year differences within this range:

| Concept | 2007–2008 | 2009–2022 |
|---|---|---|
| Wait time | `WAITTIME` (waiting time to see physician) | `WAITTIME` (waiting time to see MD/DO/PA/NP) |
| Triage urgency | `IMMED` | `IMMEDR` (recoded, unimputed) |
| Arrival mode | `ARRIVE` (mode of arrival) | `ARREMS` (arrival by ambulance) |
| Payment | `PAYTYPE` (2007 only) / `PAYTYPER` | `PAYTYPER` (recoded, hierarchy-based) |
| Pain | `PAIN` (categorical) | `PAINSCALE` (0–10 numeric) |
| Race | `RACEUN` (present throughout) | `RACEUN` (present throughout) |

`wait_time_min` is populated for all 16 of 16 files in this range — this
is why the modeling pipeline (`scripts/preprocess.py`) restricts to 2007–2022,
avoiding the outcome-variable gaps and additional questionnaire eras present in years before
2007.

Files included: **16** (`2007`–`2022`).


## 2. Missing Values

| Metric | Value |
|---|---|
| Total cells (key columns) | 7,851,960 |
| Total missing values | 962,916 |
| Overall missing rate | 12.26% |

### Per-column missingness (project-relevant columns)

| column              | dtype   |   n_missing |   pct_missing |   n_non_missing |
|:--------------------|:--------|------------:|--------------:|----------------:|
| disposition         | str     |      352874 |         89.88 |           39724 |
| pain_scale          | float64 |      168208 |         42.84 |          224390 |
| triage_immediacy    | str     |       76321 |         19.44 |          316277 |
| race                | str     |       62126 |         15.82 |          330472 |
| wait_time_min       | float64 |       60276 |         15.35 |          332322 |
| length_of_visit_min | float64 |       55324 |         14.09 |          337274 |
| seen_72h            | str     |       49944 |         12.72 |          342654 |
| systolic_bp         | float64 |       46260 |         11.78 |          346338 |
| payment_type        | str     |       31227 |          7.95 |          361371 |
| pulse               | float64 |       27804 |          7.08 |          364794 |
| arrival_mode        | str     |       17218 |          4.39 |          375380 |
| residence           | str     |       13685 |          3.49 |          378913 |
| age_years           | float64 |        1649 |          0.42 |          390949 |
| visit_day_of_week   | str     |           0 |          0    |          392598 |
| visit_month         | str     |           0 |          0    |          392598 |
| survey_year         | int64   |           0 |          0    |          392598 |
| admit_hospital      | str     |           0 |          0    |          392598 |
| ethnicity           | str     |           0 |          0    |          392598 |
| sex                 | str     |           0 |          0    |          392598 |
| admit_observation   | str     |           0 |          0    |          392598 |

## 3. Outlier Analysis (IQR method, numeric columns)

Outliers defined as values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR.

| column              |   n_valid |   min |   q1 |   median |   q3 |   max |   iqr_lower |   iqr_upper |   n_outliers_iqr |   pct_outliers_iqr |
|:--------------------|----------:|------:|-----:|---------:|-----:|------:|------------:|------------:|-----------------:|-------------------:|
| wait_time_min       |    332322 |   0   |    9 |       23 |   53 |  1440 |       -57   |       119   |            28351 |            8.53118 |
| age_years           |    390949 |   0.5 |   19 |       35 |   55 |    99 |       -35   |       109   |                0 |            0       |
| length_of_visit_min |    337274 |   0   |   92 |      159 |  264 |  5760 |      -166   |       522   |            21811 |            6.46685 |
| pulse               |    364794 |   0   |   76 |       88 |  102 |   244 |        37   |       141   |            14671 |            4.02172 |
| systolic_bp         |    346338 |   0   |  117 |      131 |  146 |   290 |        73.5 |       189.5 |             8859 |            2.55791 |
| pain_scale          |    224390 |   0   |    0 |        5 |    8 |    10 |       -12   |        20   |                0 |            0       |

## 4. Covariates / Predictors

**Total planned predictor fields: 19** (excluding dependent variable `wait_time_min`).

**Demographic (7):** `age_years`, `sex`, `race`, `ethnicity`, `residence`, `visit_month`, `visit_day_of_week`
**Clinical (10):** `triage_immediacy`, `pulse`, `systolic_bp`, `pain_scale`, `seen_72h`, `length_of_visit_min`, `admit_hospital`, `admit_observation`, `disposition`, `disposition_category`
**Access (3):** `arrival_mode`, `payment_type`, `survey_year`

These fields support the research question: *Are Emergency Department wait times determined by medical need?*
Triage immediacy (`triage_immediacy`), vitals, pain, and admission/disposition variables proxy medical acuity;
demographics and access variables capture non-clinical factors.

**Note — planned vs. final modeling features:** the list above is the full candidate set
identified during EDA. `scripts/preprocess.py` narrows this for actual modeling: `admit_hospital`,
`admit_observation`, `disposition`, `disposition_category`, and `length_of_visit_min` are dropped
as post-visit outcomes (not known at triage time — including them would leak the outcome into the
predictors), and `pain_scale` is dropped for 71% missingness (only collected 2011+). The two
model-ready feature matrices this produces are documented in `MODELING.md` and
`data/processed/preprocessing_report.json`.

## 5. Dependent Variable (Wait-Time ML Project)

| Field | Description |
|---|---|
| **`wait_time_min`** | Wait time to first provider contact (minutes). Mapped from NHAMCS `WAITTIME` across years. |
| Non-missing observations | 332,322 (84.6% of combined rows) |
| Summary stats range (0–480 min) | 330,682 visits |
| Median (0–480 min) | 22 minutes |
| Mean (0–480 min) | 42.1 minutes |
| Range (all non-missing) | 0 – 1440 minutes |

**Secondary outcomes for exploratory analysis:**
- `admit_hospital`, `admit_observation`, `disposition` / `disposition_category` — ED disposition and admission
- `length_of_visit_min` — total ED length of stay

## 6. Visualizations

![wait_time_distribution](../outputs/visualizations/wait_time_distribution.png)
![target_disposition_bar_chart](../outputs/visualizations/target_disposition_bar_chart.png)
![age_by_disposition_boxplot](../outputs/visualizations/age_by_disposition_boxplot.png)
![scatter_age_vs_pulse_by_disposition](../outputs/visualizations/scatter_age_vs_pulse_by_disposition.png)
![correlation_matrix_selected_numeric](../outputs/visualizations/correlation_matrix_selected_numeric.png)
![wait_time_by_triage_bar](../outputs/visualizations/wait_time_by_triage_bar.png)
![wait_time_by_payment_bar](../outputs/visualizations/wait_time_by_payment_bar.png)
![wait_time_by_race_bar](../outputs/visualizations/wait_time_by_race_bar.png)
![wait_time_by_arrivalmode_bar](../outputs/visualizations/wait_time_by_arrivalmode_bar.png)
![admit_rate_by_immediacy](../outputs/visualizations/admit_rate_by_immediacy.png)
![admit_rate_by_arrivalmode](../outputs/visualizations/admit_rate_by_arrivalmode.png)
![admit_rate_by_insurance_status](../outputs/visualizations/admit_rate_by_insurance_status.png)
![admit_rate_by_gender](../outputs/visualizations/admit_rate_by_gender.png)
![admit_rate_by_race](../outputs/visualizations/admit_rate_by_race.png)
![admit_rate_by_ethnicity](../outputs/visualizations/admit_rate_by_ethnicity.png)
![wait_time_yearly_trend](../outputs/visualizations/wait_time_yearly_trend.png)

---

*Generated by `scripts/dataset_exploration.py`. Re-run to reproduce statistics and figures.*
