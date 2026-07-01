# NHAMCS Emergency Department — Dataset Exploration Report

> **Dataset: NHAMCS Emergency Department (not Yale).**  
> This report summarizes the National Hospital Ambulatory Medical Care Survey (NHAMCS)
> Emergency Department component, combining all decoded annual files available in the workspace.

## 1. Dataset Overview

| Item | Value |
|---|---|
| Source | NHAMCS Emergency Department public-use files (decoded CSV) |
| Files combined | **31** (`NHAMCS_ED_1992_decoded.csv` – `NHAMCS_ED_2022_decoded.csv`) |
| Survey years | 1992–2022 |
| Raw rows (stacked) | 839,772 |
| Duplicates removed | 76,258 |
| Empty columns removed | 0 (none) |
| **Final rows** | **763,514** |
| **Final columns** | **21** (standardized + derived) |

### Combining approach

1. Load each annual `NHAMCS_ED_YYYY_decoded.csv` with year-specific column resolution.
2. Rename fields to a common schema (`wait_time_min`, `triage_immediacy`, etc.).
3. Append a `survey_year` column when stacking.
4. Remove exact duplicate rows and wholly empty columns.
5. Parse numeric outcomes/predictors; normalize categorical missing tokens (`Blank`, `Not Applicable`, etc.).


### Schema differences across survey years

NHAMCS ED questionnaires evolved substantially from 1992 to 2022. After stacking files,
the combined dataset uses **20 standardized fields** (plus derived
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
(24 of
31 files). Years **1992–1996**, **2001–2002**, and early files without a
`WAITTIME`/`waittime` field appear as missing for the outcome variable.

Files included: **31** (`1992`–`2022`).


## 2. Missing Values

| Metric | Value |
|---|---|
| Total cells (key columns) | 15,270,280 |
| Total missing values | 3,271,412 |
| Overall missing rate | 21.42% |

### Per-column missingness (project-relevant columns)

| column              | dtype   |   n_missing |   pct_missing |   n_non_missing |
|:--------------------|:--------|------------:|--------------:|----------------:|
| disposition         | str     |      570997 |         74.79 |          192517 |
| pain_scale          | float64 |      539124 |         70.61 |          224390 |
| residence           | str     |      318128 |         41.67 |          445386 |
| wait_time_min       | float64 |      253386 |         33.19 |          510128 |
| systolic_bp         | float64 |      230048 |         30.13 |          533466 |
| length_of_visit_min | float64 |      229324 |         30.04 |          534190 |
| seen_72h            | str     |      221679 |         29.03 |          541835 |
| pulse               | float64 |      194086 |         25.42 |          569428 |
| arrival_mode        | str     |      160403 |         21.01 |          603111 |
| admit_observation   | str     |      152793 |         20.01 |          610721 |
| triage_immediacy    | str     |      135568 |         17.76 |          627946 |
| payment_type        | str     |      111047 |         14.54 |          652467 |
| race                | str     |       62126 |          8.14 |          701388 |
| visit_day_of_week   | str     |       59818 |          7.83 |          703696 |
| ethnicity           | str     |       31044 |          4.07 |          732470 |
| age_years           | float64 |        1841 |          0.24 |          761673 |
| admit_hospital      | str     |           0 |          0    |          763514 |
| sex                 | str     |           0 |          0    |          763514 |
| visit_month         | str     |           0 |          0    |          763514 |
| survey_year         | int64   |           0 |          0    |          763514 |

## 3. Outlier Analysis (IQR method, numeric columns)

Outliers defined as values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR.

| column              |   n_valid |   min |   q1 |   median |   q3 |   max |   iqr_lower |   iqr_upper |   n_outliers_iqr |   pct_outliers_iqr |
|:--------------------|----------:|------:|-----:|---------:|-----:|------:|------------:|------------:|-----------------:|-------------------:|
| wait_time_min       |    510128 |     0 |   10 |       25 |   55 |  1440 |       -57.5 |       122.5 |            43439 |         8.51531    |
| age_years           |    761673 |     0 |   19 |       34 |   54 |   107 |       -33.5 |       106.5 |                1 |         0.00013129 |
| length_of_visit_min |    534190 |     0 |   87 |      151 |  253 |  5760 |      -162   |       502   |            35375 |         6.62218    |
| pulse               |    569428 |     0 |   76 |       88 |  102 |   244 |        37   |       141   |            24550 |         4.31134    |
| systolic_bp         |    533466 |     0 |  116 |      130 |  146 |   290 |        71   |       191   |            12517 |         2.34635    |
| pain_scale          |    224390 |     0 |    0 |        5 |    8 |    10 |       -12   |        20   |                0 |         0          |

## 4. Covariates / Predictors

**Total planned predictor fields: 19** (excluding dependent variable `wait_time_min`).

**Demographic (7):** `age_years`, `sex`, `race`, `ethnicity`, `residence`, `visit_month`, `visit_day_of_week`
**Clinical (10):** `triage_immediacy`, `pulse`, `systolic_bp`, `pain_scale`, `seen_72h`, `length_of_visit_min`, `admit_hospital`, `admit_observation`, `disposition`, `disposition_category`
**Access (3):** `arrival_mode`, `payment_type`, `survey_year`

These fields support the research question: *Are Emergency Department wait times determined by medical need?*
Triage immediacy (`triage_immediacy`), vitals, pain, and admission/disposition variables proxy medical acuity;
demographics and access variables capture non-clinical factors.

## 5. Dependent Variable (Wait-Time ML Project)

| Field | Description |
|---|---|
| **`wait_time_min`** | Wait time to first provider contact (minutes). Mapped from NHAMCS `WAITTIME` across years. |
| Non-missing observations | 510,128 (66.8% of combined rows) |
| Summary stats range (0–480 min) | 507,949 visits |
| Median (0–480 min) | 25 minutes |
| Mean (0–480 min) | 44.2 minutes |
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
![admit_rate_by_immediacy](../outputs/visualizations/admit_rate_by_immediacy.png)
![admit_rate_by_arrivalmode](../outputs/visualizations/admit_rate_by_arrivalmode.png)
![admit_rate_by_insurance_status](../outputs/visualizations/admit_rate_by_insurance_status.png)
![admit_rate_by_gender](../outputs/visualizations/admit_rate_by_gender.png)
![admit_rate_by_race](../outputs/visualizations/admit_rate_by_race.png)
![admit_rate_by_ethnicity](../outputs/visualizations/admit_rate_by_ethnicity.png)
![wait_time_yearly_trend](../outputs/visualizations/wait_time_yearly_trend.png)

---

*Generated by `scripts/dataset_exploration.py`. Re-run to reproduce statistics and figures.*
