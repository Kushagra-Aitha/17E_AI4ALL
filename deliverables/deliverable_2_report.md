# Deliverable #2: Dataset Exploration Report

## Emergency Department Wait Times & Medical Need

**Research question:** *Are Emergency Department wait times determined by medical need?*
**Team:** Kushagra Aitha, Rupsa Bose, Miskatul Moon, Shahriar Shabib, Sofi Le

---

> **Data source note:** The assignment references the Yale Admission GitHub repository. Our team uses the **National Hospital Ambulatory Medical Care Survey (NHAMCS) Emergency Department** public-use files stored in this project's `NHAMCS_Data/` folder (`NHAMCS_ED_2007_decoded.csv` – `NHAMCS_ED_2022_decoded.csv`). NHAMCS is a nationally representative sample of ED visits and is well suited to studying how clinical acuity, demographics, and access factors relate to wait times.

---

## 1. Dataset Overview

We combined the 16 available NHAMCS ED annual files (2007–2022), harmonized variable names across the one questionnaire transition in this range (around 2009), and applied light cleaning before exploration.

| Item | Value |
| --- | --- |
| **Source** | NHAMCS Emergency Department public-use files (decoded CSV) |
| **Files used** | **16** annual files (`NHAMCS_ED_2007_decoded.csv` – `NHAMCS_ED_2022_decoded.csv`) |
| **Survey years** | 2007–2022 |
| **Raw rows (stacked)** | 392,761 |
| **Duplicates removed** | 163 |
| **Empty columns removed** | 0 |
| **Final rows (after cleaning)** | **392,598** |
| **Final columns (standardized + derived)** | **21** |

### Combining & cleaning approach

1. Load each annual `NHAMCS_ED_YYYY_decoded.csv` with year-specific column resolution (field names changed once in this range, around 2009).
2. Rename fields to a common schema (`wait_time_min`, `triage_immediacy`, `payment_type`, etc.).
3. Append a `survey_year` column when stacking files.
4. Remove exact duplicate rows and wholly empty columns.
5. Parse numeric outcomes and predictors; normalize categorical missing tokens (`Blank`, `Not Applicable`, `Unknown`, etc.).
6. Harmonize `triage_immediacy` and `arrival_mode` labels across the pre-/post-2009 questionnaire eras (see caveat below) and derive `disposition_category` for visualization.

**Cross-year caveat:** within 2007–2022, NHAMCS switched several field names and codings around 2009: triage moved from `IMMED` to `IMMEDR`, arrival mode from a categorical `ARRIVE` field to a binary "arrived by ambulance" `ARREMS` field, and pain from a categorical `PAIN` field to a numeric 0–10 `PAINSCALE`. `wait_time_min` itself is populated in **all 16 of 16** files in this range, which is exactly why the modeling pipeline (`scripts/preprocess.py`) restricts to 2007–2022 rather than reaching back further — earlier years have real gaps in the outcome variable and additional questionnaire eras to reconcile.

*Reproducibility:* `scripts/dataset_exploration.py` (EDA) and `scripts/preprocess.py` (modeling pipeline) · Supporting files: `deliverables/dataset_exploration.md`, `deliverables/exploration_stats.json`, `data/processed/preprocessing_report.json`

---

## 2. Missing Values / Null Values

### Overall missingness

| Metric | Value |
| --- | --- |
| Total cells analyzed (20 key project columns × 392,598 rows) | **7,851,960** |
| **Total missing / null values** | **962,916** |
| **Overall missing rate** | **12.26%** |

Missing values arise from (a) fields not collected in every questionnaire era within 2007–2022 (e.g. `pain_scale` only from 2011+), (b) NHAMCS skip patterns and "not applicable" responses, and (c) item non-response at the visit level.

### Per-column missingness

| Column | Type | Missing Count | Missing % | Non-Missing Count |
| --- | --- | --- | --- | --- |
| `disposition` | categorical | 352,874 | 89.88% | 39,724 |
| `pain_scale` | numeric | 168,208 | 42.84% | 224,390 |
| `triage_immediacy` | categorical | 76,321 | 19.44% | 316,277 |
| `race` | categorical | 62,126 | 15.82% | 330,472 |
| **`wait_time_min`** | **numeric (outcome)** | **60,276** | **15.35%** | **332,322** |
| `length_of_visit_min` | numeric | 55,324 | 14.09% | 337,274 |
| `seen_72h` | categorical | 49,944 | 12.72% | 342,654 |
| `systolic_bp` | numeric | 46,260 | 11.78% | 346,338 |
| `payment_type` | categorical | 31,227 | 7.95% | 361,371 |
| `pulse` | numeric | 27,804 | 7.08% | 364,794 |
| `arrival_mode` | categorical | 17,218 | 4.39% | 375,380 |
| `residence` | categorical | 13,685 | 3.49% | 378,913 |
| `age_years` | numeric | 1,649 | 0.42% | 390,949 |
| `visit_day_of_week` | categorical | 0 | 0.00% | 392,598 |
| `visit_month` | categorical | 0 | 0.00% | 392,598 |
| `survey_year` | numeric | 0 | 0.00% | 392,598 |
| `admit_hospital` | categorical | 0 | 0.00% | 392,598 |
| `ethnicity` | categorical | 0 | 0.00% | 392,598 |
| `sex` | categorical | 0 | 0.00% | 392,598 |
| `admit_observation` | categorical | 0 | 0.00% | 392,598 |

### Brief interpretation

- **Core demographics** (`age_years`, `sex`, `visit_month`, `visit_day_of_week`, `ethnicity`) are nearly or fully complete and reliable for modeling.
- **Clinical acuity fields** show moderate-to-heavy missingness: triage immediacy (~19% missing), vitals (~7–12% missing), and pain scale (~43% missing, since `PAINSCALE` was only introduced in 2011).
- **The outcome variable** `wait_time_min` is missing for **15.35%** of visits within this 2007–2022 range — much less severe than it would be if earlier survey years (with real `WAITTIME` field gaps) were included.
- **Disposition** is the sparsest field (~90% missing) and, along with `admit_hospital`/`admit_observation`/`length_of_visit_min`, is excluded from the final model features regardless of missingness — these are post-visit outcomes, not predictors known at triage time (see Section 4).

Modeling handles this missingness explicitly rather than assuming full coverage: `scripts/preprocess.py` routes every missing categorical value into its own `Unknown` one-hot level (so it doesn't silently collapse into the reference category), adds `_missing` indicator flags before median-imputing `pulse`/`systolic_bp`, and drops the small remainder of rows still missing `age_years` (0.24%) as a negligible complete-case restriction.

---

## 3. Outliers

### Are there outlier values?

**Yes.** Numeric columns contain values outside the interquartile range (IQR) bounds (below Q1 − 1.5×IQR or above Q3 + 1.5×IQR). Many of these reflect real but extreme clinical situations (very long waits, prolonged ED stays, abnormal vitals) rather than data-entry errors.

**Outliers were not removed** from the EDA dataset — all observed values are retained for exploration. The modeling pipeline does apply one deliberate bound: `wait_time_min` is clamped to 480 minutes (8 hours) for the regression target, since values above that are rare and disproportionately influence a linear fit; the unclamped values are kept in the cleaned pre-modeling file.

### Outlier summary (IQR method)

| Column | Valid *n* | Min | Q1 | Median | Q3 | Max | IQR Lower | IQR Upper | Outlier Count | Outlier % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `wait_time_min` | 332,322 | 0 | 9 | 23 | 53 | 1,440 | −57.0 | 119.0 | 28,351 | 8.53% |
| `length_of_visit_min` | 337,274 | 0 | 92 | 159 | 264 | 5,760 | −166.0 | 522.0 | 21,811 | 6.47% |
| `pulse` | 364,794 | 0 | 76 | 88 | 102 | 244 | 37.0 | 141.0 | 14,671 | 4.02% |
| `systolic_bp` | 346,338 | 0 | 117 | 131 | 146 | 290 | 73.5 | 189.5 | 8,859 | 2.56% |
| `age_years` | 390,949 | 0.5 | 19 | 35 | 55 | 99 | −35.0 | 109.0 | 0 | 0.00% |
| `pain_scale` | 224,390 | 0 | 0 | 5 | 8 | 10 | −12.0 | 20.0 | 0 | 0.00% |

**Notable patterns:** wait times extend up to **1,440 minutes** (24 hours); length of visit reaches **5,760 minutes** (4 days). Zero values in `pulse` and `systolic_bp` almost certainly represent "not recorded" rather than physiologic zeros — the modeling pipeline treats `pulse == 0` and `systolic_bp == 0` as missing rather than as real vital signs, then median-imputes them with a `_missing` flag preserved. A small number of physiologically implausible non-zero low readings remain (pulse ≤ 5 in 32 rows, systolic BP ≤ 10 in 3 rows out of 332k) — negligible in count, not worth a separate filtering pass.

---

## 4. Covariates / Predictors

### How many potential covariates can be used?

After standardizing the combined dataset, we identified **19 planned predictor fields** (excluding the dependent variable `wait_time_min`) during EDA. `survey_year` is included as a temporal control in all models.

### Planned covariate list (by category)

#### Demographic (7)

| Variable | Rationale |
| --- | --- |
| `age_years` | Age affects acuity presentation and ED resource use. |
| `sex` | Sex differences in presentation and admission patterns are well documented. |
| `race` | Captures potential disparities in ED throughput unrelated to clinical need. |
| `ethnicity` | Complements race for equity-focused access analysis. |
| `residence` | Care-setting context (e.g. private residence vs. nursing home vs. homeless). |
| `visit_month` | Seasonal variation in ED volume and staffing. |
| `visit_day_of_week` | Weekend/weekday crowding may affect wait times independently of acuity. |

#### Clinical / medical need (10)

| Variable | Rationale |
| --- | --- |
| `triage_immediacy` | Primary proxy for medical urgency. |
| `pulse` | Initial heart rate reflects physiologic stress. |
| `systolic_bp` | Blood pressure indicates hemodynamic status. |
| `pain_scale` | Patient-reported pain intensity (0–10, 2011+ only). |
| `seen_72h` | Prior ED/urgent-care use may signal chronic or recurring conditions. |
| `length_of_visit_min` | Total ED stay; correlated with complexity of workup. |
| `admit_hospital` | Hospital admission indicates higher acuity outcomes. |
| `admit_observation` | Observation status signals intermediate acuity. |
| `disposition` | Raw ED disposition code. |
| `disposition_category` | Derived grouping for interpretable disposition analysis. |

#### Access to care (2 primary + 1 control)

| Variable | Rationale |
| --- | --- |
| `arrival_mode` | Ambulance vs. walk-in affects triage pathway and wait dynamics. |
| `payment_type` | Insurance/payer type proxies financial access and care navigation. |
| `survey_year` *(control)* | Controls for secular trends in ED operations and survey design. |

These covariates directly support testing whether **clinical need** (triage, vitals, pain, disposition) explains wait-time variation after accounting for **demographics** and **access** factors.

### Note — planned vs. final modeling features

The 19 fields above are the full candidate set identified during EDA. The modeling pipeline (`scripts/preprocess.py`) narrows this: `admit_hospital`, `admit_observation`, `disposition`, `disposition_category`, and `length_of_visit_min` are **excluded as post-visit outcomes** — they aren't known at triage time, so including them as predictors would leak the outcome into the model. `pain_scale` is also excluded (43% missing here, and only collected from 2011 onward). After one-hot encoding the remaining demographic, access, and clinical fields, the two model-ready feature matrices are:

| Feature set | Predictors | Encoded columns | File |
| --- | --- | --- | --- |
| **Model 1** (demographic + access) | 13 raw fields → | 45 | `data/processed/nhamcs_model1_encoded.csv` |
| **Model 2** (+ clinical) | 15 raw fields → | 54 | `data/processed/nhamcs_model2_encoded.csv` |

Full details in `MODELING.md` and `data/processed/preprocessing_report.json`.

---

## 5. Dependent Variable

### What is being predicted?

The primary modeling target is **`wait_time_min`** — minutes from ED arrival to first contact with a physician or advanced practice provider, harmonized from NHAMCS `WAITTIME` across the two questionnaire eras in this range. Modeling uses `log_wait_time = log(wait_time_min + 1)`, since the raw variable is heavily right-skewed (skewness 5.89, reduced to −0.33 after the log transform).

### Distribution statistics

| Statistic | Value |
| --- | --- |
| Non-missing observations | 332,322 (84.6% of all rows) |
| Missing | 60,276 (**15.35%**) |
| Valid range used for summary (0–480 min) | 330,682 visits |
| **Median** | **22 minutes** |
| **Mean** | **42.1 minutes** |
| Range (raw, all non-missing) | 0 – 1,440 minutes |

The distribution is **right-skewed**: most patients are seen within roughly 22 minutes (median), but a long tail of extended waits pulls the mean substantially higher.

### Secondary variables (exploratory / supplementary outcomes)

| Variable | Role |
| --- | --- |
| `admit_hospital` | Binary admission outcome; links acuity to disposition. |
| `admit_observation` | Observation-unit placement; intermediate acuity signal. |
| `disposition` / `disposition_category` | How the ED visit resolved (discharged, admitted, transferred, etc.). |
| `length_of_visit_min` | Total time in the ED; distinct from wait-to-provider but related to system load. |

These secondary variables help interpret whether patients who wait longer differ in clinical outcomes, not just in queue position. They are excluded from the wait-time model's predictors for the leakage reason noted in Section 4.

---

## 6. Visualizations

Sixteen figures are generated in `outputs/visualizations/`. Chart types include **histogram**, **bar charts**, **boxplot**, **scatter plot**, **correlation matrix (heatmap)**, and **line graph**. Visualizations emphasizing the outcome–predictor relationship are noted below, with the two new charts added to give race and arrival mode — the two strongest predictors found once modeling was run (Section 7) — direct visual coverage.

---

### 6.1 Wait Time Distribution

**Chart type:** Histogram
**File:** `wait_time_distribution.png`

Wait times are heavily concentrated below ~60 minutes, with a long right tail extending toward multi-hour waits. The median (22 min) sits well below the bulk of the distribution's upper range, confirming right-skewness in the outcome variable.

---

### 6.2 ED Disposition Categories

**Chart type:** Horizontal bar chart
**File:** `target_disposition_bar_chart.png`

The majority of coded visits end in routine discharge or related non-admission categories; hospital admission and transfer represent smaller but clinically important subsets. This frames the secondary outcome landscape for acuity analysis.

---

### 6.3 Age by Disposition

**Chart type:** Boxplot
**File:** `age_by_disposition_boxplot.png`

Admitted and higher-acuity disposition groups tend toward older median ages than routine discharge groups, suggesting age and disposition are linked — a pattern our model must disentangle from wait-time effects.

---

### 6.4 Age vs. Heart Rate by Admission Status

**Chart type:** Scatter plot (colored by admission status)
**File:** `scatter_age_vs_pulse_by_disposition.png`

Younger patients show wider pulse variability; admitted patients cluster at higher heart rates across ages. This supports using vitals alongside demographics when proxying medical need.

---

### 6.5 Correlation Matrix — Selected Numeric Variables

**Chart type:** Correlation matrix (heatmap)
**File:** `correlation_matrix_selected_numeric.png`

`wait_time_min` shows **weak linear correlation** with clinical numerics (`pulse` r≈−0.02, `systolic_bp` r≈−0.02) and `age_years` (r≈−0.01); `survey_year` has the strongest — still weak — relationship (r≈−0.11). Weak individual correlations don't rule out a real multivariable effect once other variables are controlled for, which is exactly what regression modeling (Section 7) tests.

---

### 6.6 Mean Wait Time by Triage Immediacy *(outcome vs. predictor)*

**Chart type:** Bar chart
**File:** `wait_time_by_triage_bar.png`

Mean wait times vary across triage categories but do not follow a strict monotonic "sickest first" ordering in every category — an early signal that **medical need alone may not fully determine queue position**, motivating multivariable modeling.

---

### 6.7 Mean Wait Time by Payment Type *(outcome vs. predictor)*

**Chart type:** Horizontal bar chart
**File:** `wait_time_by_payment_bar.png`

Mean wait times differ across payer categories (e.g. private insurance, Medicaid, self-pay, workers' compensation). This suggests **access and insurance status** may influence wait times independent of triage.

---

### 6.8 Mean Wait Time by Race *(outcome vs. predictor — new)*

**Chart type:** Horizontal bar chart
**File:** `wait_time_by_race_bar.png`

Patients reporting more than one race and Black/African American patients have the longest mean waits; White, Asian, and Native Hawaiian/Pacific Islander patients cluster lower. This raw gap is the central pattern the demographic-vs-clinical model comparison (Section 7) tests for a clinical explanation.

---

### 6.9 Mean Wait Time by Arrival Mode *(outcome vs. predictor — new)*

**Chart type:** Horizontal bar chart
**File:** `wait_time_by_arrivalmode_bar.png`

Ambulance arrivals wait shortest, consistent with arriving with documented acuity already known to staff. Personal transportation and non-ambulance public-service arrivals wait considerably longer than walk-in — the largest access-related gap found anywhere in this dataset.

---

### 6.10 Admission Rate by Triage Immediacy

**Chart type:** Bar chart
**File:** `admit_rate_by_immediacy.png`

Admission rates rise sharply from nonurgent to immediate triage levels, validating `triage_immediacy` as a strong clinical-need proxy for downstream disposition.

---

### 6.11 Admission Rate by Arrival Mode

**Chart type:** Bar chart
**File:** `admit_rate_by_arrivalmode.png`

Ambulance arrivals are admitted at higher rates than walk-in or other modes, reflecting pre-hospital triage and higher baseline acuity among EMS patients.

---

### 6.12 Admission Rate by Insurance / Payment Type

**Chart type:** Bar chart
**File:** `admit_rate_by_insurance_status.png`

Admission rates vary by expected payer, indicating that **access factors** co-vary with clinical outcomes and should be included when testing whether wait times track medical need.

---

### 6.13 Admission Rate by Sex

**Chart type:** Bar chart
**File:** `admit_rate_by_gender.png`

Male and female patients show modestly different admission rates, supporting inclusion of `sex` as a demographic control in wait-time models.

---

### 6.14 Admission Rate by Race

**Chart type:** Bar chart
**File:** `admit_rate_by_race.png`

Admission rates differ across racial groups, highlighting potential equity dimensions that may confound the relationship between wait times and clinical acuity.

---

### 6.15 Admission Rate by Ethnicity

**Chart type:** Bar chart
**File:** `admit_rate_by_ethnicity.png`

Hispanic vs. non-Hispanic patients show distinct admission patterns, reinforcing the need for ethnicity as a covariate when assessing whether waits reflect medical need or social/access factors.

---

### 6.16 ED Wait Time Trends by Survey Year

**Chart type:** Line graph (mean and median)
**File:** `wait_time_yearly_trend.png`

Both mean and median wait times trend downward over 2007–2022 (mean ~47 min → ~30 min), with mean consistently above median and a sharp dip in 2020 (likely COVID-era ED volume changes) before a partial rebound. Temporal trends underscore the importance of `survey_year` as a control when comparing acuity and access effects across years.

---

## 7. Key Conclusions

Tying findings back to: *Are ED wait times determined by medical need?* This section now reflects the completed OLS modeling in `notebooks/linear_regression.ipynb`, not just EDA.

1. **The outcome is usable and mostly complete.** 84.6% of visits (332,322) have a valid `wait_time_min` within 2007–2022. The distribution is right-skewed (median 22 min, mean 42.1 min), which is why the modeling target is `log_wait_time` rather than raw minutes.
2. **Clinical-need proxies are present but imperfectly aligned with wait times.** Triage immediacy is available for most visits and strongly predicts admission, yet mean wait times by triage category do not show a clean "higher acuity → shorter wait" pattern (Section 6.6). Medical need appears **necessary but not sufficient** to explain waits.
3. **Access and demographic factors show independent variation — and this is no longer just a hypothesis.** Fitting OLS on Model 1 (demographic + access) vs. Model 2 (+ clinical) shows R² barely moves (0.074 → 0.079) and the race/ethnicity/payment coefficients barely shrink (largest shift 0.009, vs. total effects up to 0.23) when clinical controls are added. Black patients wait ~23–26% longer than White patients, multiracial patients ~23–27% longer, and No charge/Charity patients ~18% longer than privately insured — all highly significant (p<0.001) **before and after** controlling for triage acuity and vitals.
4. **Missing data and outliers require deliberate handling, not deletion.** ~12% of cells are missing overall; disposition and pain scale are especially sparse. Numeric outliers (e.g. waits up to 1,440 min) were **retained** in the cleaned dataset; the modeling target clamps at 480 min rather than deleting extreme rows.
5. **Multivariable regression was warranted, and confirmed the concern raised by the EDA.** Weak individual correlations between wait time and numeric predictors did not mean no effect existed — the demographic and payment coefficients are precise and persistent even though R² is low, which is expected for this outcome (individual-level NHAMCS fields can't capture hospital-level drivers like staffing or bed availability).

---

*Generated from `scripts/dataset_exploration.py` (EDA), `scripts/preprocess.py` (modeling pipeline), and `notebooks/linear_regression.ipynb` (OLS modeling). Re-run each to reproduce statistics, figures, and model results.*
