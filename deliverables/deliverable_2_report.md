# Deliverable #2: Dataset Exploration Report

## Emergency Department Wait Times & Medical Need

**Course check-in:** Wednesday  
**Research question:** *Are Emergency Department wait times determined by medical need?*  
**Team:** ED Wait-Time ML Project

---

> **Data source note:** The assignment references the Yale Admission GitHub repository. Our team uses the **National Hospital Ambulatory Medical Care Survey (NHAMCS) Emergency Department** public-use files stored in this project (`NHAMCS_ED_1992_decoded.csv` – `NHAMCS_ED_2022_decoded.csv`). NHAMCS is a nationally representative sample of ED visits and is well suited to studying how clinical acuity, demographics, and access factors relate to wait times.

---

## 1. Dataset Overview

We combined all available NHAMCS ED annual files, harmonized variable names across survey years, and applied light cleaning before exploration.


| Item                                       | Value                                                                             |
| ------------------------------------------ | --------------------------------------------------------------------------------- |
| **Source**                                 | NHAMCS Emergency Department public-use files (decoded CSV)                        |
| **Files used**                             | **31** annual files (`NHAMCS_ED_1992_decoded.csv` – `NHAMCS_ED_2022_decoded.csv`) |
| **Survey years**                           | 1992–2022                                                                         |
| **Raw rows (stacked)**                     | 839,772                                                                           |
| **Duplicates removed**                     | 76,258                                                                            |
| **Empty columns removed**                  | 0                                                                                 |
| **Final rows (after cleaning)**            | **763,514**                                                                       |
| **Final columns (standardized + derived)** | **21**                                                                            |


### Combining & cleaning approach

1. Load each annual `NHAMCS_ED_YYYY_decoded.csv` with year-specific column resolution (field names changed across NHAMCS cycles).
2. Rename fields to a common schema (`wait_time_min`, `triage_immediacy`, `payment_type`, etc.).
3. Append a `survey_year` column when stacking files.
4. Remove exact duplicate rows and wholly empty columns.
5. Parse numeric outcomes and predictors; normalize categorical missing tokens (`Blank`, `Not Applicable`, `Unknown`, etc.).
6. Derive `disposition_category` from raw disposition codes for visualization.

**Cross-year caveat:** NHAMCS questionnaires evolved substantially from 1992 to 2022. Wait time (`wait_time_min`) is available in **24 of 31** survey years (1997–2000, 2003–2006, 2007–2022); years without a `WAITTIME`/`waittime` field contribute missing values for the outcome variable.

*Reproducibility:* `scripts/dataset_exploration.py` · Supporting tables: `deliverables/dataset_exploration.md`, `deliverables/exploration_stats.json`

---

## 2. Missing Values / Null Values

### Overall missingness


| Metric                                                       | Value          |
| ------------------------------------------------------------ | -------------- |
| Total cells analyzed (20 key project columns × 763,514 rows) | **15,270,280** |
| **Total missing / null values**                              | **3,271,412**  |
| **Overall missing rate**                                     | **21.42%**     |


Missing values arise from (a) variables not collected in early survey years, (b) NHAMCS skip patterns and “not applicable” responses, and (c) item non-response at the visit level.

### Per-column missingness


| Column                | Type                  | Missing Count | Missing %  | Non-Missing Count |
| --------------------- | --------------------- | ------------- | ---------- | ----------------- |
| `disposition`         | categorical           | 570,997       | 74.79%     | 192,517           |
| `pain_scale`          | numeric               | 539,124       | 70.61%     | 224,390           |
| `residence`           | categorical           | 318,128       | 41.67%     | 445,386           |
| `**wait_time_min`**   | **numeric (outcome)** | **253,386**   | **33.19%** | **510,128**       |
| `systolic_bp`         | numeric               | 230,048       | 30.13%     | 533,466           |
| `length_of_visit_min` | numeric               | 229,324       | 30.04%     | 534,190           |
| `seen_72h`            | categorical           | 221,679       | 29.03%     | 541,835           |
| `pulse`               | numeric               | 194,086       | 25.42%     | 569,428           |
| `arrival_mode`        | categorical           | 160,403       | 21.01%     | 603,111           |
| `admit_observation`   | categorical           | 152,793       | 20.01%     | 610,721           |
| `triage_immediacy`    | categorical           | 135,568       | 17.76%     | 627,946           |
| `payment_type`        | categorical           | 111,047       | 14.54%     | 652,467           |
| `race`                | categorical           | 62,126        | 8.14%      | 701,388           |
| `visit_day_of_week`   | categorical           | 59,818        | 7.83%      | 703,696           |
| `ethnicity`           | categorical           | 31,044        | 4.07%      | 732,470           |
| `age_years`           | numeric               | 1,841         | 0.24%      | 761,673           |
| `admit_hospital`      | categorical           | 0             | 0.00%      | 763,514           |
| `sex`                 | categorical           | 0             | 0.00%      | 763,514           |
| `visit_month`         | categorical           | 0             | 0.00%      | 763,514           |
| `survey_year`         | numeric               | 0             | 0.00%      | 763,514           |


### Brief interpretation

- **Core demographics** (`age_years`, `sex`, `visit_month`, `visit_day_of_week`) are nearly complete and reliable for modeling.
- **Clinical acuity fields** show moderate missingness: triage immediacy (~~18% missing), vitals (~~25–30% missing), and pain scale (~71% missing due to later introduction of the 0–10 pain scale).
- **The outcome variable** `wait_time_min` is missing for **33.19%** of visits, concentrated in years without a wait-time field and in item non-response.
- **Disposition** is the sparsest field (~75% missing) because detailed disposition coding was not uniformly available across all survey cycles.

Modeling will require explicit handling of missing data (e.g., complete-case subsets for specific analyses, indicator variables, or imputation) rather than assuming full coverage.

---

## 3. Outliers

### Are there outlier values?

**Yes.** Numeric columns contain values outside the interquartile range (IQR) bounds (below Q1 − 1.5×IQR or above Q3 + 1.5×IQR). Many of these reflect real but extreme clinical situations (very long waits, prolonged ED stays, abnormal vitals) rather than data-entry errors.

**Outliers were not removed.** Per project protocol, all observed values are retained for exploration and modeling; downstream models may apply winsorization or robust methods if needed.

### Outlier summary (IQR method)


| Column                | Valid *n* | Min | Q1  | Median | Q3  | Max   | IQR Lower | IQR Upper | Outlier Count | Outlier % |
| --------------------- | --------- | --- | --- | ------ | --- | ----- | --------- | --------- | ------------- | --------- |
| `wait_time_min`       | 510,128   | 0   | 10  | 25     | 55  | 1,440 | −57.5     | 122.5     | 43,439        | 8.52%     |
| `length_of_visit_min` | 534,190   | 0   | 87  | 151    | 253 | 5,760 | −162.0    | 502.0     | 35,375        | 6.62%     |
| `pulse`               | 569,428   | 0   | 76  | 88     | 102 | 244   | 37.0      | 141.0     | 24,550        | 4.31%     |
| `systolic_bp`         | 533,466   | 0   | 116 | 130    | 146 | 290   | 71.0      | 191.0     | 12,517        | 2.35%     |
| `age_years`           | 761,673   | 0   | 19  | 34     | 54  | 107   | −33.5     | 106.5     | 1             | 0.0001%   |
| `pain_scale`          | 224,390   | 0   | 0   | 5      | 8   | 10    | −12.0     | 20.0      | 0             | 0.00%     |


**Notable patterns:** Wait times extend up to **1,440 minutes** (24 hours); length of visit reaches **5,760 minutes**. Zero values in vitals and blood pressure likely represent “not recorded” rather than physiologic zeros and will be treated carefully in modeling.

---

## 4. Covariates / Predictors

### How many potential covariates can be used?

After standardizing the combined dataset, we identified **19 planned predictor fields** (excluding the dependent variable `wait_time_min`). An additional temporal control (`survey_year`) may be included in robustness checks.

### Planned covariate list (by category)

#### Demographic (7)


| Variable            | Rationale                                                                   |
| ------------------- | --------------------------------------------------------------------------- |
| `age_years`         | Age affects acuity presentation and ED resource use.                        |
| `sex`               | Sex differences in presentation and admission patterns are well documented. |
| `race`              | Captures potential disparities in ED throughput unrelated to clinical need. |
| `ethnicity`         | Complements race for equity-focused access analysis.                        |
| `residence`         | Urban vs. rural residence proxies care-setting context.                     |
| `visit_month`       | Seasonal variation in ED volume and staffing.                               |
| `visit_day_of_week` | Weekend/weekday crowding may affect wait times independently of acuity.     |


#### Clinical / medical need (10)


| Variable               | Rationale                                                            |
| ---------------------- | -------------------------------------------------------------------- |
| `triage_immediacy`     | Primary proxy for medical urgency (Immediate → Nonurgent).           |
| `pulse`                | Initial heart rate reflects physiologic stress.                      |
| `systolic_bp`          | Blood pressure indicates hemodynamic status.                         |
| `pain_scale`           | Patient-reported pain intensity (0–10, when available).              |
| `seen_72h`             | Prior ED/urgent-care use may signal chronic or recurring conditions. |
| `length_of_visit_min`  | Total ED stay; correlated with complexity of workup.                 |
| `admit_hospital`       | Hospital admission indicates higher acuity outcomes.                 |
| `admit_observation`    | Observation status signals intermediate acuity.                      |
| `disposition`          | Raw ED disposition code.                                             |
| `disposition_category` | Derived grouping for interpretable disposition analysis.             |


#### Access to care (2 primary + 1 optional control)


| Variable                   | Rationale                                                          |
| -------------------------- | ------------------------------------------------------------------ |
| `arrival_mode`             | Ambulance vs. walk-in affects triage pathway and wait dynamics.    |
| `payment_type`             | Insurance/payer type proxies financial access and care navigation. |
| `survey_year` *(optional)* | Controls for secular trends in ED operations and survey design.    |


These covariates directly support testing whether **clinical need** (triage, vitals, pain, disposition) explains wait-time variation after accounting for **demographics** and **access** factors.

---

## 5. Dependent Variable

### What is being predicted?

The primary modeling target is `**wait_time_min`** — minutes from ED arrival to **first contact with a physician or advanced practice provider**, harmonized from NHAMCS `WAITTIME` / `waittime` fields across survey years.

### Distribution statistics


| Statistic                                | Value                       |
| ---------------------------------------- | --------------------------- |
| Non-missing observations                 | 510,128 (66.8% of all rows) |
| Missing                                  | 253,386 (**33.19%**)        |
| Valid range used for summary (0–480 min) | 507,949 visits              |
| **Median**                               | **25 minutes**              |
| **Mean**                                 | **44.2 minutes**            |
| Range (raw, all non-missing)             | 0 – 1,440 minutes           |


The distribution is **right-skewed**: most patients are seen within roughly 25 minutes (median), but a long tail of extended waits pulls the mean substantially higher.

### Secondary variables (exploratory / supplementary outcomes)


| Variable                               | Role                                                                             |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| `admit_hospital`                       | Binary admission outcome; links acuity to disposition.                           |
| `admit_observation`                    | Observation-unit placement; intermediate acuity signal.                          |
| `disposition` / `disposition_category` | How the ED visit resolved (discharged, admitted, transferred, etc.).             |
| `length_of_visit_min`                  | Total time in the ED; distinct from wait-to-provider but related to system load. |


These secondary variables help interpret whether patients who wait longer differ in clinical outcomes, not just in queue position.

---

## 6. Visualizations

Fourteen figures were generated in `outputs/visualizations/`. Chart types include **histogram**, **bar charts**, **boxplot**, **scatter plot**, **correlation matrix (heatmap)**, and **line graph**. Visualizations emphasizing the outcome–predictor relationship are noted below.

---

### 6.1 Wait Time Distribution

**Chart type:** Histogram  
**File:** `wait_time_distribution.png`

Distribution of ED wait time

Wait times are heavily concentrated below ~60 minutes, with a long right tail extending toward multi-hour waits. The median (25 min) sits well below the bulk of the distribution’s upper range, confirming right-skewness in the outcome variable.

---

### 6.2 ED Disposition Categories

**Chart type:** Horizontal bar chart  
**File:** `target_disposition_bar_chart.png`

Top ED disposition categories

The majority of coded visits end in routine discharge or related non-admission categories; hospital admission and transfer represent smaller but clinically important subsets. This frames the secondary outcome landscape for acuity analysis.

---

### 6.3 Age by Disposition

**Chart type:** Boxplot  
**File:** `age_by_disposition_boxplot.png`

Patient age by ED disposition

Admitted and higher-acuity disposition groups tend toward older median ages than routine discharge groups, suggesting age and disposition are linked — a pattern our model must disentangle from wait-time effects.

---

### 6.4 Age vs. Heart Rate by Admission Status

**Chart type:** Scatter plot (colored by admission status)  
**File:** `scatter_age_vs_pulse_by_disposition.png`

Age vs heart rate by admission status

Younger patients show wider pulse variability; admitted patients cluster at higher heart rates across ages. This supports using vitals alongside demographics when proxying medical need.

---

### 6.5 Correlation Matrix — Selected Numeric Variables

**Chart type:** Correlation matrix (heatmap)  
**File:** `correlation_matrix_selected_numeric.png`

Correlation matrix of numeric variables

`wait_time_min` shows **weak linear correlation** with clinical numerics (`pulse`, `systolic_bp`, `pain_scale`) and `age_years`, while `length_of_visit_min` is more strongly associated with wait time. Non-linear models may be needed to capture acuity–wait relationships.

---

### 6.6 Mean Wait Time by Triage Immediacy *(outcome vs. predictor)*

**Chart type:** Bar chart  
**File:** `wait_time_by_triage_bar.png`

Mean wait time by triage immediacy

Mean wait times vary across triage categories but do not follow a strict monotonic “sickest first” ordering in every category — an early signal that **medical need alone may not fully determine queue position**, motivating multivariable modeling.

---

### 6.7 Mean Wait Time by Payment Type *(outcome vs. predictor)*

**Chart type:** Horizontal bar chart  
**File:** `wait_time_by_payment_bar.png`

Mean wait time by payment type

Mean wait times differ across payer categories (e.g., private insurance, Medicaid, self-pay, workers’ compensation). This suggests **access and insurance status** may influence wait times independent of triage.

---

### 6.8 Admission Rate by Triage Immediacy

**Chart type:** Bar chart  
**File:** `admit_rate_by_immediacy.png`

Admission rate by triage immediacy

Admission rates rise sharply from nonurgent to immediate triage levels, validating `triage_immediacy` as a strong clinical-need proxy for downstream disposition.

---

### 6.9 Admission Rate by Arrival Mode

**Chart type:** Bar chart  
**File:** `admit_rate_by_arrivalmode.png`

Admission rate by arrival mode

Ambulance arrivals are admitted at higher rates than walk-in or other modes, reflecting pre-hospital triage and higher baseline acuity among EMS patients.

---

### 6.10 Admission Rate by Insurance / Payment Type

**Chart type:** Bar chart  
**File:** `admit_rate_by_insurance_status.png`

Admission rate by payment type

Admission rates vary by expected payer, indicating that **access factors** co-vary with clinical outcomes and should be included when testing whether wait times track medical need.

---

### 6.11 Admission Rate by Sex

**Chart type:** Bar chart  
**File:** `admit_rate_by_gender.png`

Admission rate by sex

Male and female patients show modestly different admission rates, supporting inclusion of `sex` as a demographic control in wait-time models.

---

### 6.12 Admission Rate by Race

**Chart type:** Bar chart  
**File:** `admit_rate_by_race.png`

Admission rate by race

Admission rates differ across racial groups, highlighting potential equity dimensions that may confound the relationship between wait times and clinical acuity.

---

### 6.13 Admission Rate by Ethnicity

**Chart type:** Bar chart  
**File:** `admit_rate_by_ethnicity.png`

Admission rate by ethnicity

Hispanic vs. non-Hispanic patients show distinct admission patterns, reinforcing the need for ethnicity as a covariate when assessing whether waits reflect medical need or social/access factors.

---

### 6.14 ED Wait Time Trends by Survey Year

**Chart type:** Line graph (mean and median)  
**File:** `wait_time_yearly_trend.png`

ED wait time trends by year

Both mean and median wait times trend upward over the study period, with mean consistently above median. Temporal trends underscore the importance of `survey_year` as a control when comparing acuity and access effects across decades.

---

## 7. Key Conclusions

Tying findings back to: *Are ED wait times determined by medical need?*

1. **The outcome is usable but incomplete.** Roughly two-thirds of visits (510,128) have a valid `wait_time_min`; one-third are missing due to survey-year gaps. The distribution is right-skewed (median 25 min, mean 44.2 min), so models should account for skew and extreme waits.
2. **Clinical-need proxies are present but imperfectly aligned with wait times.** Triage immediacy and vitals are available for most visits and strongly predict admission, yet mean wait times by triage category do not show a clean “higher acuity → shorter wait” pattern in the exploratory bar chart. Medical need appears **necessary but not sufficient** to explain waits.
3. **Access and demographic factors show independent variation.** Payment type, arrival mode, race, and ethnicity all display distinct wait-time and admission patterns, suggesting **non-clinical factors** may influence ED throughput.
4. **Missing data and outliers require deliberate handling, not deletion.** One in five cells is missing overall; pain and disposition are especially sparse. Numeric outliers (e.g., waits up to 1,440 min) were **retained** as real extremes. Imputation, year-specific subsets, and robust modeling will be part of the next phase.
5. **Multivariable machine learning is warranted.** Weak linear correlations between wait time and individual clinical numerics, combined with stronger access-type effects, support building models with all **19 planned covariates** to test whether medical-need features explain wait-time variance after controlling for demographics, access, and survey year.

---




|     |     |
| --- | --- |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |


---

