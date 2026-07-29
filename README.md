# Drivers of Emergency Department Wait Times

**Research question:** To what extent are emergency department wait times explained by patient
symptoms and clinical urgency, and do demographic factors remain associated with wait times
after accounting for medical need?

**Team:** Kushagra Aitha, Rupsa Bose, Miskatul Moon, Shahriar Shabib, Sofi Le

---

## Project overview

This project analyzes 31 years of the National Hospital Ambulatory Medical Care Survey (NHAMCS)
Emergency Department public-use files (1992–2022) to identify what drives ED wait times. The
core question is whether longer waits reflect clinical need (triage urgency, vitals, symptoms)
or whether demographic and access factors — race, insurance type, arrival mode — remain
associated with wait time after controlling for medical acuity.

---

## Repository structure

```
├── scripts/
│   ├── dataset_exploration.py      EDA
│   ├── preprocess.py               cleaning / encoding
│   ├── train_linear.py             OLS equity models
│   └── export_deploy_artifacts.py  Streamlit linear joblibs
├── data/processed/                 model-ready CSVs (gitignored)
├── notebooks/                      linear, lasso, RF, classification
├── outputs/
│   ├── visualizations/             EDA + classification plots
│   └── linear_regression/          OLS metrics, equity tables, residual plots
├── models/                         Streamlit deploy artifacts (.joblib, features, samples)
├── deliverables/                   EDA reports
└── MODELING.md                     modeling guide
```

Local-only (gitignored): `_archive/random_forest/` — RF script, outputs, and joblibs kept in case we need them later.

---

## Dataset

| Item | Value |
|---|---|
| Source | NHAMCS ED public-use files (CDC/NCHS) |
| Raw files | 31 annual CSVs, 1992–2022 |
| Raw rows (stacked) | 839,772 |
| Modeling subset | 2007–2022 (16 years, consistent wait-time field) |
| Final modeling rows | 329,249 |
| Target variable | `wait_time_min` — minutes from ED arrival to first provider contact |

### Known data limitations

- **Selection bias:** patients who left before being seen (eloped) are absent; their wait would have been longer than recorded.
- **Underrepresentation:** federal hospitals (VA, military, IHS) are excluded — veterans and many tribal patients are not in the sample.
- **Measurement shift:** NHAMCS changed its wait-time definition and EHR timestamping across years; `survey_year` is included in all models as a temporal control.
- **Survey weights:** NHAMCS uses complex sampling with unequal selection probabilities. Models treat each row equally; results describe the sample, not the nationally-weighted population. This is noted as a limitation.

---

## What was done

### 1. Exploratory data analysis (`scripts/dataset_exploration.py`)

Combines all 31 annual files into a single dataset with 20 standardized columns. Generates:

- Missing-value table (overall 21.4% missing across key columns)
- Outlier summary using IQR method on all numeric columns
- 14 visualizations covering the target distribution, triage effects, payment-type effects,
  admission rates by race/ethnicity/sex, and wait-time trends by year

**Key EDA findings:**

- `wait_time_min` is right-skewed (median 25 min, mean 44 min, max 1440 min); requires log
  transformation for linear regression.
- Triage immediacy has a partial but non-monotonic relationship with wait time: Immediate
  patients wait ~22 min on average, but Urgent, Semi-urgent, and Nonurgent categories all
  cluster around 37–38 min. Clinical urgency alone does not explain queue position.
- Payment type shows a wider spread (30–49 min) than triage category. Medicaid/SCHIP patients
  wait the longest; Medicare patients wait the shortest.
- Wait times rose from ~35 min mean (1997) to a peak of ~48 min (2008–2009), then declined
  steadily to ~30 min (2022). `survey_year` is a mandatory control.
- Numeric correlations with wait time are weak (all ≤ 0.27); non-linear models may fit better,
  but linear regression is the right starting point for interpretability.

### 2. Data cleaning and preprocessing (`scripts/preprocess.py`)

Filters to 2007–2022 and produces three model-ready CSV files. Key steps:

| Step | Detail |
|---|---|
| Year filter | 2007–2022 only — consistent WAITTIME field and modern triage coding |
| Deduplication | 163 exact duplicate rows removed |
| Wait-time filter | Rows without a recorded wait time dropped (33% of all rows) |
| Wait-time clamp | Values > 480 min (8 hrs) excluded from encoded model files; kept in cleaned CSV |
| Zero vitals | `pulse == 0` and `systolic_bp == 0` treated as missing (recording artifacts) |
| Log transform | `log_wait_time = log(wait_time_min + 1)` added as regression target |
| Triage harmonization | Multi-era labels collapsed to 3-level acuity: High / Medium / Low |
| Arrival mode harmonization | Binary Yes/No ambulance coding from early years → Ambulance / Walk-in/Other |
| Payment consolidation | Medicaid / Medicaid or CHIP / SCHIP → `Medicaid/CHIP` |
| Missing indicators | `pulse_missing`, `systolic_bp_missing`, `seen_72h_missing` added before imputation |
| Imputation | Pulse (median = 88) and systolic BP (median = 131) filled with training-set median |
| One-hot encoding | All categoricals encoded; reference categories dropped (see below) |

**Reference categories dropped (regression baselines):**

| Variable | Reference (dropped) |
|---|---|
| `sex` | Female |
| `race` | White Only |
| `ethnicity` | Not Hispanic or Latino |
| `payment_type` | Private insurance |
| `arrival_mode` | Walk-in/Other |
| `residence` | Private residence |
| `triage_acuity` | Medium |
| `visit_day_of_week` | Wednesday |
| `visit_month` | July |

**Columns deliberately excluded from model features:**

| Column | Reason |
|---|---|
| `length_of_visit_min` | Post-visit outcome — not known at triage time (data leakage) |
| `admit_hospital` | Post-visit outcome |
| `admit_observation` | Post-visit outcome |
| `disposition` | 75% missing and post-visit |
| `pain_scale` | 71% missing; only available 2011+; use in a dedicated sub-analysis |

### 3. Dual-model design for the equity research question

Two encoded datasets are produced to answer whether demographic disparities persist after
controlling for clinical need:

- **Model 1** (`nhamcs_model1_encoded.csv`) — demographic + access features only.
  Coefficient on race/ethnicity/payment here measures the *total* association with wait time.
- **Model 2** (`nhamcs_model2_encoded.csv`) — adds clinical features (triage acuity, vitals,
  prior visit flag).
  Coefficient on race/ethnicity/payment here measures the *residual* association after
  controlling for medical need.

If the race/payment coefficients shrink substantially from Model 1 to Model 2, clinical need
explains the disparity. If they remain significant, non-clinical factors are independently
associated with wait time.

---

## Results so far

All models predict `log_wait_time` on ~329k visits (2007–2022). Model 1 = demographics + access;
Model 2 = Model 1 + clinical features (triage, vitals, prior visit).

### Linear regression (main model — equity / interpretation)

| Model | R² (test) | MAE | RMSE |
|---|---:|---:|---:|
| Model 1 (demographics + access) | 0.074 | 31.5 min | 57.6 min |
| Model 2 (+ clinical) | 0.078 | 31.4 min | 57.5 min |

| Finding | Effect | After clinical controls? |
|---|---|---|
| Black / African American vs White | **+26%** longer wait | Essentially unchanged (~0% explained by clinical need) |
| Hispanic / Latino vs non-Hispanic | **+11%** longer wait | Essentially unchanged |
| Medicaid / CHIP vs private | **+7%** longer wait | Essentially unchanged |
| Ambulance vs walk-in | **−46%** shorter wait | Strongest single predictor |
| High triage acuity vs medium | **−23%** shorter wait | “Sickest first” partially enforced |

**Research-question answer:** Demographic disparities in ED wait times **persist almost entirely** after controlling for clinical urgency. Race, ethnicity, and insurance remain associated with longer waits independent of triage and vitals.

Deploy artifacts for Streamlit: `models/nhamcs_model1.joblib`, `models/nhamcs_model2.joblib` (plus feature lists, sample inputs, metrics). More detail: `outputs/linear_regression/README.md`.

### Random Forest (tried as a non-linear benchmark)

We also tried Random Forest (`n_estimators=200`, `max_depth=12`, `min_samples_leaf=50`) to see whether wait times have non-linear structure that OLS misses. **Not kept as the main project model** — linear is better for the equity write-up — but results are recorded here. Script/outputs/joblibs live locally in `_archive/random_forest/` (gitignored).

| Model | R² (test) | MAE | RMSE |
|---|---:|---:|---:|
| RF Model 1 | 0.088 | 31.3 min | 57.3 min |
| RF Model 2 | 0.098 | 31.2 min | 57.2 min |

- RF beat linear by ~+2 pp R² on Model 2 → some real non-linear structure exists.
- Overall fit still modest (~10%); hospital-level factors aren’t in NHAMCS.
- Top predictors (permutation): `survey_year`, ambulance arrival, Black/AA race, high triage, Hispanic/Latino ethnicity — race/ethnicity still mattered after clinical features.

---

## How to reproduce

```bash
# EDA
python scripts/dataset_exploration.py

# Cleaning / encoding
python scripts/preprocess.py

# Linear regression (equity models)
python scripts/train_linear.py

# Streamlit linear joblibs / sample inputs / metrics
python scripts/export_deploy_artifacts.py
```

Dependencies are in `requirements.txt` (`pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `joblib`, …).

---

## Also in the repo

- Notebooks: linear, lasso, random forest, wait-time classification (`notebooks/`)
- Full modeling guide: [MODELING.md](MODELING.md)
- EDA deliverables: `deliverables/`
