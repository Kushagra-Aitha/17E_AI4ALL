# ED Wait-Time Modeling Guide

This document describes four models built on the preprocessed CSVs produced by
`scripts/preprocess.py`: three regression models predicting `log_wait_time` directly
(Linear Regression, Lasso Regression, Random Forest), and one classification model
(Multinomial Logistic Regression) predicting a derived Short/Medium/Long wait-time
category. Each model is implemented in its own notebook under `notebooks/`.

## Models at a glance

| # | Model | Notebook | Why it was built |
|---|---|---|---|
| 1 | Linear Regression (OLS) | `notebooks/linear_regression.ipynb` | Interpretable baseline — coefficients on `log_wait_time` read as % effects, with HC3-robust inference for the core equity question |
| 2 | Lasso Regression | `notebooks/lasso_regression.ipynb` | Automatic variable selection/screening across the many sparse one-hot dummy columns the encoding produces |
| 3 | Random Forest | `notebooks/random_forest.ipynb` | Benchmark for non-linear relationships and interactions that OLS's linear form can't represent |
| 4 | Multinomial Logistic Regression | `notebooks/wait_time_classification.ipynb` | Reframes the question as classification — can the same predictors flag a *Long*-wait visit directly, not just explain average minutes? |

---

## Files you need

| File | Use |
|---|---|
| `data/processed/nhamcs_model1_encoded.csv` | Model 1 feature matrix — demographics + access (41 features + 2 targets) |
| `data/processed/nhamcs_model2_encoded.csv` | Model 2 feature matrix — demographics + access + clinical (50 features + 2 targets) |
| `data/processed/preprocessing_report.json` | Imputed medians and feature lists for reference |

Both CSVs contain two target columns: `wait_time_min` (raw) and `log_wait_time` (log-transformed).
All other columns are features. Do not include either target as a predictor.

---

## Step 1: Load and split

```python
import pandas as pd
from sklearn.model_selection import train_test_split

m1 = pd.read_csv("data/processed/nhamcs_model1_encoded.csv")
m2 = pd.read_csv("data/processed/nhamcs_model2_encoded.csv")

TARGET = "log_wait_time"   # use this for all regression models
DROP   = ["wait_time_min", "log_wait_time"]

X1 = m1.drop(columns=DROP)
X2 = m2.drop(columns=DROP)
y  = m1[TARGET]            # same rows in both files, so y is identical

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
X2_train, X2_test, _,       _      = train_test_split(X2, y, test_size=0.2, random_state=42)
```

**Why `log_wait_time` and not `wait_time_min`?**
The raw target is heavily right-skewed (median 23 min, mean 42 min, max 480 min after clamping).
OLS on the raw scale will produce biased coefficients and poor residual diagnostics. The
log-transformed target is approximately normal, and coefficients are interpretable as percentage
effects: a coefficient of 0.10 on a dummy variable means that group waits ~10% longer.

---

## Step 2: Scale numeric features

Apply `StandardScaler` **inside** the training split only — never fit the scaler on the full
dataset before splitting, as that leaks test-set information.

```python
from sklearn.preprocessing import StandardScaler

NUMERIC = ["age_years", "survey_year"]          # Model 1
NUMERIC_M2 = ["age_years", "survey_year", "pulse", "systolic_bp"]  # Model 2

scaler1 = StandardScaler()
X1_train[NUMERIC]    = scaler1.fit_transform(X1_train[NUMERIC])
X1_test[NUMERIC]     = scaler1.transform(X1_test[NUMERIC])

scaler2 = StandardScaler()
X2_train[NUMERIC_M2] = scaler2.fit_transform(X2_train[NUMERIC_M2])
X2_test[NUMERIC_M2]  = scaler2.transform(X2_test[NUMERIC_M2])
```

Binary dummy columns (0/1) do not need scaling — the scaler will compress them toward zero
and make coefficients harder to interpret.

---

## Step 3: Linear Regression (OLS)

Run on both Model 1 and Model 2 and compare the race/ethnicity/payment coefficients.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

def evaluate(model, X_test, y_test, label):
    preds = model.predict(X_test)
    r2  = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    # Convert MAE back to minutes for interpretability
    mae_min = mean_absolute_error(np.expm1(y_test), np.expm1(preds))
    print(f"{label}: R²={r2:.4f}  MAE(log)={mae:.4f}  MAE(min)={mae_min:.1f}")
    return preds

lr1 = LinearRegression().fit(X1_train, y_train)
lr2 = LinearRegression().fit(X2_train, y_train)

evaluate(lr1, X1_test, y_test, "Linear Regression — Model 1")
evaluate(lr2, X2_test, y_test, "Linear Regression — Model 2")
```

### Reading the equity comparison

```python
import pandas as pd

coef1 = pd.Series(lr1.coef_, index=X1_train.columns)
coef2 = pd.Series(lr2.coef_, index=X2_train.columns)

# Columns present in both models (demographic + access features)
shared = coef1.index.intersection(coef2.index)
comparison = pd.DataFrame({
    "Model 1 (total effect)":    coef1[shared],
    "Model 2 (residual effect)": coef2[shared],
})
comparison["change"] = comparison["Model 2 (residual effect)"] - comparison["Model 1 (total effect)"]
print(comparison.sort_values("Model 1 (total effect)").to_string())
```

A large negative `change` for a race/payment dummy means that adding clinical controls
explains away some of that group's wait-time disadvantage. A `change` near zero means
clinical need does not account for the disparity.

### Use robust standard errors for significance testing

sklearn's `LinearRegression` does not compute p-values. Use `statsmodels` for inference:

```python
import statsmodels.api as sm

X1_train_sm = sm.add_constant(X1_train)
ols1 = sm.OLS(y_train, X1_train_sm).fit(cov_type="HC3")  # HC3 = heteroskedasticity-robust
print(ols1.summary())

X2_train_sm = sm.add_constant(X2_train)
ols2 = sm.OLS(y_train, X2_train_sm).fit(cov_type="HC3")
print(ols2.summary())
```

`HC3` robust standard errors correct for heteroskedasticity, which is almost certain in
this data (longer waits tend to have higher variance).

### Results (`notebooks/linear_regression.ipynb`)

Model 1 R²=0.074 (MAE 31.5 min), Model 2 R²=0.078 (MAE 31.4 min) — adding clinical
controls barely improves fit. The central finding: race, ethnicity, and payment-type
coefficients (e.g. `race_Black/African American Only` ≈ +0.23) barely shrink from
Model 1 to Model 2, meaning documented clinical need does not explain away these
disparities. `arrival_mode_Ambulance` is the single largest coefficient in either
model (≈ −0.46 to −0.49, both p<0.001).

---

## Step 4: Lasso Regression

Lasso adds an L1 penalty that shrinks irrelevant coefficients to exactly zero, acting as
automatic variable selection. This is especially useful here because one-hot encoding creates
many sparse dummy columns (e.g., rare residence categories).

```python
from sklearn.linear_model import LassoCV

# LassoCV finds the best alpha via cross-validation automatically
lasso1 = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X1_train, y_train)
lasso2 = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X2_train, y_train)

evaluate(lasso1, X1_test, y_test, "Lasso — Model 1")
evaluate(lasso2, X2_test, y_test, "Lasso — Model 2")

print(f"Lasso M1 alpha: {lasso1.alpha_:.5f}")
print(f"Lasso M2 alpha: {lasso2.alpha_:.5f}")

# Features kept (non-zero coefficient)
kept1 = pd.Series(lasso1.coef_, index=X1_train.columns)
kept1 = kept1[kept1 != 0].sort_values(key=abs, ascending=False)
print(f"\nLasso M1 kept {len(kept1)} of {X1_train.shape[1]} features:")
print(kept1.to_string())
```

**Important:** Lasso can zero out some levels of a categorical variable while keeping others.
This means a race category being zeroed out does not necessarily mean race is unimportant —
check the full group together. Use Lasso as a screening step; interpret residual effects
from the OLS output.

### Results (`notebooks/lasso_regression.ipynb`)

Both models select a small `alpha` (0.00028), so performance nearly matches unpenalized
OLS (R² 0.074 / 0.078) — sparsity comes at no real cost in fit. Lasso zeroes out only
5 of 45 features in Model 1 and 4 of 54 in Model 2, and every one of them is a small,
already non-significant category (rare race/residence/payment/arrival-mode levels,
n roughly 1,500–3,000) — Lasso independently agrees with the OLS significance testing
about which features are weak. Every clinical variable offered to Model 2 survives
selection, but demographic and access effects still rank above them by magnitude,
corroborating the OLS equity finding via a second, independent method.

---

## Step 5: Random Forest (for comparison)

Random Forest captures non-linear relationships and interactions that OLS misses. Use it
to benchmark performance and to produce feature importances that complement the linear
regression coefficients.

```python
from sklearn.ensemble import RandomForestRegressor

# Train on a sample first to tune; 329k rows × full forest is slow
rf2 = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=50,
                             random_state=42, n_jobs=-1)
rf2.fit(X2_train, y_train)
evaluate(rf2, X2_test, y_test, "Random Forest — Model 2")

importances = pd.Series(rf2.feature_importances_, index=X2_train.columns)
print(importances.sort_values(ascending=False).head(20).to_string())
```

If Random Forest R² is substantially higher than OLS R², the relationship has important
non-linear structure that linear regression is missing. If they are similar, the linear
model is adequate and its coefficients are more trustworthy for interpretation.

### Results (`notebooks/random_forest.ipynb`)

R² is substantially higher than OLS/Lasso on both feature sets (Model 1: 0.092 vs.
0.074; Model 2: 0.108 vs. 0.078) — real non-linear structure exists, particularly
around `survey_year`'s non-monotonic trend (a sharp dip around 2020) and the
High/Medium/Low triage-acuity effect the EDA flagged as non-monotonic. Impurity-based
feature importance is biased toward continuous features (`age_years`, vitals) over
one-hot dummies; permutation importance corrects this and shows `survey_year` and
`arrival_mode_Ambulance` as the two most trustworthy top predictors either way.
`race_Black/African American Only` and `ethnicity_Hispanic or Latino` retain real,
non-impurity-inflated importance even in a model flexible enough to capture non-linear
clinical effects — a third, independent corroboration of the equity finding.

---

## Step 6: Multinomial Logistic Regression (Wait-Time Classification)

The three models above all predict `log_wait_time` as a continuous outcome. This model
reframes the research question as classification: instead of asking "how many more
minutes does this group wait," it asks "can these same predictors flag a `Long`-wait
visit directly" — a framing that maps more directly onto an operational triage/staffing
decision than an R² on a log-transformed target does.

Implemented in `notebooks/wait_time_classification.ipynb`, using the Model 2
(demographic + access + clinical) feature matrix:

1. **Derive categories from `wait_time_min`** using fixed, clinically interpretable
   cutoffs — Short (<15 min), Medium (15–60 min), Long (>60 min) — rather than
   equal-width bins, since the raw variable is right-skewed and equal-width bins would
   put almost everything in one bin. These cutoffs happen to land close to the
   dataset's own 33rd/75th percentiles.
2. **Check class balance**: ~39% Short, ~40% Medium, ~21% Long — moderately balanced,
   though `Long` is a minority class worth watching once the model is fit.
3. **Stratified 80/20 train/test split** (`stratify=y`) to preserve those class
   proportions in both splits, given `Long`'s minority status.
4. **Scale numeric features** the same way as the OLS/Lasso notebooks — `StandardScaler`
   fit on the training split only.
5. **Fit a baseline multinomial `LogisticRegression`** (`class_weight=None`).
6. **Refit with `class_weight="balanced"`** to test whether reweighting the loss
   recovers minority-class performance.

### Results

The unweighted baseline reaches 47.0% accuracy (vs. a 40% no-information rate) but
`Long` recall is just 0.02 — it hides the minority class inside `Short`/`Medium`
predictions almost entirely, which balanced accuracy (0.40) exposes but raw accuracy
does not. The `class_weight="balanced"` refit trades overall accuracy for minority-class
recall rather than improving for free: accuracy falls to 42.3% and `Medium` recall drops
(0.62 → 0.25), but `Long` recall rises to 0.51 and balanced accuracy improves to 0.44.
Neither model is strong in an absolute sense (`Long` precision stays 0.30–0.47 either
way), but the balanced version is the more defensible choice for this research
question — silently predicting almost no `Long` waits would understate exactly the
long-wait visits the equity analysis cares about most.

---

## Step 7: Evaluate and report

### Metrics to report

| Metric | Why |
|---|---|
| R² on test set | Fraction of variance explained; expected to be low (0.05–0.20) for this noisy outcome |
| MAE in minutes (back-transformed) | `mean_absolute_error(expm1(y_test), expm1(preds))` — actionable and interpretable |
| Coefficients + 95% CI on equity variables | The central research finding |

### Expected R² range

Do not be discouraged by low R². ED wait times are driven partly by hospital-level factors
(how many beds, how many staff on shift) that NHAMCS does not record at the visit level.
Individual-level predictors explaining 10–20% of variance is consistent with the published
literature on this outcome. The research value is in the coefficient patterns, not prediction accuracy.

### Answering the research question

Actual values from `notebooks/linear_regression.ipynb` (HC3-robust OLS, `***`=p<0.001):

| Variable | Model 1 coef | Model 2 coef | Interpretation |
|---|---|---|---|
| `race_Black/African American Only` | +0.229*** | +0.231*** | Persists: clinical controls do not explain the disparity |
| `ethnicity_Hispanic or Latino` | +0.108*** | +0.108*** | Persists, essentially unchanged |
| `payment_type_Medicaid/CHIP` | +0.068*** | +0.068*** | Persists after controlling for triage and vitals |
| `payment_type_No charge/Charity` | +0.161*** | +0.168*** | Largest payment-type effect; persists |
| `triage_acuity_High` | — | −0.234*** | Clinical: high-acuity patients seen faster |
| `arrival_mode_Ambulance` | −0.485*** | −0.462*** | Largest coefficient in either model |

---

## Step 8: Robustness checks

1. **Run separately by era** — fit the same models on 2007–2013 and 2014–2022 (pre/post ACA)
   to test whether insurance expansion changed the equity pattern.

2. **Check residuals** — plot residuals vs fitted values and a Q-Q plot. Fan-shaped residuals
   confirm heteroskedasticity; HC3 standard errors address it but noting the pattern is good practice.

3. **Pain-scale sub-analysis** — filter to rows where `pain_scale` is non-null (roughly 2011+,
   ~30% of the modeling rows) and add `pain_scale` as a feature. If adding patient-reported
   pain further reduces the race/payment coefficients, it means some of the residual disparity
   in Model 2 is carried through pain assessment differences.

---

## What was actually built

All four models above ended up as standalone, executed notebooks (rather than the
`scripts/train_*.py` layout originally sketched for this project) so that EDA,
findings, and diagnostic plots stay next to the code that produced them:

```
scripts/
├── dataset_exploration.py    (done)
└── preprocess.py             (done)

notebooks/
├── linear_regression.ipynb          Model 1 + Model 2 OLS, HC3 inference, equity comparison
├── lasso_regression.ipynb           Model 1 + Model 2 LassoCV, feature selection
├── random_forest.ipynb              Model 1 + Model 2 RF, permutation importance, partial dependence
└── wait_time_classification.ipynb   Short/Medium/Long multinomial logistic regression
```

---

## Model comparison for deployment

There's no single answer — it depends on what the deployment is for:

- **Operational wait-time predictor** (surfacing an expected wait to patients/staff):
  **Random Forest (Model 2)** is the strongest choice. It has the best accuracy of
  the three regression models (R²=0.108 vs. 0.078 for OLS/Lasso, lowest MAE) and is
  the only one that captures structure the linear models miss — the COVID-era 2020
  dip in `survey_year` and the non-monotonic triage-acuity effect (both confirmed via
  the partial dependence plots and permutation importance in Step 5).
- **Decision-support / policy-facing tool** (anything where the prediction needs to
  be justified — to a hospital board, a regulator, or in response to a
  disparate-impact question): **Linear Regression (OLS)** is the safer choice despite
  its lower R². Its coefficients + HC3 confidence intervals are directly auditable in
  a way Random Forest's permutation importances are not, which matters specifically
  because the core finding here is a demographic/access disparity — an opaque model
  making wait-time-relevant predictions in that context is itself a harder thing to
  defend.
- **`wait_time_classification.ipynb` (Short/Medium/Long) is not recommended for
  deployment as-is**, regardless of context: even after `class_weight="balanced"`,
  `Long`-wait recall is only 0.51 with precision in the 0.30–0.47 range — too
  unreliable to act on directly. It's useful as an analysis lens (Step 6), not as a
  production classifier.

**Bottom line:** Random Forest for pure predictive accuracy with no explainability
requirement; OLS if the deployment needs to explain *why* — which, given this
project's equity research question, is the more likely real-world requirement.
