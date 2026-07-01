# Regression Implementation Guide

This document describes how to implement the three planned models (Linear Regression,
Lasso Regression, Random Forest) using the preprocessed CSVs produced by `scripts/preprocess.py`.

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

---

## Step 6: Evaluate and report

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

Report a table like this (made from the OLS outputs):

| Variable | Model 1 coef | Model 2 coef | Interpretation |
|---|---|---|---|
| `race_Black/African American Only` | +0.08** | +0.05* | Partial: clinical need explains ~38% of the disparity |
| `payment_type_Medicaid/CHIP` | +0.12** | +0.09** | Persists after controlling for triage and vitals |
| `triage_acuity_High` | — | −0.14** | Clinical: High-acuity patients seen faster |

(Actual values will come from your fitted models.)

---

## Step 7: Robustness checks

1. **Run separately by era** — fit the same models on 2007–2013 and 2014–2022 (pre/post ACA)
   to test whether insurance expansion changed the equity pattern.

2. **Check residuals** — plot residuals vs fitted values and a Q-Q plot. Fan-shaped residuals
   confirm heteroskedasticity; HC3 standard errors address it but noting the pattern is good practice.

3. **Pain-scale sub-analysis** — filter to rows where `pain_scale` is non-null (roughly 2011+,
   ~30% of the modeling rows) and add `pain_scale` as a feature. If adding patient-reported
   pain further reduces the race/payment coefficients, it means some of the residual disparity
   in Model 2 is carried through pain assessment differences.

---

## Suggested script layout

```
scripts/
├── dataset_exploration.py    (done)
├── preprocess.py             (done)
├── train_linear.py           OLS + Lasso on Model 1 and Model 2
├── train_random_forest.py    RF on Model 2 for benchmarking
└── evaluate_equity.py        coefficient comparison table + residual plots
```
