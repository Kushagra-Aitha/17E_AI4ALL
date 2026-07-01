#!/usr/bin/env python3
"""
Linear Regression training script — ED Wait Times project.

Trains OLS linear regression on Model 1 (demographics + access) and
Model 2 (+ clinical features), then compares coefficients to answer the
equity research question:

  Do demographic disparities in wait time persist after controlling for
  clinical need (triage, vitals)?

Outputs saved to outputs/linear_regression/:
  metrics.json                    R², MAE, RMSE for both models
  model1_coefficients.csv         OLS coefficients + p-values (HC3 robust SEs)
  model2_coefficients.csv
  equity_comparison.csv           side-by-side race/payment coefficients M1 vs M2
  residuals_model1.png
  residuals_model2.png
  coefficient_plot_equity.png     bar chart comparing equity variable coefficients
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT        = Path(__file__).resolve().parent.parent
PROCESSED   = ROOT / "data" / "processed"
OUT_DIR     = ROOT / "outputs" / "linear_regression"

TARGET      = "log_wait_time"
DROP_COLS   = ["wait_time_min", "log_wait_time"]
NUMERIC_M1  = ["age_years", "survey_year"]
NUMERIC_M2  = ["age_years", "survey_year", "pulse", "systolic_bp"]
RANDOM_STATE = 42

# Equity variables to highlight in the Model 1 vs Model 2 comparison
EQUITY_PREFIXES = ("race_", "ethnicity_", "payment_type_")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    m1 = pd.read_csv(PROCESSED / "nhamcs_model1_encoded.csv")
    m2 = pd.read_csv(PROCESSED / "nhamcs_model2_encoded.csv")
    y  = m1[TARGET]
    X1 = m1.drop(columns=DROP_COLS)
    X2 = m2.drop(columns=DROP_COLS)
    return X1, X2, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    # Work on copies so the originals are unchanged
    X_train = X_train.copy()
    X_test  = X_test.copy()
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])
    return X_train, X_test, y_train, y_test, scaler


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, label: str) -> dict:
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    # Back-transform to minutes for interpretability
    y_min      = np.expm1(y_true)
    y_pred_min = np.expm1(y_pred)
    mae_min  = mean_absolute_error(y_min, y_pred_min)
    rmse_min = mean_squared_error(y_min, y_pred_min) ** 0.5
    print(
        f"  {label}: R²={r2:.4f}  "
        f"MAE={mae:.4f} ({mae_min:.1f} min)  "
        f"RMSE={rmse:.4f} ({rmse_min:.1f} min)"
    )
    return {
        "r2": round(r2, 6),
        "mae_log": round(mae, 6),
        "rmse_log": round(rmse, 6),
        "mae_minutes": round(mae_min, 2),
        "rmse_minutes": round(rmse_min, 2),
    }


def fit_statsmodels_ols(
    X_train: pd.DataFrame, y_train: pd.Series, label: str
) -> pd.DataFrame:
    """Fit OLS with HC3 robust standard errors; return coefficient table."""
    Xc = sm.add_constant(X_train)
    model = sm.OLS(y_train, Xc).fit(cov_type="HC3")
    print(f"\n  {label} OLS summary (top 20 by |coef|):")
    coef_df = pd.DataFrame({
        "coef":    model.params,
        "se":      model.bse,
        "t":       model.tvalues,
        "p_value": model.pvalues,
        "ci_low":  model.conf_int()[0],
        "ci_high": model.conf_int()[1],
    }).drop(index="const", errors="ignore")
    top20 = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index).head(20)
    print(top20[["coef", "se", "p_value"]].to_string())
    return coef_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true: pd.Series, y_pred: np.ndarray, label: str, path: Path
) -> None:
    residuals = y_true.values - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Residual Diagnostics — {label}", fontsize=13)

    # Residuals vs fitted
    axes[0].scatter(y_pred, residuals, alpha=0.05, s=4, color="#2E86AB")
    axes[0].axhline(0, color="red", linewidth=1)
    axes[0].set_xlabel("Fitted values (log scale)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Fitted")

    # Q-Q plot
    sm.qqplot(residuals, line="s", ax=axes[1], alpha=0.3, markersize=2)
    axes[1].set_title("Normal Q-Q Plot")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_equity_comparison(comparison: pd.DataFrame, path: Path) -> None:
    df = comparison.copy().sort_values("model1_coef")
    x  = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.5)))
    bars1 = ax.barh(x - width / 2, df["model1_coef"], width, label="Model 1 (no clinical controls)", color="#E76F51", alpha=0.85)
    bars2 = ax.barh(x + width / 2, df["model2_coef"], width, label="Model 2 (with clinical controls)", color="#2A9D8F", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(df["variable"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("OLS coefficient (log wait time)")
    ax.set_title("Equity Variable Coefficients: Model 1 vs Model 2\n(reference = White, Private insurance, Not Hispanic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X1, X2, y = load_data()
    print(f"  Model 1: {X1.shape[1]} features, {len(y):,} rows")
    print(f"  Model 2: {X2.shape[1]} features")

    print("\nSplitting and scaling...")
    X1_train, X1_test, y_train, y_test, _ = split_and_scale(X1, y, NUMERIC_M1)
    X2_train, X2_test, _,      _,       _ = split_and_scale(X2, y, NUMERIC_M2)
    print(f"  Train: {len(y_train):,}  Test: {len(y_test):,}")

    # ------------------------------------------------------------------
    # sklearn LinearRegression (fast; used for predictions and metrics)
    # ------------------------------------------------------------------
    print("\nFitting sklearn LinearRegression...")
    lr1 = LinearRegression().fit(X1_train, y_train)
    lr2 = LinearRegression().fit(X2_train, y_train)

    print("\nTest-set performance:")
    all_metrics = {}
    all_metrics["model1_test"] = compute_metrics(y_test, lr1.predict(X1_test), "Model 1")
    all_metrics["model2_test"] = compute_metrics(y_test, lr2.predict(X2_test), "Model 2")

    print("\nTrain-set performance (sanity check for overfitting):")
    all_metrics["model1_train"] = compute_metrics(y_train, lr1.predict(X1_train), "Model 1")
    all_metrics["model2_train"] = compute_metrics(y_train, lr2.predict(X2_train), "Model 2")

    (OUT_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2))

    # ------------------------------------------------------------------
    # statsmodels OLS with HC3 robust standard errors (for inference)
    # ------------------------------------------------------------------
    print("\nFitting statsmodels OLS (HC3 robust SEs)...")
    coef1 = fit_statsmodels_ols(X1_train, y_train, "Model 1")
    coef2 = fit_statsmodels_ols(X2_train, y_train, "Model 2")

    coef1.to_csv(OUT_DIR / "model1_coefficients.csv")
    coef2.to_csv(OUT_DIR / "model2_coefficients.csv")
    print(f"\n  Saved coefficient tables → {OUT_DIR.relative_to(ROOT)}/")

    # ------------------------------------------------------------------
    # Equity comparison table: Model 1 vs Model 2 for race/ethnicity/payment
    # ------------------------------------------------------------------
    equity_vars = [c for c in coef1.index if c.startswith(EQUITY_PREFIXES)]
    equity_df = pd.DataFrame({
        "variable":    equity_vars,
        "model1_coef": coef1.loc[equity_vars, "coef"].values,
        "model1_p":    coef1.loc[equity_vars, "p_value"].values,
        "model2_coef": coef2.loc[equity_vars, "coef"].values,
        "model2_p":    coef2.loc[equity_vars, "p_value"].values,
    })
    equity_df["coef_change"]   = equity_df["model2_coef"] - equity_df["model1_coef"]
    equity_df["pct_explained"] = (
        -equity_df["coef_change"] / equity_df["model1_coef"].abs() * 100
    ).round(1)
    equity_df = equity_df.sort_values("model1_coef", ascending=False)
    equity_df.to_csv(OUT_DIR / "equity_comparison.csv", index=False)

    print("\nEquity variable comparison (Model 1 → Model 2):")
    print(equity_df[["variable", "model1_coef", "model2_coef", "coef_change", "pct_explained"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")
    plot_residuals(y_test, lr1.predict(X1_test), "Model 1", OUT_DIR / "residuals_model1.png")
    plot_residuals(y_test, lr2.predict(X2_test), "Model 2", OUT_DIR / "residuals_model2.png")
    plot_equity_comparison(equity_df, OUT_DIR / "coefficient_plot_equity.png")
    print(f"  Saved plots → {OUT_DIR.relative_to(ROOT)}/")

    print("\n=== Done ===")
    print(f"All outputs in {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
