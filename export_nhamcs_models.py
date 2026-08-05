"""Re-export the NHAMCS wait-time model bundles under the deployment scikit-learn version.

models/nhamcs_model1.joblib and nhamcs_model2.joblib were pickled with
scikit-learn 1.9.0, but the app is pinned to scikit-learn==1.6.1 (requirements.txt,
matching the Streamlit Cloud runtime). Loading them raises
``InconsistentVersionWarning`` and is unsupported by scikit-learn's own
compatibility guarantees. This script refits both models from the same encoded
data, split, and random_state used originally, so the fitted coefficients and
test metrics are identical, then re-saves each bundle under the scikit-learn
version installed in this environment.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_CONFIGS = [
    {
        "bundle_path": MODELS_DIR / "nhamcs_model1.joblib",
        "csv_path": DATA_DIR / "nhamcs_model1_encoded.csv",
        "rename": {},
    },
    {
        "bundle_path": MODELS_DIR / "nhamcs_model2.joblib",
        "csv_path": DATA_DIR / "nhamcs_model2_encoded.csv",
        # A later feature-engineering pass renamed this missing-value indicator
        # after this encoded CSV snapshot was written; the trained bundle's
        # feature list uses the newer name, so line it up before refitting.
        "rename": {"seen_72h_Unknown": "seen_72h_missing"},
    },
]


def refit_bundle(bundle_path: Path, csv_path: Path, rename: dict[str, str]) -> None:
    bundle = joblib.load(bundle_path)
    df = pd.read_csv(csv_path).rename(columns=rename)

    features = bundle["feature_columns"]
    numeric_columns = bundle["numeric_columns"]
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"{bundle_path.name}: encoded CSV is missing columns {missing}")

    X = df[features]
    y = df["log_wait_time"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=bundle["test_size"], random_state=bundle["random_state"]
    )

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    pred_log = model.predict(X_test_scaled)
    pred_minutes = np.maximum(0.0, np.expm1(pred_log))
    true_minutes = df.loc[y_test.index, "wait_time_min"]

    metrics_test = {
        "r2": round(float(r2_score(y_test, pred_log)), 6),
        "mae_log": round(float(mean_absolute_error(y_test, pred_log)), 6),
        "rmse_log": round(float(mean_squared_error(y_test, pred_log) ** 0.5), 6),
        "mae_minutes": round(float(mean_absolute_error(true_minutes, pred_minutes)), 2),
        "rmse_minutes": round(float(mean_squared_error(true_minutes, pred_minutes) ** 0.5), 2),
    }

    old_metrics = bundle["metrics_test"]
    changed = {
        key: (old_metrics[key], metrics_test[key])
        for key in metrics_test
        if abs(metrics_test[key] - old_metrics[key]) > 1e-4
    }
    if changed:
        raise ValueError(
            f"{bundle_path.name}: refit metrics don't match the original bundle: {changed}"
        )

    bundle["model"] = model
    bundle["scaler"] = scaler
    bundle["metrics_test"] = metrics_test
    bundle["sklearn_version"] = sklearn.__version__

    joblib.dump(bundle, bundle_path)
    print(f"Re-exported {bundle_path.name} under scikit-learn {sklearn.__version__}")
    print(f"  test metrics (unchanged): {metrics_test}")


def main() -> None:
    for config in MODEL_CONFIGS:
        refit_bundle(config["bundle_path"], config["csv_path"], config["rename"])


if __name__ == "__main__":
    main()
