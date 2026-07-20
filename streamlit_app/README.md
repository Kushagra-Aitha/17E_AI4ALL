# Streamlit Integration App

This app combines the NHAMCS wait-time models with the Yale admission-risk model in four tabs: Home, Wait-Time Prediction, Admission Risk, and Monitoring.

## Run Locally

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

The app expects all deployment artifacts in `models/`. NHAMCS predictions are stored as `log(wait_time_min + 1)` and are converted back to minutes with `numpy.expm1` before display.

## Compatibility Note

The environment pins `scikit-learn==1.6.1`, which is the locally verified version and matches the Yale artifact. The NHAMCS artifacts were exported with scikit-learn 1.9.0. They load and predict under 1.6.1 locally but emit an `InconsistentVersionWarning`. Re-export the NHAMCS artifacts with 1.6.1 before production deployment to remove this serialization risk.

This is a research demonstration and is not intended for clinical decision-making.
