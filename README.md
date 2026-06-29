# Yale EMMLC Admission-Risk Analysis

This folder contains the Yale EMMLC admission-risk analysis.

- `yale_logistic_regression.py` runs Logistic Regression on the cleaned Yale triage CSV using `disposition_admit` as the target.
- `yale_logistic_results.txt` contains the model metrics and coefficient summaries.
- CSV files are ignored and are not committed.

## Main Results

| Metric | Value |
|---|---:|
| Accuracy | 0.7872 |
| Precision | 0.6078 |
| Recall | 0.8014 |
| F1-score | 0.6913 |
| ROC-AUC | 0.8754 |

Admission is interpreted as admission risk/admission disposition, not as pure clinical severity. Coefficients represent associations, not causal effects.
