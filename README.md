# Yale EMMLC Admission-Risk Analysis

This folder contains the Yale EMMLC supporting admission-risk analysis. The model predicts admission versus discharge from information available at arrival or initial triage. 

- `yale_clean_triage.csv` contains 560,484 visits, 219 predictors, and the binary target `disposition_admit`.
- `yale_logistic_regression.py` runs Logistic Regression using `disposition_admit` as the target, where `1` is admit and `0` is discharge.
- `yale_logistic_results.txt` contains the model metrics and coefficient summaries.

The complete dataset has 560,484 rows and 220 columns. The target contains 393,846 discharge visits (70.27%) and 166,638 admission visits (29.73%).

## Main Results

| Metric | Value |
|---|---:|
| Accuracy | 0.787202 |
| Precision | 0.607796 |
| Recall | 0.801398 |
| F1-score | 0.691298 |
| ROC-AUC | 0.875407 |

Confusion matrix, shown as `[[TN, FP], [FN, TP]]`:

```text
[[61534, 17235],
 [ 6619, 26709]]
```

## What We Learn From the Model Numbers

The ROC-AUC is 0.875407, showing that the model separates admitted and discharged patients well overall.

The recall is 0.801398, meaning the model identifies about 80% of actual admitted patients.

The precision is 0.607796, meaning about 61% of predicted admissions are actual admissions.

The confusion matrix gives the exact counts: 26,709 admitted patients were correctly predicted, 6,619 admitted patients were missed, and 17,235 discharged patients were incorrectly predicted as admitted.

These results show strong overall separation and high admission recall, with moderate precision.
