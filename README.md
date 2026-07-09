# Yale EMMLC Admission-Risk Analysis

This folder contains the Yale EMMLC admission-risk analysis. The model predicts admission versus discharge using information available at arrival or initial triage.

- `yale_clean_triage.csv` is the cleaned triage-time dataset.
- `yale_logistic_regression.ipynb` trains and evaluates Logistic Regression using `disposition_admit` as the target, where `1` is admit and `0` is discharge.
- `yale_logistic_regression.py` is kept as a backup script version of the same analysis.
- `yale_logistic_results.txt` contains the earlier script output with test metrics and coefficient summaries.

The dataset has 560,484 rows and 220 columns. There are 219 predictors plus the target column, `disposition_admit`. The target contains 393,846 discharge visits (70.27%) and 166,638 admission visits (29.73%).

## How to Run

Install the needed packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open `yale_logistic_regression.ipynb` in VS Code or Jupyter, select the `.venv` Python kernel, and run all cells. The notebook expects `yale_clean_triage.csv` to be in the same folder.

## Model Setup

The target variable is `disposition_admit`, where `1` represents admission and `0` represents discharge. The data is split into an 80/20 stratified train-test split so that the admit/discharge ratio is preserved in both sets.

The pipeline handles missing values before training. Numeric features are imputed with the median and scaled. Categorical features are imputed with the most frequent value and one-hot encoded. Chief complaint indicator columns are also imputed before being passed into the model.

The model used is Logistic Regression with balanced class weights to account for the class imbalance between discharge and admission visits.

## Train/Test Metrics

The model is evaluated on both the training and test splits using accuracy, recall, and precision.

| split | accuracy | recall | precision |
|---|---:|---:|---:|
| train | 0.787188 | 0.803931 | 0.607358 |
| test | 0.787211 | 0.801398 | 0.607810 |

Accuracy is about 0.787 on both train and test, meaning the model correctly predicts admit vs discharge for about 79% of visits. Because the train and test accuracy are almost identical, the model does not show a major train/test performance gap.

Recall is about 0.804 on train and 0.801 on test. This means the model catches about 80% of actual admitted patients, so it is strong at identifying admission cases.

Precision is about 0.607 on both train and test. This means that when the model predicts admission, around 61% of those predictions are correct. The remaining predicted admissions are false positives, meaning some discharged patients are predicted as admitted.

Overall, the train and test scores are very similar, so the model generalizes consistently across the split. The main pattern is high recall with moderate precision: it catches many admitted patients, but it also overpredicts admission for some discharged patients.
