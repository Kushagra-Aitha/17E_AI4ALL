# Yale EMMLC Admission-Risk Analysis

This folder contains the Yale EMMLC admission-risk analysis. The model predicts admission versus discharge using information available at arrival or initial triage.

- `yale_clean_triage.csv` is the cleaned triage-time dataset.
- `yale_logistic_regression.ipynb` trains and evaluates Logistic Regression using `disposition_admit` as the target, where `1` is admit and `0` is discharge.
- `yale_logistic_regression.py` is kept as a backup script version of the same analysis.
- `yale_logistic_results.txt` contains the earlier script output with test metrics and coefficient summaries.

The dataset has 560,484 rows and 220 columns. There are 219 predictors plus the target column, `disposition_admit`. The target contains 393,846 discharge visits (70.27%) and 166,638 admission visits (29.73%).

## Data Cleaning

The cleaned file `yale_clean_triage.csv` was created from the Yale raw dataset by keeping the same 560,484 visits and reducing the dataset from 968 raw columns to 220 cleaned columns. The cleaned file contains 219 predictor columns plus the newly created target column, `disposition_admit`.

The target column `disposition_admit` was created from the original `disposition` column. `Admit` was mapped to `1` and `Discharge` was mapped to `0`. The final target distribution is 166,638 admitted visits (29.73%) and 393,846 discharged visits (70.27%). After creating `disposition_admit`, the original `disposition` column was removed.

The retained predictors include 4 demographic columns (`age`, `gender`, `race`, `ethnicity`), 1 insurance column (`insurance_status`), 4 arrival/time columns (`arrivalmode`, `arrivalmonth`, `arrivalday`, `arrivalhour_bin`), 1 triage urgency column (`esi`), 7 triage vital-sign columns, 2 prior-utilization columns (`n_edvisits`, `n_admissions`), and 200 existing `cc_*` chief complaint columns.

Columns that could leak later-care information were removed, including diagnosis/comorbidity indicators, lab/result summary columns, imaging counts, medication columns, procedure/surgery columns, discharge information, length of stay, post-triage vitals, and summary columns ending in `_last`, `_min`, `_max`, or `_median`.

No rows were removed during this cleaning step. Missing values were left in the cleaned CSV and are handled later inside the modeling pipeline through imputation.

## How to Run

Install the needed packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open `yale_logistic_regression.ipynb` in VS Code or Jupyter, select the `.venv` Python kernel, and run all cells. The notebook expects `yale_clean_triage.csv` to be in the same folder.

## Class Balance

The Yale target variable is imbalanced. Discharge visits make up 393,846 records (70.27%), while admission visits make up 166,638 records (29.73%). Because discharge is the larger class, accuracy alone is not enough to evaluate the model.

The train-test split uses stratification, so the admission/discharge ratio is preserved in both train and test sets. The Logistic Regression model also uses balanced class weights, which gives more weight to the minority admission class during training.

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

## Full Performance Metrics

| split | accuracy | precision | recall | specificity | f1 |
|---|---:|---:|---:|---:|---:|
| train | 0.787188 | 0.607358 | 0.803931 | 0.780105 | 0.691955 |
| test | 0.787211 | 0.607810 | 0.801398 | 0.781208 | 0.691307 |

These metrics give a fuller evaluation than accuracy alone, especially because the dataset is imbalanced toward discharge. Accuracy shows the overall share of correct admit/discharge predictions. Recall shows how well the model catches actual admitted patients. Precision shows how often predicted admissions are truly admissions. Specificity shows how well the model identifies actual discharged patients. F1-score summarizes the balance between precision and recall.

The train and test values are very close across all five metrics, so the model performs consistently across the split. The main pattern is high recall with moderate precision: the model catches many admitted patients, but it also overpredicts admission for some discharged patients.

## ROC Curve and AUC

The notebook includes a ROC curve for the baseline Logistic Regression model using predicted admission probabilities on the test set. The baseline ROC-AUC is 0.875407, which shows strong overall separation between admitted and discharged patients.

## Model Tuning

GridSearchCV was used to tune the Logistic Regression hyperparameters while keeping the same preprocessing pipeline. The search used recall as the scoring metric. The best parameters were `C=10`, `max_iter=1000`, and `solver="lbfgs"`. The best cross-validation recall was 0.803211.

Tuned Logistic Regression metrics:

| split | accuracy | recall | precision | specificity | f1 |
|---|---:|---:|---:|---:|---:|
| train | 0.787104 | 0.803856 | 0.607240 | 0.780016 | 0.691850 |
| test | 0.787131 | 0.801338 | 0.607695 | 0.781119 | 0.691211 |

Compared with the baseline model, tuning did not improve test recall, precision, or F1-score. The tuned model performed almost the same, with slightly lower test recall, precision, and F1.
