# Online Shoppers Purchase Intent Classification

## a. Problem statement

Online shopping platforms record how a user moves through the site during a session (pages visited, time spent, bounce/exit rates, visitor type, and so on). Using that session information, this assignment predicts whether the visit will finish with a purchase (`Revenue = True` / `False`).

I trained five standard classifiers on the same UCI dataset, compared them with the same train/test split and metrics, saved the fitted pipelines, and built a Streamlit UI that evaluates an uploaded test CSV with those saved models (the app does not retrain).

## b. Dataset description

| Item | Detail |
|------|--------|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository ([dataset page](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)) |
| Task | Binary classification |
| Instances | 12,330 sessions |
| Features | 17 input attributes + 1 target |
| Target | `Revenue` — whether the session completed a purchase |
| Class balance | Approx. 84.5% non-purchase / 15.5% purchase (imbalanced) |

**Feature summary**

- **Numeric / count features:** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `Weekend`
- **Categorical features:** `Month`, `VisitorType`
- **Preprocessing:** `StandardScaler` on numeric columns; one-hot encoding on categorical columns; stratified 80/20 train–test split (`random_state=42`)

## c. Github Repository Link

**Repository:** `https://github.com/amitkumar570/online_shoppers_classifier`

Repository layout:

```
online_shoppers_classifier/
|-- app.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- data/online_shoppers_intention.csv
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- svm.joblib
|   |-- metrics.json
```

**Live Streamlit app:** `https://amitkumar570-online-shoppers-classifier-app-ghgvnd.streamlit.app/`

## d. Models used

All five models were trained on the same training split and evaluated on the same held-out test set (`test_data.csv`, 2,466 rows).

### Comparison table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|--------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression      | 0.8500   | 0.8963 | 0.5107    | 0.7487 | 0.6072 | 0.5330 |
| Decision Tree            | 0.8325   | 0.8766 | 0.4758    | 0.7984 | 0.5963 | 0.5256 |
| KNN                      | 0.8755   | 0.8213 | 0.6904    | 0.3560 | 0.4698 | 0.4361 |
| Naive Bayes              | 0.6736   | 0.7939 | 0.2941    | 0.7906 | 0.4287 | 0.3249 |
| Support Vector Machine   | 0.8569   | 0.8840 | 0.5279    | 0.7173 | 0.6082 | 0.5323 |

### Observations on model performance

| ML Model Name            | Observation about model performance |
|--------------------------|-------------------------------------|
| Logistic Regression      | Strongest overall ranking ability (highest AUC 0.8963) with a solid MCC (0.5330). Class-weight balancing helps it recover purchase sessions without collapsing precision as badly as Naive Bayes. Coefficients remain easy to interpret for session-level drivers such as `PageValues` and exit/bounce behaviour. |
| Decision Tree            | Highest recall (0.7984) among the tree/linear group, so it catches more true purchases, but precision drops to 0.4758 because several non-purchase sessions are also flagged. Depth and leaf constraints reduce overfit versus an unpruned tree; AUC (0.8766) stays competitive. |
| KNN                      | Highest raw accuracy (0.8755) yet weakest purchase recall (0.3560). On this imbalanced set, neighbour voting is dominated by the majority (non-purchase) class, so many genuine buyers are missed. Useful as a baseline, not as a purchase-detection model. |
| Naive Bayes              | Lowest accuracy (0.6736) and MCC (0.3249). The Gaussian independence assumption fits poorly with mixed scaled numeric + one-hot features, producing many false positives (precision 0.2941) despite decent recall. |
| Support Vector Machine   | Very close to logistic regression: AUC 0.8840, F1 0.6082, MCC 0.5323. The RBF kernel captures non-linear session patterns; with `class_weight='balanced'` it maintains good recall without the severe majority-class bias seen in KNN. |
| **Overall winner for this dataset?** | **Logistic Regression** — best AUC, tied-best MCC/F1 region, and the most reliable precision–recall trade-off for an imbalanced purchase-intent problem. SVM is a near second. |

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain models from scratch:

```bash
python train_models.py
```

## Streamlit Cloud notes

- Main file: `app.py`
- Upload `test_data.csv` (or any CSV with the same columns including `Revenue`)
- Choose a model from the dropdown to view metrics, confusion matrix, and classification report
