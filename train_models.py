"""
Train five classifiers on the UCI Online Shoppers Purchasing Intention dataset.
Saves fitted models, preprocessing artifacts, metrics, and a held-out test CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
ARTIFACT_DIR = ROOT / "artifacts"
TARGET_COL = "Revenue"
RANDOM_SEED = 42


def load_and_prepare():
    frame = pd.read_csv(DATA_CSV)
    # Weekend / Revenue arrive as bool; keep target as 0/1 ints for metrics
    frame[TARGET_COL] = frame[TARGET_COL].astype(int)
    frame["Weekend"] = frame["Weekend"].astype(int)

    label_vec = frame[TARGET_COL]
    feat_frame = frame.drop(columns=[TARGET_COL])

    cat_cols = ["Month", "VisitorType"]
    num_cols = [c for c in feat_frame.columns if c not in cat_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        feat_frame,
        label_vec,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=label_vec,
    )
    return x_train, x_test, y_train, y_test, cat_cols, num_cols


def build_preprocessor(cat_cols, num_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )


def score_model(estimator, x_eval, y_true):
    y_hat = estimator.predict(x_eval)
    # probability / decision scores for AUC
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(x_eval)[:, 1]
    else:
        y_score = estimator.decision_function(x_eval)

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_hat)),
        "AUC": float(roc_auc_score(y_true, y_score)),
        "Precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "Recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "F1": float(f1_score(y_true, y_hat, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_hat)),
    }
    cm = confusion_matrix(y_true, y_hat).tolist()
    return metrics, cm, y_hat


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test, cat_cols, num_cols = load_and_prepare()

    # Persist held-out test set with target for Streamlit evaluation
    test_out = x_test.copy()
    test_out[TARGET_COL] = y_test.values
    test_path = ROOT / "test_data.csv"
    test_out.to_csv(test_path, index=False)
    print(f"Wrote {test_path} shape={test_out.shape}")

    preprocessor = build_preprocessor(cat_cols, num_cols)

    model_specs = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED,
        ),
    }

    file_keys = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "KNN": "knn",
        "Naive Bayes": "naive_bayes",
        "Support Vector Machine": "svm",
    }

    all_metrics = {}
    all_cms = {}

    for display_name, clf in model_specs.items():
        print(f"Training {display_name} ...")
        pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("clf", clf),
            ]
        )
        pipe.fit(x_train, y_train)
        metrics, cm, _ = score_model(pipe, x_test, y_test)
        all_metrics[display_name] = metrics
        all_cms[display_name] = cm

        out_file = MODEL_DIR / f"{file_keys[display_name]}.joblib"
        joblib.dump(pipe, out_file)
        print(f"  saved {out_file.name} | Acc={metrics['Accuracy']:.4f} AUC={metrics['AUC']:.4f}")

    meta = {
        "target_column": TARGET_COL,
        "feature_columns": list(x_train.columns),
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "class_labels": [0, 1],
        "positive_label_meaning": "Purchase completed (Revenue=True)",
        "dataset": "UCI Online Shoppers Purchasing Intention Dataset",
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "model_files": file_keys,
        "metrics": all_metrics,
        "confusion_matrices": all_cms,
    }
    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote {metrics_path}")

    # Pretty print comparison table
    print("\nComparison on held-out test set:")
    header = f"{'ML Model Name':<28} {'Accuracy':>9} {'AUC':>9} {'Precision':>10} {'Recall':>9} {'F1':>9} {'MCC':>9}"
    print(header)
    print("-" * len(header))
    for name, m in all_metrics.items():
        print(
            f"{name:<28} {m['Accuracy']:>9.4f} {m['AUC']:>9.4f} "
            f"{m['Precision']:>10.4f} {m['Recall']:>9.4f} {m['F1']:>9.4f} {m['MCC']:>9.4f}"
        )


if __name__ == "__main__":
    main()
