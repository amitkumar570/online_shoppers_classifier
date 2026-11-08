"""
Streamlit frontend for Online Shoppers purchase-intent classifiers.
Loads pre-trained pipelines and scores an uploaded test CSV (no retraining).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
META_PATH = MODEL_DIR / "metrics.json"

DISPLAY_TO_FILE = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbor": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_NAME_MAP = {
    "Logistic Regression": "Logistic Regression",
    "Decision Tree": "Decision Tree",
    "K-Nearest Neighbor": "KNN",
    "Naive Bayes (Gaussian)": "Naive Bayes",
    "Support Vector Machine": "Support Vector Machine",
}


@st.cache_resource
def load_meta():
    with open(META_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource
def load_pipeline(model_filename: str):
    return joblib.load(MODEL_DIR / model_filename)


def compute_metrics(pipe, feat_frame, y_true):
    y_hat = pipe.predict(feat_frame)
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(feat_frame)[:, 1]
    else:
        y_score = pipe.decision_function(feat_frame)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_hat),
        "AUC Score": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_hat, zero_division=0),
        "Recall": recall_score(y_true, y_hat, zero_division=0),
        "F1 Score": f1_score(y_true, y_hat, zero_division=0),
        "MCC Score": matthews_corrcoef(y_true, y_hat),
    }
    cm = confusion_matrix(y_true, y_hat)
    report = classification_report(
        y_true,
        y_hat,
        target_names=["No Purchase (0)", "Purchase (1)"],
        zero_division=0,
    )
    return metrics, cm, report, y_hat


def style_page():
    st.set_page_config(
        page_title="Shopper Intent Classifiers",
        page_icon="🛒",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(165deg, #f3f6f4 0%, #e7eef2 55%, #f7f3ea 100%); }
        .block-container { padding-top: 1.4rem; }
        h1, h2, h3 { color: #1f3d3a !important; font-family: "Trebuchet MS", sans-serif; }
        div[data-testid="stMetricValue"] { color: #0f5c56; }
        .soft-card {
            background: rgba(255,255,255,0.78);
            border: 1px solid #c9d6d3;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    style_page()
    meta = load_meta()
    target_col = meta["target_column"]
    expected_feats = meta["feature_columns"]

    st.title("Online Shoppers Purchase Intent")
    st.caption(
        "UCI Online Shoppers Purchasing Intention · five pre-trained classifiers · "
        "upload test CSV to evaluate without retraining"
    )

    with st.expander("Dataset & setup notes", expanded=False):
        st.markdown(
            f"""
            - **Source:** {meta['dataset']}
            - **Target:** `{target_col}` (1 = completed purchase)
            - **Train / test sizes used during training:** {meta['train_size']} / {meta['test_size']}
            - Upload a CSV that includes all feature columns **and** the `{target_col}` column.
            - Models are loaded from saved `.joblib` pipelines (scaler + encoder + classifier).
            """
        )

    st.subheader("1. Upload test data (CSV)")
    uploaded = st.file_uploader(
        "Choose a test CSV file",
        type=["csv"],
        help="Use the provided test_data.csv or another hold-out file with the same schema.",
    )

    st.subheader("2. Select model")
    chosen = st.selectbox(
        "Classifier",
        list(DISPLAY_TO_FILE.keys()),
        index=0,
    )

    if uploaded is None:
        st.info("Upload a test CSV to see metrics and the confusion matrix.")
        st.subheader("Reference metrics (from training run on held-out test split)")
        ref = meta["metrics"]
        ref_rows = []
        for name, vals in ref.items():
            ref_rows.append(
                {
                    "ML Model Name": name,
                    "Accuracy": round(vals["Accuracy"], 4),
                    "AUC": round(vals["AUC"], 4),
                    "Precision": round(vals["Precision"], 4),
                    "Recall": round(vals["Recall"], 4),
                    "F1": round(vals["F1"], 4),
                    "MCC": round(vals["MCC"], 4),
                }
            )
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True)
        return

    try:
        eval_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    missing = [c for c in expected_feats + [target_col] if c not in eval_df.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        st.write("Expected feature columns:", expected_feats)
        return

    # Align dtypes similar to training
    work = eval_df.copy()
    if work[target_col].dtype == bool or work[target_col].dtype == object:
        work[target_col] = work[target_col].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
        ).fillna(work[target_col])
        work[target_col] = work[target_col].astype(int)
    if "Weekend" in work.columns:
        if work["Weekend"].dtype == bool or work["Weekend"].dtype == object:
            work["Weekend"] = work["Weekend"].astype(str).str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0}
            ).fillna(work["Weekend"])
            work["Weekend"] = work["Weekend"].astype(int)

    y_true = work[target_col]
    feat_frame = work[expected_feats]

    pipe = load_pipeline(DISPLAY_TO_FILE[chosen])
    metrics, cm, report, y_hat = compute_metrics(pipe, feat_frame, y_true)

    st.subheader("3. Evaluation metrics")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    c2.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
    c3.metric("Precision", f"{metrics['Precision']:.4f}")
    c4.metric("Recall", f"{metrics['Recall']:.4f}")
    c5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    c6.metric("MCC Score", f"{metrics['MCC Score']:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("4. Confusion matrix")
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="GnBu",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Actual 0", "Actual 1"],
            ax=ax,
            cbar=False,
        )
        ax.set_title(chosen)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with right:
        st.subheader("Classification report")
        st.code(report)

    st.subheader("Quick peek at predictions")
    preview = feat_frame.head(12).copy()
    preview["Actual"] = y_true.head(12).values
    preview["Predicted"] = y_hat[:12]
    st.dataframe(preview, use_container_width=True)

    # Side-by-side comparison for all five models on the same upload
    st.subheader("Compare all models on this upload")
    if st.button("Run all five classifiers"):
        rows = []
        for label, fname in DISPLAY_TO_FILE.items():
            p = load_pipeline(fname)
            m, _, _, _ = compute_metrics(p, feat_frame, y_true)
            rows.append({"Model": label, **{k: round(v, 4) for k, v in m.items()}})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    main()
