"""
Streamlit app: evaluate saved purchase-intent classifiers on an uploaded test CSV.
"""

import json
import os
from pathlib import Path

# Headless-friendly matplotlib (Streamlit Cloud)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib
from sklearn.metrics import (
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


@st.cache_resource
def load_meta():
    with open(META_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_resource
def load_pipeline(model_filename):
    path = MODEL_DIR / model_filename
    if not path.exists():
        raise FileNotFoundError("Missing model file: %s" % path)
    return joblib.load(path)


def compute_metrics(pipe, feat_frame, y_true):
    y_hat = pipe.predict(feat_frame)
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(feat_frame)[:, 1]
    else:
        y_score = pipe.decision_function(feat_frame)

    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_hat)),
        "AUC Score": float(roc_auc_score(y_true, y_score)),
        "Precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "Recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_hat, zero_division=0)),
        "MCC Score": float(matthews_corrcoef(y_true, y_hat)),
    }
    cm = confusion_matrix(y_true, y_hat)
    report = classification_report(
        y_true,
        y_hat,
        target_names=["No Purchase (0)", "Purchase (1)"],
        zero_division=0,
    )
    return metrics, cm, report, y_hat


def normalize_frame(work, target_col):
    """Align Weekend / Revenue dtypes with training."""
    out = work.copy()
    if target_col in out.columns:
        if out[target_col].dtype == bool or out[target_col].dtype == object:
            mapped = (
                out[target_col]
                .astype(str)
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
            )
            out[target_col] = mapped.fillna(out[target_col]).astype(int)
    if "Weekend" in out.columns:
        if out["Weekend"].dtype == bool or out["Weekend"].dtype == object:
            mapped = (
                out["Weekend"]
                .astype(str)
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
            )
            out["Weekend"] = mapped.fillna(out["Weekend"]).astype(int)
    return out


def main():
    st.set_page_config(
        page_title="Shopper Intent Classifiers",
        page_icon="🛒",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(165deg, #f3f6f4 0%, #e7eef2 55%, #f7f3ea 100%); }
        h1, h2, h3 { color: #1f3d3a !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Online Shoppers Purchase Intent")
    st.caption(
        "UCI Online Shoppers Purchasing Intention · five saved classifiers · "
        "upload test CSV (no retraining)"
    )

    if not META_PATH.exists():
        st.error("model/metrics.json not found. Repo layout may be incomplete.")
        st.write("Looking in:", str(META_PATH))
        st.write("Files here:", os.listdir(ROOT))
        return

    meta = load_meta()
    target_col = meta["target_column"]
    expected_feats = meta["feature_columns"]

    with st.expander("Dataset & setup notes", expanded=False):
        st.markdown(
            """
- **Source:** {dataset}
- **Target:** `{target}` (1 = completed purchase)
- **Train / test sizes:** {train_n} / {test_n}
- Upload a CSV with all feature columns **and** `{target}`.
- Models are loaded from saved `.joblib` pipelines (no retraining).
            """.format(
                dataset=meta.get("dataset", "UCI Online Shoppers"),
                target=target_col,
                train_n=meta.get("train_size", "?"),
                test_n=meta.get("test_size", "?"),
            )
        )

    st.subheader("1. Upload test data (CSV)")
    uploaded = st.file_uploader("Choose a test CSV file", type=["csv"])

    st.subheader("2. Select model")
    chosen = st.selectbox("Classifier", list(DISPLAY_TO_FILE.keys()), index=0)

    if uploaded is None:
        st.info("Upload a test CSV to see metrics and the confusion matrix.")
        st.subheader("Reference metrics (held-out test split from training)")
        rows = []
        for name, vals in meta["metrics"].items():
            rows.append(
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
        st.dataframe(pd.DataFrame(rows))
        return

    try:
        eval_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error("Could not read CSV: %s" % exc)
        return

    missing = [c for c in (expected_feats + [target_col]) if c not in eval_df.columns]
    if missing:
        st.error("CSV is missing required columns: %s" % missing)
        st.write("Expected features:", expected_feats)
        return

    work = normalize_frame(eval_df, target_col)
    y_true = work[target_col]
    feat_frame = work[expected_feats]

    try:
        pipe = load_pipeline(DISPLAY_TO_FILE[chosen])
        metrics, cm, report, y_hat = compute_metrics(pipe, feat_frame, y_true)
    except Exception as exc:
        st.error("Model evaluation failed: %s" % exc)
        st.exception(exc)
        return

    st.subheader("3. Evaluation metrics")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Accuracy", "%.4f" % metrics["Accuracy"])
    c2.metric("AUC Score", "%.4f" % metrics["AUC Score"])
    c3.metric("Precision", "%.4f" % metrics["Precision"])
    c4.metric("Recall", "%.4f" % metrics["Recall"])
    c5.metric("F1 Score", "%.4f" % metrics["F1 Score"])
    c6.metric("MCC Score", "%.4f" % metrics["MCC Score"])

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
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Classification report")
        st.code(report)

    st.subheader("Quick peek at predictions")
    preview = feat_frame.head(12).copy()
    preview["Actual"] = y_true.head(12).values
    preview["Predicted"] = y_hat[:12]
    st.dataframe(preview)

    st.subheader("Compare all models on this upload")
    if st.button("Run all five classifiers"):
        rows = []
        for label, fname in DISPLAY_TO_FILE.items():
            try:
                p = load_pipeline(fname)
                m, _, _, _ = compute_metrics(p, feat_frame, y_true)
                row = {"Model": label}
                row.update({k: round(v, 4) for k, v in m.items()})
                rows.append(row)
            except Exception as exc:
                rows.append({"Model": label, "Accuracy": "ERROR: %s" % exc})
        st.dataframe(pd.DataFrame(rows))


main()
