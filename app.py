import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Page configuration and paths
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Satellite Intelligence | Land Cover Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLUMNS = [f"Attribute{i}" for i in range(1, 37)]
TARGET_COLUMN = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(31, 104, 133, 0.20), transparent 30%),
            linear-gradient(135deg, #081522 0%, #0B1D2B 55%, #102638 100%);
        color: #E8F1F7;
    }
    [data-testid="stHeader"] { background: rgba(8, 21, 34, 0.85); }
    [data-testid="stSidebar"] {
        background: #0A1926;
        border-right: 1px solid rgba(125, 211, 252, 0.15);
    }
    [data-testid="stSidebar"] * { color: #DCEAF2; }
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #F2F8FC !important;
    }
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: #C3D4DF;
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(125, 211, 252, 0.22);
        border-radius: 20px;
        background: linear-gradient(115deg, rgba(18, 49, 68, 0.96), rgba(12, 31, 46, 0.88));
        margin-bottom: 1.3rem;
        box-shadow: 0 14px 40px rgba(0,0,0,0.18);
    }
    .eyebrow {
        color: #7DD3FC;
        letter-spacing: 0.16em;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .hero-title {
        color: #F4FAFD;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.8rem, 4vw, 2.8rem);
        font-weight: 700;
        line-height: 1.1;
        margin: 0.45rem 0;
    }
    .hero-copy { color: #B8CEDB; font-size: 0.98rem; max-width: 760px; }
    .metric-card {
        background: rgba(19, 48, 66, 0.82);
        border: 1px solid rgba(125, 211, 252, 0.18);
        border-radius: 15px;
        padding: 1rem 1.1rem;
        min-height: 105px;
    }
    .metric-label {
        color: #A9C4D3;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        color: #7DD3FC;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .section-note { color: #9FB7C6; font-size: 0.88rem; }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(125, 211, 252, 0.16);
        border-radius: 12px;
        overflow: hidden;
    }
    .stButton > button {
        background: #38BDF8;
        color: #071724;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
    }
    .stButton > button:hover {
        background: #7DD3FC;
        color: #071724;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Load saved models (no training takes place in this application)
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    loaded = {}
    missing = []

    for display_name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            missing.append(filename)
            continue
        loaded[display_name] = joblib.load(path)

    return loaded, missing


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------
def calculate_metrics(model, X, y_true):
    y_pred = model.predict(X)
    y_probability = model.predict_proba(X)

    model_classes = np.asarray(model.classes_)
    observed_classes = np.unique(y_true)

    # Macro OVR AUC requires the test data to contain all model classes.
    if len(observed_classes) == len(model_classes) and set(observed_classes) == set(model_classes):
        auc_value = roc_auc_score(
            y_true,
            y_probability,
            labels=model_classes,
            multi_class="ovr",
            average="macro",
        )
    else:
        auc_value = np.nan

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc_value,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    return metrics, y_pred


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Remote sensing • Machine learning</div>
        <div class="hero-title">Satellite Intelligence</div>
        <div class="hero-copy">
            Land-cover classification using multispectral satellite measurements.
            Evaluate any of the five trained classifiers on your labelled test data
            and inspect its metrics and classification results.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🛰️ Mission Control")
    st.caption("LANDSAT · STATLOG DATASET")
    st.markdown("---")
    st.markdown("**Evaluation workflow**")
    st.markdown("1. Upload labelled test CSV")
    st.markdown("2. Validate 36 attributes")
    st.markdown("3. Select a trained model")
    st.markdown("4. View metrics and classification results")
    st.markdown("---")


# ---------------------------------------------------------------------
# Load model files
# ---------------------------------------------------------------------
try:
    saved_models, missing_models = load_models()
except Exception as exc:
    st.error(f"Could not load the saved models: {exc}")
    st.stop()

if missing_models:
    st.error(
        "The following model files are missing from the `models/` folder: "
        + ", ".join(missing_models)
    )
    st.info("Place all five saved .joblib files inside the project's models folder.")
    st.stop()

# ---------------------------------------------------------------------
# Upload and validate labelled test data
# ---------------------------------------------------------------------
st.subheader("01 · Upload test dataset")
st.markdown(
    '<div class="section-note">Upload a CSV containing Attribute1–Attribute36 and the actual Class label.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a labelled test CSV",
    type=["csv"],
    help="The CSV must contain all 36 numerical input features and the Class target column.",
)

if uploaded_file is None:
    st.info("Upload your CSV file.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read the uploaded CSV: {exc}")
    st.stop()

missing_columns = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in data.columns]

if missing_columns:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(missing_columns))
    st.stop()

if data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

if data[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
    st.error("Missing values were found in the required columns. Please clean the CSV and upload it again.")
    st.stop()

try:
    X_test = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    y_test = pd.to_numeric(data[TARGET_COLUMN], errors="raise").astype(int)
except Exception:
    st.error("All 36 input features and the Class target must contain numeric values.")
    st.stop()

if not np.isfinite(X_test.to_numpy(dtype=float)).all():
    st.error("The input features contain infinite or non-finite values.")
    st.stop()

expected_classes = set(np.asarray(next(iter(saved_models.values())).classes_).tolist())
uploaded_classes = set(y_test.unique().tolist())

if not uploaded_classes.issubset(expected_classes):
    st.error(
        "The uploaded target column contains class labels not used during training. "
        f"Expected labels: {sorted(expected_classes)}. "
        f"Found labels: {sorted(uploaded_classes)}."
    )
    st.stop()

st.success(f"Dataset validated successfully · {len(data):,} records · 36 features")

with st.expander("Preview uploaded dataset"):
    st.dataframe(data.head(10), use_container_width=True)

# ---------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------
st.subheader("02 · Test dataset overview")
summary_columns = st.columns(3)

summary_items = [
    ("TEST RECORDS", f"{len(data):,}"),
    ("INPUT FEATURES", f"{len(FEATURE_COLUMNS)}"),
    ("CLASSES PRESENT", f"{y_test.nunique()}"),
]

for column, (label, value) in zip(summary_columns, summary_items):
    column.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Individual model details
# ---------------------------------------------------------------------
st.subheader("03 · Evaluate a classifier")
selected_model_name = st.selectbox(
    "Select a trained model",
    list(saved_models.keys()),
)

selected_model = saved_models[selected_model_name]

try:
    selected_metrics, selected_predictions = calculate_metrics(
        selected_model, X_test, y_test
    )
except Exception as exc:
    st.error(f"Could not evaluate {selected_model_name}: {exc}")
    st.stop()

metric_columns = st.columns(6)
for column, (metric_name, metric_value) in zip(metric_columns, selected_metrics.items()):
    display_value = "N/A" if pd.isna(metric_value) else f"{metric_value:.4f}"
    column.markdown(
        f'<div class="metric-card"><div class="metric-label">{metric_name}</div>'
        f'<div class="metric-value">{display_value}</div></div>',
        unsafe_allow_html=True,
    )

# Confusion matrix
st.markdown("#### Confusion matrix")
class_labels = sorted(expected_classes)
cm = confusion_matrix(y_test, selected_predictions, labels=class_labels)

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#102638")
ax.set_facecolor("#102638")

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="crest",
    xticklabels=class_labels,
    yticklabels=class_labels,
    linewidths=0.5,
    linecolor="#29485A",
    cbar=True,
    ax=ax,
)

ax.set_xlabel("Predicted class", color="#DCEAF2")
ax.set_ylabel("Actual class", color="#DCEAF2")
ax.set_title(f"{selected_model_name} · Confusion Matrix", color="#F2F8FC", pad=14)
ax.tick_params(colors="#DCEAF2")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Classification report
st.markdown("#### Classification report")
report = classification_report(
    y_test,
    selected_predictions,
    labels=class_labels,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()
st.dataframe(
    report_df.style.format({
        "precision": "{:.4f}",
        "recall": "{:.4f}",
        "f1-score": "{:.4f}",
        "support": "{:.0f}",
    }),
    use_container_width=True,
)


st.markdown("---")
st.caption(
    "Satellite Intelligence · Statlog (Landsat Satellite) · "
    "Predictions and metrics are generated from the uploaded labelled test data."
)
