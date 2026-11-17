
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "Dataset", "PLFS_2024_Women_ML_Starter.csv"
)

GRAPH_DIR = os.path.join(BASE_DIR, "Graphs")
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

os.makedirs(GRAPH_DIR, exist_ok=True)

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("NAIVE BAYES CLASSIFICATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

target = "LFPR_TARGET"

features = [
    "AGE",
    "MARST",
    "REL",
    "GEDU_LVL",
    "TEDU_LVL",
    "FORM_EDU",
    "SEC",
    "ST"
]

X = df[features].copy()
y = df[target].copy()

categorical_features = [
    "MARST",
    "REL",
    "GEDU_LVL",
    "TEDU_LVL",
    "FORM_EDU",
    "SEC",
    "ST"
]

numeric_features = ["AGE"]

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("age_binning", KBinsDiscretizer(
        n_bins=10,
        encode="ordinal",
        strategy="quantile"
    ))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    sparse_threshold=0
)

# ============================================================
# 5. BUILD NAIVE BAYES MODEL
# ============================================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", CategoricalNB(min_categories=200))
])

# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\nTraining Naive Bayes model...")

model.fit(X_train, y_train)

print("Model training completed successfully!")

# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ============================================================
# 8. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("NAIVE BAYES MODEL RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# ============================================================
# 9. SAVE MODEL METRICS
# ============================================================

metrics = pd.DataFrame({
    "Model": ["Categorical Naive Bayes"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1_Score": [f1],
    "ROC_AUC": [roc_auc]
})

metrics.to_csv(
    os.path.join(DATASET_DIR, "naive_bayes_metrics.csv"),
    index=False
)

# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Participating", "Participating"],
    yticklabels=["Not Participating", "Participating"]
)

plt.title("Naive Bayes - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "12_naive_bayes_confusion_matrix.png"),
    dpi=300
)

plt.close()

# ============================================================
# 11. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Naive Bayes (AUC = {roc_auc:.3f})"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("Naive Bayes - ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "13_naive_bayes_roc_curve.png"),
    dpi=300
)

plt.close()

# ============================================================
# 12. COMPLETION MESSAGE
# ============================================================

print("\nGraphs saved successfully!")
print("Metrics saved in Dataset folder.")

print("\n" + "=" * 60)
print("NAIVE BAYES CLASSIFICATION COMPLETED!")
print("=" * 60)
