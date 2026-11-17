
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# -----------------------------------------
# 1. PATHS AND DATASET
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "Dataset" / "PLFS_2024_Women_ML_Starter.csv"
GRAPH_DIR = BASE_DIR / "Graphs"

GRAPH_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

# -----------------------------------------
# 2. SELECT FEATURES AND TARGET
# -----------------------------------------

# ACWS is excluded because it was used to create the target.
# MULT is a survey multiplier, not an individual predictor.

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

X = df[features]
y = df["LFPR_TARGET"]

# Treat coded categories as categorical variables.
categorical_features = [
    "MARST",
    "REL",
    "GEDU_LVL",
    "TEDU_LVL",
    "SEC",
    "ST"
]

numerical_features = [
    "AGE",
    "FORM_EDU"
]

# -----------------------------------------
# 3. PREPROCESSING
# -----------------------------------------

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# -----------------------------------------
# 4. TRAIN-TEST SPLIT
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# -----------------------------------------
# 5. TRAIN LOGISTIC REGRESSION
# -----------------------------------------

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

print("\nTraining model...")

model.fit(X_train, y_train)

# -----------------------------------------
# 6. MAKE PREDICTIONS
# -----------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------------------
# 7. EVALUATION METRICS
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nMODEL PERFORMANCE")
print("-" * 40)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# -----------------------------------------
# 8. CONFUSION MATRIX
# -----------------------------------------

plt.figure(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Not Participating", "Participating"],
    cmap="Blues"
)

plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "06_logistic_confusion_matrix.png",
    dpi=300
)

plt.show()

# -----------------------------------------
# 9. ROC CURVE
# -----------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    ax=ax
)

plt.title("Logistic Regression - ROC Curve")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "07_logistic_roc_curve.png",
    dpi=300
)

plt.show()

# -----------------------------------------
# 10. ODDS RATIOS
# -----------------------------------------

fitted_preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()

coefficients = classifier.coef_[0]

odds_ratios = np.exp(coefficients)

odds_table = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Odds_Ratio": odds_ratios
})

odds_table = odds_table.sort_values(
    by="Odds_Ratio",
    ascending=False
)

odds_table.to_csv(
    BASE_DIR / "Dataset" / "logistic_odds_ratios.csv",
    index=False
)

print("\nTOP 10 ODDS RATIOS")
print(odds_table.head(10).to_string(index=False))

print("\nBOTTOM 10 ODDS RATIOS")
print(odds_table.tail(10).to_string(index=False))

# -----------------------------------------
# 11. SAVE MODEL METRICS
# -----------------------------------------

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics.to_csv(
    BASE_DIR / "Dataset" / "logistic_metrics.csv",
    index=False
)

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION COMPLETED!")
print("Metrics and odds ratios saved in Dataset folder.")
print("Graphs saved in Graphs folder.")
print("=" * 60)
