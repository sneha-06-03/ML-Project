
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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
print("DECISION TREE CLASSIFICATION")
print("=" * 60)

# -----------------------------------------
# 2. SELECT FEATURES AND TARGET
# -----------------------------------------

# ACWS is excluded to prevent data leakage.
# MULT is not used as an individual predictor.

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
    ("imputer", SimpleImputer(strategy="median"))
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
# 5. TRAIN DECISION TREE
# -----------------------------------------

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTraining Decision Tree model...")

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

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Not Participating", "Participating"],
    cmap="Blues",
    ax=ax
)

plt.title("Decision Tree - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "08_decision_tree_confusion_matrix.png",
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

plt.title("Decision Tree - ROC Curve")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "09_decision_tree_roc_curve.png",
    dpi=300
)

plt.show()

# -----------------------------------------
# 10. FEATURE IMPORTANCE
# -----------------------------------------

fitted_preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()

importance = classifier.feature_importances_

importance_table = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_table = importance_table.sort_values(
    by="Importance",
    ascending=False
)

importance_table.to_csv(
    BASE_DIR / "Dataset" / "decision_tree_feature_importance.csv",
    index=False
)

print("\nTOP 10 IMPORTANT FEATURES")
print(importance_table.head(10).to_string(index=False))

# Feature importance graph
top_features = importance_table.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Decision Tree - Top 10 Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "10_decision_tree_feature_importance.png",
    dpi=300
)

plt.show()

# -----------------------------------------
# 11. DECISION TREE VISUALIZATION
# -----------------------------------------

plt.figure(figsize=(22, 12))

plot_tree(
    classifier,
    feature_names=feature_names,
    class_names=["Not Participating", "Participating"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7
)

plt.title("Decision Tree Structure (First Three Levels)")
plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "11_decision_tree_structure.png",
    dpi=300
)

plt.show()

# -----------------------------------------
# 12. SAVE MODEL METRICS
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
    BASE_DIR / "Dataset" / "decision_tree_metrics.csv",
    index=False
)

print("\n" + "=" * 60)
print("DECISION TREE COMPLETED SUCCESSFULLY!")
print("Metrics and feature importance saved.")
print("Graphs saved in the Graphs folder.")
print("=" * 60)
