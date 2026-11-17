
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
GRAPH_DIR = os.path.join(BASE_DIR, "Graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)

print("=" * 65)
print("COMPARISON OF MACHINE LEARNING MODELS")
print("=" * 65)

# ============================================================
# 2. MODEL FILES
# ============================================================

model_files = {
    "Logistic Regression": "logistic_metrics.csv",
    "Decision Tree": "decision_tree_metrics.csv",
    "Categorical Naive Bayes": "naive_bayes_metrics.csv"
}

required_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

# ============================================================
# 3. FUNCTION TO READ MODEL METRICS
# ============================================================

def normalize_metric_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]", "", name)

    aliases = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1_Score",
        "f1score": "F1_Score",
        "rocauc": "ROC_AUC",
        "auc": "ROC_AUC"
    }

    return aliases.get(name)


def load_metrics(file_path, model_name):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Missing results file: {file_path}"
        )

    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError(f"{file_path} is empty.")

    result = {}

    # Format 1: Metric and Value columns
    if {"Metric", "Value"}.issubset(data.columns):

        for _, row in data.iterrows():

            metric_name = normalize_metric_name(row["Metric"])

            if metric_name is not None:
                result[metric_name] = float(row["Value"])

    # Format 2: One row with separate metric columns
    else:

        row = data.iloc[0]

        for column in data.columns:

            metric_name = normalize_metric_name(column)

            if metric_name is not None:
                result[metric_name] = float(row[column])

    missing = [
        metric for metric in required_metrics
        if metric not in result
    ]

    if missing:
        raise ValueError(
            f"\nCould not find {missing} in {model_name} results.\n"
            f"Available columns: {list(data.columns)}\n"
            "Please check the corresponding metrics CSV."
        )

    return result


# ============================================================
# 4. LOAD ALL MODEL RESULTS
# ============================================================

results = []

for model_name, filename in model_files.items():

    file_path = os.path.join(DATASET_DIR, filename)

    model_metrics = load_metrics(file_path, model_name)

    model_metrics["Model"] = model_name

    results.append(model_metrics)

    print(f"\nLoaded: {model_name}")

# ============================================================
# 5. CREATE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(results)

comparison = comparison[
    ["Model"] + required_metrics
]

print("\n" + "=" * 65)
print("MODEL COMPARISON TABLE")
print("=" * 65)

print(comparison.to_string(index=False))

comparison.to_csv(
    os.path.join(DATASET_DIR, "model_comparison.csv"),
    index=False
)

print("\nComparison table saved successfully!")

# ============================================================
# 6. FIND BEST MODEL FOR EACH METRIC
# ============================================================

print("\n" + "=" * 65)
print("BEST MODEL FOR EACH METRIC")
print("=" * 65)

for metric in required_metrics:

    best_index = comparison[metric].idxmax()

    best_model = comparison.loc[best_index, "Model"]
    best_score = comparison.loc[best_index, metric]

    print(
        f"{metric}: {best_model} "
        f"({best_score:.4f})"
    )

# ============================================================
# 7. CREATE COMPARISON GRAPH
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(12, 7))

for i, row in comparison.iterrows():

    values = [row[metric] for metric in metrics]

    bars = plt.bar(
        x + (i - 1) * width,
        values,
        width,
        label=row["Model"]
    )

    plt.bar_label(
        bars,
        fmt="%.2f",
        fontsize=8,
        padding=2
    )

plt.title(
    "Performance Comparison of Machine Learning Models",
    fontsize=14
)

plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")

plt.xticks(
    x,
    ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
)

plt.ylim(0, 1.15)

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "14_model_comparison.png"),
    dpi=300
)

plt.close()

# ============================================================
# 8. COMPLETION MESSAGE
# ============================================================

print("\nComparison graph saved successfully!")

print("\n" + "=" * 65)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
print("=" * 65)
