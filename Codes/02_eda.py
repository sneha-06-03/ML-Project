
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project folder location
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset and graph folder
DATA_PATH = BASE_DIR / "Dataset" / "PLFS_2024_Women_ML_Starter.csv"
GRAPH_DIR = BASE_DIR / "Graphs"

# Create Graphs folder if it does not exist
GRAPH_DIR.mkdir(exist_ok=True)

# Read dataset
df = pd.read_csv(DATA_PATH)

# Set graph style
sns.set_theme(style="whitegrid")

print("=" * 55)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 55)

# -----------------------------------------
# GRAPH 1: AGE DISTRIBUTION
# -----------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(data=df, x="AGE", bins=25, kde=True)

plt.title("Age Distribution of Women")
plt.xlabel("Age")
plt.ylabel("Number of Women")

plt.tight_layout()
plt.savefig(GRAPH_DIR / "01_age_distribution.png", dpi=300)
plt.show()


# -----------------------------------------
# GRAPH 2: EDUCATION VS PARTICIPATION
# -----------------------------------------

education_data = df.groupby("GEDU_LVL")["LFPR_TARGET"].mean() * 100

plt.figure(figsize=(10, 5))

education_data.plot(kind="bar")

plt.title("Education Level vs Labour Force Participation")
plt.xlabel("General Education Level Code")
plt.ylabel("Participation Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(GRAPH_DIR / "02_education_participation.png", dpi=300)
plt.show()


# -----------------------------------------
# GRAPH 3: RURAL VS URBAN PARTICIPATION
# -----------------------------------------

sector_data = df.groupby("SEC")["LFPR_TARGET"].mean() * 100

plt.figure(figsize=(7, 5))

sector_data.plot(kind="bar")

plt.title("Rural vs Urban Labour Force Participation")
plt.xlabel("Sector Code")
plt.ylabel("Participation Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(GRAPH_DIR / "03_sector_participation.png", dpi=300)
plt.show()


# -----------------------------------------
# GRAPH 4: MARITAL STATUS VS PARTICIPATION
# -----------------------------------------

marital_data = df.groupby("MARST")["LFPR_TARGET"].mean() * 100

plt.figure(figsize=(8, 5))

marital_data.plot(kind="bar")

plt.title("Marital Status vs Labour Force Participation")
plt.xlabel("Marital Status Code")
plt.ylabel("Participation Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(GRAPH_DIR / "04_marital_participation.png", dpi=300)
plt.show()


# -----------------------------------------
# GRAPH 5: CORRELATION HEATMAP
# -----------------------------------------

numerical_df = df[
    ["AGE", "FORM_EDU", "GEDU_LVL",
     "TEDU_LVL", "LFPR_TARGET"]
]

plt.figure(figsize=(9, 6))

sns.heatmap(
    numerical_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig(GRAPH_DIR / "05_correlation_heatmap.png", dpi=300)
plt.show()


# -----------------------------------------
# FINAL MESSAGE
# -----------------------------------------

print("\nEDA COMPLETED SUCCESSFULLY!")
print("All five graphs have been saved in the Graphs folder.")

print("\nGRAPH FILES:")
for file in sorted(GRAPH_DIR.glob("*.png")):
    print(file.name)
