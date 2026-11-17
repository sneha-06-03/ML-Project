
import pandas as pd
from pathlib import Path

# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset location
DATA_PATH = BASE_DIR / "Dataset" / "PLFS_2024_Women_ML_Starter.csv"

# Output report location
REPORT_PATH = BASE_DIR / "Dataset" / "data_audit_report.txt"

print("=" * 60)
print("WOMEN'S LABOUR FORCE PARTICIPATION PROJECT")
print("DATASET INSPECTION")
print("=" * 60)

# Check whether the dataset exists
if not DATA_PATH.exists():
    print("\nERROR: Dataset file not found!")
    print("Please check that the CSV is inside the Dataset folder.")
    raise SystemExit

# Read the CSV file
df = pd.read_csv(DATA_PATH)

print("\n1. DATASET SIZE")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\n2. COLUMN NAMES")
print(df.columns.tolist())

print("\n3. FIRST FIVE RECORDS")
print(df.head())

print("\n4. MISSING VALUES")
print(df.isnull().sum())

print("\n5. DUPLICATE RECORDS")
print("Number of duplicate rows:", df.duplicated().sum())

print("\n6. TARGET VARIABLE DISTRIBUTION")
print(df["LFPR_TARGET"].value_counts(dropna=False))

print("\n7. TARGET VARIABLE PERCENTAGE")
print(
    df["LFPR_TARGET"]
    .value_counts(normalize=True, dropna=False) * 100
)

print("\n8. NUMERICAL DATA SUMMARY")
print(df.describe())

print("\n9. UNIQUE VALUES IN CATEGORICAL COLUMNS")

categorical_columns = [
    "MARST", "REL", "GEDU_LVL",
    "TEDU_LVL", "SEC", "ST"
]

for column in categorical_columns:
    if column in df.columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).head(20))

# Save inspection results in a text file
with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("PLFS DATASET INSPECTION REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Rows: {df.shape[0]}\n")
    file.write(f"Columns: {df.shape[1]}\n\n")

    file.write("COLUMN NAMES:\n")
    file.write(str(df.columns.tolist()) + "\n\n")

    file.write("MISSING VALUES:\n")
    file.write(df.isnull().sum().to_string() + "\n\n")

    file.write("DUPLICATE ROWS:\n")
    file.write(str(df.duplicated().sum()) + "\n\n")

    file.write("TARGET DISTRIBUTION:\n")
    file.write(df["LFPR_TARGET"].value_counts(dropna=False).to_string())

print("\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED!")
print("Report saved at:", REPORT_PATH)
print("=" * 60)
