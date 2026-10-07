# ScamPulse AI — Sprint 1
# This script is split from ScamPulse_AI_Complete_Sprint1.ipynb.
# Run the scripts from this Python/ folder or directly from the project root.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
COMBINED_DIR = DATA_DIR / "combined"
PROCESSED_DIR = DATA_DIR / "processed"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _directory in [COMBINED_DIR, PROCESSED_DIR, VISUALIZATIONS_DIR, REPORTS_DIR]:
    _directory.mkdir(parents=True, exist_ok=True)

import os
import re
import textwrap
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# WORKFLOW: Load the selected raw FTC complaint CSV files.
raw_file_names = [
    "DNC_Complaint_sep1.csv",
    "DNC_Complaint_sep2.csv",
    "DNC_Complaint_sep3.csv",
    "DNC_Complaint_sep4.csv",
    "DNC_Complaint_sep7.csv",
    "DNC_Complaint_sep8.csv",
    "DNC_Complaint_sep9.csv",
    "DNC_Complaint_sep10.csv",
]

raw_datasets = []
for i, file_name in enumerate(raw_file_names, start=1):
    file_path = RAW_DIR / file_name
    df = pd.read_csv(file_path)
    raw_datasets.append(df)
    print(f"data{i}: {file_name} -> {df.shape[0]:,} rows × {df.shape[1]} columns")

# WORKFLOW: Combine all loaded raw datasets vertically by rows.
combined_data = pd.concat(
    raw_datasets,
    axis=0,
    ignore_index=True
)

print("Combined dataset shape:", combined_data.shape)

# WORKFLOW: Verify that all selected raw files use the same schema.
raw_columns = [list(df.columns) for df in raw_datasets]
print("All raw schemas identical:", len({tuple(cols) for cols in raw_columns}) == 1)
print("Expected number of columns in each raw file:", len(raw_columns[0]))

# WORKFLOW: Save the combined raw dataset.
combined_path = COMBINED_DIR / "DNC_September_2026_Combined_Raw.csv"
combined_data.to_csv(combined_path, index=False)
print(f"Combined raw dataset saved to: {combined_path}")
print("File exists:", combined_path.exists())

# WORKFLOW: Load the combined raw dataset for quality inspection.
dataset = pd.read_csv(combined_path)
print("Dataset shape:", dataset.shape)
print("Columns:", list(dataset.columns))
dataset.info()

# WORKFLOW: Check row-wise and column-wise missing values.
print("\nMissing values per row:")
print(dataset.isnull().sum(axis=1))

print("\nMissing values by column:")
print(dataset.isnull().sum())

# WORKFLOW: Inspect descriptive statistics across columns.
print("\nDescriptive statistics:")
print(dataset.describe(include="all"))

# WORKFLOW: Count exact duplicate rows.
duplicate_count = int(dataset.duplicated().sum())
print("\nNumber of duplicate rows:", duplicate_count)

# WORKFLOW: Inspect date ranges.
print("\nCreated Date:")
print(dataset["Created_Date"].min())
print(dataset["Created_Date"].max())
print("Violation Date:")
print(dataset["Violation_Date"].min())
print(dataset["Violation_Date"].max())

# WORKFLOW: Inspect robocall / recorded-message values.
print("\nRobocall values:")
print(dataset["Recorded_Message_Or_Robocall"].value_counts(dropna=False))

# WORKFLOW: Check unique city count.
print("\nUnique reported cities:", dataset["Consumer_City"].nunique())

# WORKFLOW: Check blank strings, unexpected robocall values, parseable dates, duplicates, and invalid date ordering before cleaning.
object_columns = dataset.select_dtypes(include=["object"]).columns
print("\nBlank / whitespace-only values by text column:")
for column in object_columns:
    blank_count = (
        dataset[column]
        .astype("string")
        .str.strip()
        .eq("")
        .sum()
    )
    print(f"{column}: {blank_count:,}")

allowed_robocall_values = {"Y", "N"}
observed_robocall_values = set(
    dataset["Recorded_Message_Or_Robocall"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()
    .unique()
)
unexpected_robocall_values = sorted(observed_robocall_values - allowed_robocall_values)
print("Unexpected robocall values:", unexpected_robocall_values)
print("Exact duplicate rows:", int(dataset.duplicated().sum()))

created_check = pd.to_datetime(dataset["Created_Date"], errors="coerce")
violation_check = pd.to_datetime(dataset["Violation_Date"], errors="coerce")
print("Invalid Created_Date values:", int(created_check.isna().sum()))
print("Invalid Violation_Date values:", int(violation_check.isna().sum()))
print("Violation_Date later than Created_Date:", int((violation_check > created_check).sum()))

print("\nData loading and quality checks completed.")
