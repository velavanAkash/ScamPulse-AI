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

# WORKFLOW: Load the combined raw dataset.
combined_path = COMBINED_DIR / "DNC_September_2026_Combined_Raw.csv"
dataset = pd.read_csv(combined_path)
print("Raw dataset shape:", dataset.shape)
print(dataset.head())

# WORKFLOW: Create a separate working copy for cleaning.
cleaned_dataset = dataset.copy()

# Shared dependency from the notebook quality section.
object_columns = cleaned_dataset.select_dtypes(include=["object"]).columns

# WORKFLOW: Check exact duplicates before removal.
print("Duplicate rows:", cleaned_dataset.duplicated().sum())

# WORKFLOW: Remove exact duplicate rows and reset the index.
cleaned_dataset = (
    cleaned_dataset
    .drop_duplicates()
    .reset_index(drop=True)
)
print("Shape after removing duplicates:", cleaned_dataset.shape)

# WORKFLOW: Confirm no exact duplicates remain.
print(
    "Remaining duplicate rows:",
    cleaned_dataset.duplicated().sum()
)

# WORKFLOW: Check missing values before cleaning.
print(cleaned_dataset.isnull().sum())

# WORKFLOW: Identify blank strings in object columns.
object_columns = cleaned_dataset.select_dtypes(
    include=["object"]
).columns

for column in object_columns:
    blank_count = (
        cleaned_dataset[column]
        .astype("string")
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"{column}: {blank_count} blank values")

# WORKFLOW: Convert blank strings to missing values.
for column in object_columns:
    cleaned_dataset[column] = (
        cleaned_dataset[column]
        .replace(r"^\s*$", pd.NA, regex=True)
    )

# WORKFLOW: Convert Created_Date to datetime.
cleaned_dataset["Created_Date"] = pd.to_datetime(
    cleaned_dataset["Created_Date"],
    errors="coerce"
)

# WORKFLOW: Convert Violation_Date to datetime.
cleaned_dataset["Violation_Date"] = pd.to_datetime(
    cleaned_dataset["Violation_Date"],
    errors="coerce"
)

# WORKFLOW: Check invalid datetime values after conversion.
print(
    "Invalid Created_Date:",
    cleaned_dataset["Created_Date"].isna().sum()
)

print(
    "Invalid Violation_Date:",
    cleaned_dataset["Violation_Date"].isna().sum()
)

# WORKFLOW: Inspect the Created_Date and Violation_Date ranges.
print("Created Date Range:")
print(cleaned_dataset["Created_Date"].min())
print(cleaned_dataset["Created_Date"].max())

print("\nViolation Date Range:")
print(cleaned_dataset["Violation_Date"].min())
print(cleaned_dataset["Violation_Date"].max())

# WORKFLOW: Flag records where Violation_Date occurs after Created_Date.
cleaned_dataset["Invalid_Date_Order"] = (
    cleaned_dataset["Violation_Date"]
    > cleaned_dataset["Created_Date"]
)

# WORKFLOW: Inspect the original phone-number values and dtype.
print(
    cleaned_dataset["Company_Phone_Number"]
    .head(20)
)

print(
    cleaned_dataset["Company_Phone_Number"]
    .dtype
)

# WORKFLOW: Normalize valid company phone numbers to 10-digit strings and label missing/invalid values as Not_Reported.
# Convert phone numbers to numeric where possible
phone_numeric = pd.to_numeric(
    cleaned_dataset["Company_Phone_Number"],
    errors="coerce"
)

# Keep only valid 10-digit phone numbers
valid_phone = phone_numeric.between(
    1000000000,
    9999999999
)

# Start with missing values
cleaned_dataset["Company_Phone_Number"] = pd.Series(
    pd.NA,
    index=cleaned_dataset.index,
    dtype="string"
)

# Convert valid numbers to 10-digit strings
cleaned_dataset.loc[valid_phone, "Company_Phone_Number"] = (
    phone_numeric.loc[valid_phone]
    .round()
    .astype("int64")
    .astype("string")
)

# Represent missing/invalid phone numbers
cleaned_dataset["Company_Phone_Number"] = (
    cleaned_dataset["Company_Phone_Number"]
    .fillna("Not_Reported")
)

# WORKFLOW: Inspect phone-number string lengths after normalization.
phone_length = (
    cleaned_dataset["Company_Phone_Number"]
    .str.len()
)
print(
    phone_length.value_counts(dropna=False).sort_index()
)

# WORKFLOW: Identify any remaining non-10-digit phone values.
invalid_phone = cleaned_dataset.loc[
    (cleaned_dataset["Company_Phone_Number"] != "Not_Reported") &
    (cleaned_dataset["Company_Phone_Number"].str.len() != 10),
    "Company_Phone_Number"
]
print(invalid_phone)

# WORKFLOW: Summarize lengths of remaining invalid phone values.
print(
    invalid_phone.astype("string").str.len().value_counts()
)

# WORKFLOW: Inspect example invalid phone values.
print(
    cleaned_dataset.loc[
        (cleaned_dataset["Company_Phone_Number"] != "Not_Reported") &
        (cleaned_dataset["Company_Phone_Number"].str.len() != 10),
        "Company_Phone_Number"
    ].head(100).to_list()
)

# WORKFLOW: Inspect the most frequent company phone values.
cleaned_dataset["Company_Phone_Number"].value_counts().head(5)

# WORKFLOW: Confirm that reported phone values have 10 digits.
phone_check = cleaned_dataset.loc[
    cleaned_dataset["Company_Phone_Number"] != "Not_Reported",
    "Company_Phone_Number"
].str.len()

print(phone_check.value_counts())

# WORKFLOW: Compare original missing and non-missing phone counts.
original_phone = dataset["Company_Phone_Number"]

print(
    "Original missing:",
    original_phone.isna().sum()
)

print(
    "Original non-missing:",
    original_phone.notna().sum()
)

# WORKFLOW: Inspect original phone-number string length distribution.
print(
    original_phone[original_phone.notna()]
    .astype("string")
    .str.len()
    .value_counts()
)

# WORKFLOW: Inspect sample non-missing source phone values.
print(
    dataset["Company_Phone_Number"]
    .dropna()
    .head(20)
)

# WORKFLOW: Inspect the sample 12-character phone values found in the source data.
phone_str = dataset["Company_Phone_Number"].dropna().astype(str)

print(phone_str[phone_str.str.len() == 12].head(20).to_list())

# WORKFLOW: Re-test phone-number numeric conversion as in the source workflow.
phone_numeric = pd.to_numeric(
    cleaned_dataset["Company_Phone_Number"],
    errors="coerce"
)

cleaned_dataset["Company_Phone_Number"] = (
    phone_numeric
    .apply(lambda x: str(int(x)) if pd.notna(x) else pd.NA)
    .astype("string")
)

# WORKFLOW: Inspect phone lengths after the re-test step.
print(
    cleaned_dataset["Company_Phone_Number"]
    .str.len()
    .value_counts(dropna=False)
)

# WORKFLOW: Build a separate phone cleaning test from the raw data.
phone_numeric = pd.to_numeric(
    dataset["Company_Phone_Number"],
    errors="coerce"
)

phone_clean_test = (
    phone_numeric
    .apply(lambda x: str(int(x)) if pd.notna(x) else pd.NA)
    .astype("string")
)

print(
    phone_clean_test.str.len()
    .value_counts(dropna=False)
)

# WORKFLOW: Inspect invalid phone values from the test series.
invalid_phone = phone_clean_test[
    phone_clean_test.notna() &
    (phone_clean_test.str.len() != 10)
]

print(invalid_phone.head(50).to_list())

# WORKFLOW: Inspect 11-digit phone values from the test series.
phone_11 = phone_clean_test[
    phone_clean_test.str.len() == 11
]
print(phone_11.head(50).to_list())

# WORKFLOW: Replace remaining invalid phone values with Not_Reported.
invalid_phone = (
    cleaned_dataset["Company_Phone_Number"].notna()
    & (cleaned_dataset["Company_Phone_Number"].str.len() != 10)
)

cleaned_dataset.loc[
    invalid_phone,
    "Company_Phone_Number"
] = "Not_Reported"

# WORKFLOW: Validate phone values against the 10-digit numeric-string pattern.
invalid_phone = (
    cleaned_dataset["Company_Phone_Number"].notna()
    &
    ~cleaned_dataset["Company_Phone_Number"].str.match(
        r"^\d{10}$",
        na=False
    )
)

print(
    "Invalid phone numbers:",
    invalid_phone.sum()
)

# WORKFLOW: Inspect examples of any phone values that fail validation.
cleaned_dataset.loc[
    invalid_phone,
    "Company_Phone_Number"
].head(20)

# WORKFLOW: Fill any remaining missing phone values with Not_Reported.
cleaned_dataset["Company_Phone_Number"] = (
    cleaned_dataset["Company_Phone_Number"]
    .fillna("Not_Reported")
)

# WORKFLOW: Convert consumer area codes to three-digit string identifiers.
area_code_numeric = pd.to_numeric(
    cleaned_dataset["Consumer_Area_Code"],
    errors="coerce"
)

cleaned_dataset["Consumer_Area_Code"] = (
    area_code_numeric
    .astype("Int64")
    .astype("string")
)

# WORKFLOW: Validate area-code formatting.
invalid_area_code = (
    cleaned_dataset["Consumer_Area_Code"].notna()
    &
    ~cleaned_dataset["Consumer_Area_Code"].str.match(
        r"^\d{3}$",
        na=False
    )
)

print(
    "Invalid area codes:",
    invalid_area_code.sum()
)

# WORKFLOW: Fill missing area codes with Not_Reported.
cleaned_dataset["Consumer_Area_Code"] = (
    cleaned_dataset["Consumer_Area_Code"]
    .fillna("Not_Reported")
)

# WORKFLOW: Define the text columns to standardize.
text_columns = [
    "Consumer_City",
    "Consumer_State",
    "Subject",
    "Recorded_Message_Or_Robocall"
]
print(text_columns)

# WORKFLOW: Convert text fields to string dtype.
for column in text_columns:
    cleaned_dataset[column] = (
        cleaned_dataset[column]
        .astype("string")
    )

# WORKFLOW: Strip whitespace and title-case consumer city values.
cleaned_dataset["Consumer_City"] = (
    cleaned_dataset["Consumer_City"]
    .str.strip()
    .str.title()
)

# WORKFLOW: Fill missing consumer cities with Not_Reported.
cleaned_dataset["Consumer_City"] = (
    cleaned_dataset["Consumer_City"]
    .fillna("Not_Reported")
)

# WORKFLOW: Strip whitespace and title-case consumer state values.
cleaned_dataset["Consumer_State"] = (
    cleaned_dataset["Consumer_State"]
    .str.strip()
    .str.title()
)

# WORKFLOW: Fill missing consumer states with Not_Reported.
cleaned_dataset["Consumer_State"] = (
    cleaned_dataset["Consumer_State"]
    .fillna("Not_Reported")
)

# WORKFLOW: Strip whitespace from complaint subjects.
cleaned_dataset["Subject"] = (
    cleaned_dataset["Subject"]
    .str.strip()
)

# WORKFLOW: Standardize robocall values using trimmed uppercase text.
cleaned_dataset["Recorded_Message_Or_Robocall"] = (
    cleaned_dataset["Recorded_Message_Or_Robocall"]
    .str.strip()
    .str.upper()
)

# WORKFLOW: Fill missing robocall values with Not_Reported.
cleaned_dataset["Recorded_Message_Or_Robocall"] = (
    cleaned_dataset["Recorded_Message_Or_Robocall"]
    .fillna("Not_Reported")
)

# WORKFLOW: Review standardized state and robocall value distributions.
for column in [
    "Consumer_State",
    "Recorded_Message_Or_Robocall"
]:
    print(f"\n{column}")
    print(cleaned_dataset[column].value_counts(dropna=False).head(20))

# WORKFLOW: Calculate reporting delay in days.
cleaned_dataset["Reporting_Delay_Days"] = (
    cleaned_dataset["Created_Date"]
    - cleaned_dataset["Violation_Date"]
).dt.days

# WORKFLOW: Describe the reporting-delay distribution.
print(
    cleaned_dataset["Reporting_Delay_Days"]
    .describe()
)

# WORKFLOW: Count negative reporting delays.
print(
    "Negative reporting delays:",
    (
        cleaned_dataset["Reporting_Delay_Days"] < 0
    ).sum()
)

# WORKFLOW: Check final dataset shape.
print(
    "Final dataset shape:",
    cleaned_dataset.shape
)

# WORKFLOW: Check remaining missing values.
print(
    cleaned_dataset.isnull().sum()
)

# WORKFLOW: Confirm exact duplicates are removed.
print(
    "Remaining exact duplicates:",
    cleaned_dataset.duplicated().sum()
)

# WORKFLOW: Prepare company-phone frequency counts for repeated-number analysis.
phone_counts = (
    cleaned_dataset.loc[
        cleaned_dataset["Company_Phone_Number"] != "Not_Reported",
        "Company_Phone_Number"
    ]
    .value_counts()
)

print(phone_counts.head(20))

# WORKFLOW: Inspect final data types.
print(cleaned_dataset.dtypes)

# WORKFLOW: Inspect final descriptive statistics.
cleaned_dataset.describe(include="all")

# WORKFLOW: Save the processed dataset to the processed-data path.
processed_path = PROCESSED_DIR / "DNC_September_2026_Processed.csv"
cleaned_dataset.to_csv(
    processed_path,
    index=False
)
print("Cleaned dataset saved successfully.")

# WORKFLOW: Reload the saved processed dataset and verify its shape.
processed_dataset = pd.read_csv(
    processed_path
)

print("Saved dataset shape:", processed_dataset.shape)
processed_dataset.head()
