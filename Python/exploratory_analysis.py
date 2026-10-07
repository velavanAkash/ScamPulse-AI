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

# WORKFLOW: Load the processed dataset used for EDA from the original processed-data path.
cleaned_dataset = pd.read_csv(
    PROCESSED_DIR / "DNC_September_2026_Processed.csv"
)

print("Shape:", cleaned_dataset.shape)
cleaned_dataset.head()

# WORKFLOW: List the processed dataset columns.
print("Columns:")
for column in cleaned_dataset.columns:
    print("-", column)

# WORKFLOW: Convert the EDA date fields to datetime.
cleaned_dataset["Created_Date"] = pd.to_datetime(
    cleaned_dataset["Created_Date"],
    errors="coerce"
)

cleaned_dataset["Violation_Date"] = pd.to_datetime(
    cleaned_dataset["Violation_Date"],
    errors="coerce"
)

# WORKFLOW: Confirm the datetime dtypes.
print(cleaned_dataset[["Created_Date", "Violation_Date"]].dtypes)

# WORKFLOW: Calculate the main dataset-level EDA metrics.
total_complaints = len(cleaned_dataset)

unique_subjects = cleaned_dataset["Subject"].nunique()

unique_states = cleaned_dataset[
    cleaned_dataset["Consumer_State"] != "Not_Reported"
]["Consumer_State"].nunique()

unique_cities = cleaned_dataset[
    cleaned_dataset["Consumer_City"] != "Not_Reported"
]["Consumer_City"].nunique()

reported_phone_numbers = cleaned_dataset[
    cleaned_dataset["Company_Phone_Number"] != "Not_Reported"
]["Company_Phone_Number"].nunique()

robocall_reports = (
    cleaned_dataset["Recorded_Message_Or_Robocall"] == "Y"
).sum()

print("Total Complaints:", total_complaints)
print("Unique Subjects:", unique_subjects)
print("Unique States:", unique_states)
print("Unique Cities:", unique_cities)
print("Unique Reported Phone Numbers:", reported_phone_numbers)
print("Robocall Reports:", robocall_reports)

# WORKFLOW: Aggregate daily complaint volume by Created_Date.
daily_complaints = (
    cleaned_dataset
    .groupby(cleaned_dataset["Created_Date"].dt.date)
    .size()
)
print(daily_complaints)

# WORKFLOW: Plot daily complaint volume.
plt.figure(figsize=(12, 5))
plt.plot(
    daily_complaints.index,
    daily_complaints.values,
    marker="o"
)
plt.title("Daily Complaint Volume")
plt.xlabel("Created Date")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# WORKFLOW: Calculate daily percentage changes in complaint volume.
daily_percentage_change = (
    daily_complaints
    .pct_change() * 100
)

daily_change_df = pd.DataFrame({
    "Complaint_Count": daily_complaints,
    "Percentage_Change": daily_percentage_change
})

daily_change_df

# WORKFLOW: Plot daily percentage change in complaints.
plt.figure(figsize=(12, 6))

plt.bar(
    daily_change_df.index,
    daily_change_df["Percentage_Change"]
)

plt.axhline(
    0,
    linewidth=1
)

plt.title("Daily Change in Complaint Volume")
plt.xlabel("Created Date")
plt.ylabel("Percentage Change (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# WORKFLOW: Plot daily complaints with a three-day rolling average.
rolling_3_day = daily_complaints.rolling(
    window=3
).mean()

plt.figure(figsize=(12, 6))

plt.plot(
    daily_complaints.index,
    daily_complaints.values,
    marker="o",
    alpha=0.5,
    label="Daily Complaints"
)

plt.plot(
    rolling_3_day.index,
    rolling_3_day.values,
    linewidth=2,
    label="3-Day Rolling Average"
)

plt.title("Daily Complaints and 3-Day Rolling Average")
plt.xlabel("Created Date")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# WORKFLOW: Extract complaint activity by hour of day.
cleaned_dataset["Created_Hour"] = (
    cleaned_dataset["Created_Date"].dt.hour
)

hourly_complaints = (
    cleaned_dataset["Created_Hour"]
    .value_counts()
    .sort_index()
)

print(hourly_complaints)

# WORKFLOW: Plot complaint activity by hour.
plt.figure(figsize=(12, 5))

plt.plot(
    hourly_complaints.index,
    hourly_complaints.values,
    marker="o"
)

plt.title("Complaint Activity by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Complaints")
plt.xticks(range(24))

plt.tight_layout()
plt.show()

# WORKFLOW: Calculate complaint volume by day of week.
day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

cleaned_dataset["Created_Day"] = (
    cleaned_dataset["Created_Date"].dt.day_name()
)

day_of_week = (
    cleaned_dataset["Created_Day"]
    .value_counts()
    .reindex(day_order)
)

print(day_of_week)

# WORKFLOW: Plot complaints by day of week.
plt.figure(figsize=(10, 5))

plt.bar(
    day_of_week.index,
    day_of_week.values
)

plt.title("Complaints by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

# WORKFLOW: Count complaints by subject.
subject_counts = (
    cleaned_dataset["Subject"]
    .value_counts()
)
print(subject_counts)

# WORKFLOW: Plot the top ten complaint subjects.
top_subjects = subject_counts.head(10).sort_values()

plt.figure(figsize=(12, 7))

plt.barh(
    top_subjects.index[::-1],
    top_subjects.values[::-1]
)

plt.title("Top 10 Complaint Subjects")
plt.xlabel("Number of Complaints")
plt.ylabel("Subject")

plt.tight_layout()
plt.show()

# WORKFLOW: Build subject counts and percentages.
subject_percentage = (
    subject_counts /
    len(cleaned_dataset) *
    100
)

subject_summary = pd.DataFrame({
    "Complaint_Count": subject_counts,
    "Percentage": subject_percentage.round(2)
})

subject_summary

# WORKFLOW: Plot the top complaint subjects by percentage.
plt.figure(figsize=(10, 6))

plt.barh(
    subject_percentage.head(10).index[::-1],
    subject_percentage.head(10).values[::-1]
)

plt.title("Top Complaint Subjects by Percentage")
plt.xlabel("Percentage of Complaints")
plt.ylabel("Subject")

plt.tight_layout()
plt.show()

# WORKFLOW: Count robocall / recorded-message statuses.
robocall_counts = (
    cleaned_dataset["Recorded_Message_Or_Robocall"]
    .value_counts()
    .reindex(
        ["Y", "N", "Not_Reported"],
        fill_value=0
    )
)

print(robocall_counts)

# WORKFLOW: Plot robocall / recorded-message status counts.
plt.figure(figsize=(8, 6))

plt.bar(
    robocall_counts.index,
    robocall_counts.values
)

plt.title("Robocall / Recorded Message Pattern")
plt.xlabel("Robocall Status")
plt.ylabel("Number of Complaints")

plt.tight_layout()
plt.show()

# WORKFLOW: Create a daily robocall-status cross-tabulation.
daily_robocall = pd.crosstab(
    cleaned_dataset["Created_Date"].dt.date,
    cleaned_dataset["Recorded_Message_Or_Robocall"]
)

daily_robocall = daily_robocall.reindex(
    columns=["Y", "N", "Not_Reported"],
    fill_value=0
)

daily_robocall.head()

# WORKFLOW: Plot daily robocall / recorded-message trends.
plt.figure(figsize=(12, 6))

for column in daily_robocall.columns:
    plt.plot(
        daily_robocall.index,
        daily_robocall[column],
        marker="o",
        label=column
    )

plt.title("Daily Robocall / Recorded Message Trend")
plt.xlabel("Created Date")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

# WORKFLOW: Cross-tab top subjects by robocall status.
top_10_subject_names = subject_counts.head(10).index

subject_robocall = pd.crosstab(
    cleaned_dataset["Subject"],
    cleaned_dataset["Recorded_Message_Or_Robocall"]
)

subject_robocall = subject_robocall.loc[
    top_10_subject_names
]

subject_robocall = subject_robocall.reindex(
    columns=["Y", "N", "Not_Reported"],
    fill_value=0
)

subject_robocall

# WORKFLOW: Plot top subjects by robocall status.
subject_robocall.plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title("Top Complaint Subjects by Robocall Status")
plt.xlabel("Complaint Subject")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Robocall")

plt.tight_layout()
plt.show()

# WORKFLOW: Count complaint volume by state.
state_counts = (
    cleaned_dataset[
        cleaned_dataset["Consumer_State"] != "Not_Reported"
    ]["Consumer_State"]
    .value_counts()
)

print(state_counts.head(20))

# WORKFLOW: Plot the top states by complaint volume.
top_states = state_counts.head(15).sort_values()

plt.figure(figsize=(12, 8))

plt.barh(
    top_states.index,
    top_states.values
)

plt.title("Top 15 States by Complaint Volume")
plt.xlabel("Number of Complaints")
plt.ylabel("State")

plt.tight_layout()
plt.show()

# WORKFLOW: Cross-tab top states by robocall status.
top_state_names = state_counts.head(10).index

state_robocall = pd.crosstab(
    cleaned_dataset["Consumer_State"],
    cleaned_dataset["Recorded_Message_Or_Robocall"]
)

state_robocall = state_robocall.loc[
    top_state_names
]

state_robocall = state_robocall.reindex(
    columns=["Y", "N", "Not_Reported"],
    fill_value=0
)

state_robocall

# WORKFLOW: Plot top states by robocall status.
state_robocall.plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title("Top States by Robocall Status")
plt.xlabel("State")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# WORKFLOW: Count complaint volume by city.
city_counts = (
    cleaned_dataset[
        cleaned_dataset["Consumer_City"] != "Not_Reported"
    ]["Consumer_City"]
    .value_counts()
)

print(city_counts.head(15))

# WORKFLOW: Plot the top cities by complaint volume.
top_cities = city_counts.head(15).sort_values()

plt.figure(figsize=(12, 8))

plt.barh(
    top_cities.index,
    top_cities.values
)

plt.title("Top 15 Cities by Complaint Volume")
plt.xlabel("Number of Complaints")
plt.ylabel("City")

plt.tight_layout()
plt.show()

# WORKFLOW: Count complaint volume by consumer area code.
area_code_counts = (
    cleaned_dataset[
        cleaned_dataset["Consumer_Area_Code"] != "Not_Reported"
    ]["Consumer_Area_Code"]
    .value_counts()
)

print(area_code_counts.head(15))

# WORKFLOW: Plot the top consumer area codes.
top_area_codes = area_code_counts.head(15).sort_values()

plt.figure(figsize=(12, 7))

plt.barh(
    top_area_codes.index.astype(str),
    top_area_codes.values
)

plt.title("Top 15 Consumer Area Codes")
plt.xlabel("Number of Complaints")
plt.ylabel("Area Code")

plt.tight_layout()
plt.show()

# WORKFLOW: Count frequently reported company phone numbers.
phone_counts = (
    cleaned_dataset[
        cleaned_dataset["Company_Phone_Number"] != "Not_Reported"
    ]["Company_Phone_Number"]
    .value_counts()
)

print("Distinct Reported Phone Numbers:", len(phone_counts))

print(
    "Numbers Reported More Than Once:",
    (phone_counts > 1).sum()
)

print("\nTop Repeated Numbers:")
print(phone_counts.head(15))

# WORKFLOW: Plot the most frequently reported originating numbers.
top_phone_numbers = phone_counts.head(15).sort_values()

plt.figure(figsize=(12, 8))

plt.barh(
    top_phone_numbers.index.astype(str),
    top_phone_numbers.values
)

plt.title("Top 15 Frequently Reported Originating Numbers")
plt.xlabel("Number of Complaint Reports")
plt.ylabel("Phone Number")

plt.tight_layout()
plt.show()

# WORKFLOW: Analyze repeated phone numbers by complaint subject.
repeated_phones = phone_counts[
    phone_counts > 1
].head(15).index

phone_subject_analysis = (
    cleaned_dataset[
        cleaned_dataset["Company_Phone_Number"].isin(repeated_phones)
    ]
    .groupby(
        ["Company_Phone_Number", "Subject"]
    )
    .size()
    .reset_index(name="Complaint_Count")
    .sort_values(
        "Complaint_Count",
        ascending=False
    )
)

phone_subject_analysis.head(30)

# WORKFLOW: Cross-tab top states and top complaint subjects.
top_5_subject_names = subject_counts.head(5).index
top_10_state_names = state_counts.head(10).index

state_subject = pd.crosstab(
    cleaned_dataset["Consumer_State"],
    cleaned_dataset["Subject"]
)

state_subject = state_subject.loc[
    top_10_state_names,
    top_5_subject_names
]

state_subject

# WORKFLOW: Plot top subjects across top states.
state_subject.plot(
    kind="bar",
    stacked=True,
    figsize=(14, 7)
)

plt.title("Top Complaint Subjects Across Top States")
plt.xlabel("State")
plt.ylabel("Number of Complaints")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# WORKFLOW: Summarize negative reporting delays and invalid date-order records.
negative_delay_count = (
    cleaned_dataset["Reporting_Delay_Days"] < 0
).sum()

invalid_date_count = (
    cleaned_dataset["Invalid_Date_Order"] == True
).sum()

print("Negative Reporting Delays:", negative_delay_count)
print("Invalid Date Order Records:", invalid_date_count)

# WORKFLOW: Select records with non-negative reporting delays.
valid_delay = cleaned_dataset[
    cleaned_dataset["Reporting_Delay_Days"] >= 0
].copy()

print(
    valid_delay["Reporting_Delay_Days"].describe()
)

# WORKFLOW: Plot the distribution of valid reporting delays.
plt.figure(figsize=(10, 6))

plt.hist(
    valid_delay["Reporting_Delay_Days"],
    bins=30
)

plt.title("Distribution of Valid Reporting Delay")
plt.xlabel("Reporting Delay (Days)")
plt.ylabel("Number of Complaints")

plt.tight_layout()
plt.show()

# WORKFLOW: Summarize reporting-delay statistics.
delay_summary = pd.DataFrame({
    "Metric": [
        "Mean",
        "Median",
        "Minimum",
        "Maximum"
    ],
    "Days": [
        valid_delay["Reporting_Delay_Days"].mean(),
        valid_delay["Reporting_Delay_Days"].median(),
        valid_delay["Reporting_Delay_Days"].min(),
        valid_delay["Reporting_Delay_Days"].max()
    ]
})

delay_summary

# WORKFLOW: Define the previous and recent three-date periods.
unique_dates = sorted(
    cleaned_dataset["Created_Date"]
    .dt.normalize()
    .dropna()
    .unique()
)

previous_period = unique_dates[-6:-3]
recent_period = unique_dates[-3:]

print("Previous Period:")
print(previous_period)

print("\nRecent Period:")
print(recent_period)

# WORKFLOW: Split the processed dataset into previous and recent periods.
previous_data = cleaned_dataset[
    cleaned_dataset["Created_Date"]
    .dt.normalize()
    .isin(previous_period)
]

recent_data = cleaned_dataset[
    cleaned_dataset["Created_Date"]
    .dt.normalize()
    .isin(recent_period)
]

# WORKFLOW: Calculate emerging complaint subjects by period-over-period change.
previous_subject = previous_data["Subject"].value_counts()

recent_subject = recent_data["Subject"].value_counts()

emerging_subjects = pd.DataFrame({
    "Previous_Count": previous_subject,
    "Recent_Count": recent_subject
}).fillna(0)

emerging_subjects["Absolute_Change"] = (
    emerging_subjects["Recent_Count"]
    - emerging_subjects["Previous_Count"]
)

emerging_subjects["Percentage_Change"] = np.where(
    emerging_subjects["Previous_Count"] > 0,
    (
        emerging_subjects["Absolute_Change"]
        / emerging_subjects["Previous_Count"]
    ) * 100,
    np.nan
)

emerging_subjects = emerging_subjects[
    (emerging_subjects["Previous_Count"] >= 50) &
    (emerging_subjects["Recent_Count"] >= 50)
]

emerging_subjects = emerging_subjects.sort_values(
    "Absolute_Change",
    ascending=False
)

emerging_subjects

# WORKFLOW: Calculate emerging states by period-over-period change.
previous_state = previous_data[
    previous_data["Consumer_State"] != "Not_Reported"
]["Consumer_State"].value_counts()

recent_state = recent_data[
    recent_data["Consumer_State"] != "Not_Reported"
]["Consumer_State"].value_counts()

emerging_states = pd.DataFrame({
    "Previous_Count": previous_state,
    "Recent_Count": recent_state
}).fillna(0)

emerging_states["Absolute_Change"] = (
    emerging_states["Recent_Count"]
    - emerging_states["Previous_Count"]
)

emerging_states["Percentage_Change"] = np.where(
    emerging_states["Previous_Count"] > 0,
    (
        emerging_states["Absolute_Change"]
        / emerging_states["Previous_Count"]
    ) * 100,
    np.nan
)

emerging_states = emerging_states[
    (emerging_states["Previous_Count"] >= 50) &
    (emerging_states["Recent_Count"] >= 50)
]

emerging_states = emerging_states.sort_values(
    "Absolute_Change",
    ascending=False
)

emerging_states.head(15)

# WORKFLOW: Calculate emerging reported phone numbers by period-over-period change.
previous_phone = previous_data[
    previous_data["Company_Phone_Number"] != "Not_Reported"
]["Company_Phone_Number"].value_counts()

recent_phone = recent_data[
    recent_data["Company_Phone_Number"] != "Not_Reported"
]["Company_Phone_Number"].value_counts()

emerging_phones = pd.DataFrame({
    "Previous_Count": previous_phone,
    "Recent_Count": recent_phone
}).fillna(0)

emerging_phones["Absolute_Change"] = (
    emerging_phones["Recent_Count"]
    - emerging_phones["Previous_Count"]
)

emerging_phones["Percentage_Change"] = np.where(
    emerging_phones["Previous_Count"] > 0,
    (
        emerging_phones["Absolute_Change"]
        / emerging_phones["Previous_Count"]
    ) * 100,
    np.nan
)

emerging_phones = emerging_phones[
    (emerging_phones["Recent_Count"] >= 5)
]

emerging_phones = emerging_phones.sort_values(
    ["Absolute_Change", "Recent_Count"],
    ascending=False
)

emerging_phones.head(20)

# WORKFLOW: Summarize state × subject × robocall combinations.
pattern_analysis = (
    cleaned_dataset
    .groupby(
        [
            "Consumer_State",
            "Subject",
            "Recorded_Message_Or_Robocall"
        ]
    )
    .size()
    .reset_index(name="Complaint_Count")
    .sort_values(
        "Complaint_Count",
        ascending=False
    )
)

pattern_analysis.head(30)

# WORKFLOW: Build the final KPI summary table.
kpi_summary = pd.DataFrame({
    "Metric": [
        "Total Complaints",
        "Unique Subjects",
        "Unique States",
        "Unique Cities",
        "Distinct Reported Phone Numbers",
        "Robocall Reports",
        "Repeated Phone Numbers"
    ],
    "Value": [
        len(cleaned_dataset),
        cleaned_dataset["Subject"].nunique(),
        cleaned_dataset[
            cleaned_dataset["Consumer_State"] != "Not_Reported"
        ]["Consumer_State"].nunique(),
        cleaned_dataset[
            cleaned_dataset["Consumer_City"] != "Not_Reported"
        ]["Consumer_City"].nunique(),
        len(phone_counts),
        (cleaned_dataset["Recorded_Message_Or_Robocall"] == "Y").sum(),
        (phone_counts > 1).sum()
    ]
})

kpi_summary
