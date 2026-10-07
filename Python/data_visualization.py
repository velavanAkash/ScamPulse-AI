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

import textwrap
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# WORKFLOW: Load the final processed dataset for dashboard visualization.
processed_path = PROCESSED_DIR / "DNC_September_2026_Processed.csv"
cleaned_dataset = pd.read_csv(processed_path)
cleaned_dataset["Created_Date"] = pd.to_datetime(cleaned_dataset["Created_Date"], errors="coerce")
cleaned_dataset["Violation_Date"] = pd.to_datetime(cleaned_dataset["Violation_Date"], errors="coerce")
print("Processed dataset shape:", cleaned_dataset.shape)

# WORKFLOW: Prepare dashboard KPIs, counts, geography summaries, phone frequencies, and emerging-pattern tables.

# Dashboard aggregations

total_complaints = len(cleaned_dataset)

robocall_reports = int(
    (cleaned_dataset["Recorded_Message_Or_Robocall"] == "Y").sum()
)

reported_phone_numbers = int(
    cleaned_dataset.loc[
        cleaned_dataset["Company_Phone_Number"] != "Not_Reported",
        "Company_Phone_Number"
    ].nunique()
)

repeated_numbers = int(
    (
        cleaned_dataset.loc[
            cleaned_dataset["Company_Phone_Number"] != "Not_Reported",
            "Company_Phone_Number"
        ]
        .value_counts() > 1
    ).sum()
)

unique_subjects = int(
    cleaned_dataset["Subject"].nunique()
)

unique_states = int(
    cleaned_dataset.loc[
        cleaned_dataset["Consumer_State"] != "Not_Reported",
        "Consumer_State"
    ].nunique()
)

daily = (
    cleaned_dataset
    .groupby(cleaned_dataset["Created_Date"].dt.date)
    .size()
    .sort_index()
)

subject_counts = cleaned_dataset["Subject"].value_counts()

robocall_counts = (
    cleaned_dataset["Recorded_Message_Or_Robocall"]
    .value_counts()
    .reindex(["Y", "N", "Not_Reported"], fill_value=0)
)

state_counts = (
    cleaned_dataset.loc[
        cleaned_dataset["Consumer_State"] != "Not_Reported",
        "Consumer_State"
    ]
    .value_counts()
)

city_counts = (
    cleaned_dataset.loc[
        cleaned_dataset["Consumer_City"] != "Not_Reported",
        "Consumer_City"
    ]
    .value_counts()
)

area_code_counts = (
    cleaned_dataset.loc[
        cleaned_dataset["Consumer_Area_Code"] != "Not_Reported",
        "Consumer_Area_Code"
    ]
    .value_counts()
)

phone_counts = (
    cleaned_dataset.loc[
        cleaned_dataset["Company_Phone_Number"] != "Not_Reported",
        "Company_Phone_Number"
    ]
    .value_counts()
)

# Emerging patterns: last 3 available Created_Date dates vs previous 3.
dates = sorted(
    cleaned_dataset["Created_Date"]
    .dt.normalize()
    .dropna()
    .unique()
)

if len(dates) >= 6:
    previous_period = dates[-6:-3]
    recent_period = dates[-3:]
else:
    split = max(1, len(dates) // 2)
    previous_period = dates[:split]
    recent_period = dates[split:]

previous_data = cleaned_dataset[
    cleaned_dataset["Created_Date"].dt.normalize().isin(previous_period)
]

recent_data = cleaned_dataset[
    cleaned_dataset["Created_Date"].dt.normalize().isin(recent_period)
]

def emerging_table(column, exclude_value="Not_Reported",
                   min_previous=0, min_recent=1):
    prev = previous_data.loc[
        previous_data[column] != exclude_value, column
    ].value_counts()

    recent = recent_data.loc[
        recent_data[column] != exclude_value, column
    ].value_counts()

    out = pd.DataFrame({
        "Previous": prev,
        "Recent": recent
    }).fillna(0)

    out["Change"] = out["Recent"] - out["Previous"]

    out["Change_Pct"] = np.where(
        out["Previous"] > 0,
        (out["Change"] / out["Previous"]) * 100,
        np.nan
    )

    out = out[
        (out["Previous"] >= min_previous) &
        (out["Recent"] >= min_recent)
    ]

    return out.sort_values(
        ["Change", "Recent"],
        ascending=False
    )

emerging_subjects = emerging_table(
    "Subject",
    min_previous=50,
    min_recent=50
)

emerging_states = emerging_table(
    "Consumer_State",
    min_previous=50,
    min_recent=50
)

emerging_phones = emerging_table(
    "Company_Phone_Number",
    min_recent=5
)

print("Dashboard data prepared.")

# WORKFLOW: Define reusable Plotly helper functions for labels and layout.

# Helper functions for a clean dashboard

def short_label(value, width=24):
    return textwrap.shorten(
        str(value),
        width=width,
        placeholder="..."
    )

def base_layout(fig, height=420):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=70, r=30, t=55, b=55),
        font=dict(size=11),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(align="left")
    )
    return fig

def apply_horizontal_style(fig):
    fig.update_xaxes(showgrid=True, automargin=True)
    fig.update_yaxes(
        showgrid=False,
        automargin=True,
        tickfont=dict(size=10)
    )
    return fig

# WORKFLOW: Build the Plotly figures for daily complaints, subjects, robocalls, geography, reported numbers, and emerging patterns.

# Build Plotly figures

# 1. Daily complaints
fig_daily = go.Figure(
    go.Scatter(
        x=list(daily.index),
        y=list(daily.values),
        mode="lines+markers",
        hovertemplate=(
            "Date: %{x}<br>"
            "Complaints: %{y:,}<extra></extra>"
        ),
        name="Complaints"
    )
)

fig_daily.update_layout(
    title="Daily Complaint Volume",
    xaxis=dict(
        title="Created Date",
        rangeslider=dict(visible=True),
        automargin=True
    ),
    yaxis=dict(
        title="Complaints",
        automargin=True
    )
)
base_layout(fig_daily, 450)


# 2. Top subjects
top_subjects = subject_counts.head(10).sort_values()

fig_subjects = go.Figure(
    go.Bar(
        x=top_subjects.values,
        y=[short_label(x, 24) for x in top_subjects.index],
        customdata=top_subjects.index,
        orientation="h",
        hovertemplate=(
            "Subject: %{customdata}<br>"
            "Complaints: %{x:,}<extra></extra>"
        )
    )
)

fig_subjects.update_layout(
    title="Top Complaint Subjects",
    xaxis_title="Complaints"
)

base_layout(fig_subjects)
apply_horizontal_style(fig_subjects)
fig_subjects.update_yaxes(autorange="reversed")


# 3. Robocall
fig_robocall = go.Figure(
    go.Pie(
        labels=["Y", "N", "Not Reported"],
        values=robocall_counts.values,
        hole=0.50,
        textinfo="label+percent",
        hovertemplate=(
            "Status: %{label}<br>"
            "Complaints: %{value:,}<br>"
            "Share: %{percent}<extra></extra>"
        )
    )
)

fig_robocall.update_layout(
    title="Robocall Status",
    showlegend=False
)

base_layout(fig_robocall)


# 4. Geography with a simple dropdown
top_states = state_counts.head(10).sort_values()
top_cities = city_counts.head(10).sort_values()
top_area_codes = area_code_counts.head(10).sort_values()

fig_geo = go.Figure()

fig_geo.add_trace(
    go.Bar(
        x=top_states.values,
        y=top_states.index,
        orientation="h",
        visible=True,
        name="State",
        hovertemplate=(
            "State: %{y}<br>"
            "Complaints: %{x:,}<extra></extra>"
        )
    )
)

fig_geo.add_trace(
    go.Bar(
        x=top_cities.values,
        y=top_cities.index,
        orientation="h",
        visible=False,
        name="City",
        hovertemplate=(
            "City: %{y}<br>"
            "Complaints: %{x:,}<extra></extra>"
        )
    )
)

fig_geo.add_trace(
    go.Bar(
        x=top_area_codes.values,
        y=[str(x) for x in top_area_codes.index],
        orientation="h",
        visible=False,
        name="Area Code",
        hovertemplate=(
            "Area Code: %{y}<br>"
            "Complaints: %{x:,}<extra></extra>"
        )
    )
)

fig_geo.update_layout(
    title="Geographic Concentration",
    xaxis_title="Complaints",
    updatemenus=[
        dict(
            type="dropdown",
            direction="down",
            x=1,
            xanchor="right",
            y=1.16,
            buttons=[
                dict(
                    label="State",
                    method="update",
                    args=[
                        {"visible": [True, False, False]},
                        {"title": "Top States"}
                    ]
                ),
                dict(
                    label="City",
                    method="update",
                    args=[
                        {"visible": [False, True, False]},
                        {"title": "Top Cities"}
                    ]
                ),
                dict(
                    label="Area Code",
                    method="update",
                    args=[
                        {"visible": [False, False, True]},
                        {"title": "Top Area Codes"}
                    ]
                )
            ]
        )
    ]
)

base_layout(fig_geo)
apply_horizontal_style(fig_geo)
fig_geo.update_yaxes(autorange="reversed")


# 5. Frequently reported numbers
top_phones = phone_counts.head(10).sort_values()

fig_phones = go.Figure(
    go.Bar(
        x=top_phones.values,
        y=[str(x) for x in top_phones.index],
        orientation="h",
        hovertemplate=(
            "Phone: %{y}<br>"
            "Reports: %{x:,}<extra></extra>"
        )
    )
)

fig_phones.update_layout(
    title="Frequently Reported Numbers",
    xaxis_title="Complaint Reports"
)

base_layout(fig_phones)
apply_horizontal_style(fig_phones)
fig_phones.update_yaxes(autorange="reversed")


# 6. Emerging subjects
es = emerging_subjects.head(8).sort_values("Change")

fig_es = go.Figure(
    go.Bar(
        x=es["Change"],
        y=[short_label(x, 24) for x in es.index],
        customdata=np.array([
            [
                str(x),
                int(es.loc[x, "Previous"]),
                int(es.loc[x, "Recent"])
            ]
            for x in es.index
        ], dtype=object),
        orientation="h",
        hovertemplate=(
            "Subject: %{customdata[0]}<br>"
            "Previous: %{customdata[1]:,}<br>"
            "Recent: %{customdata[2]:,}<br>"
            "Change: %{x:,}<extra></extra>"
        )
    )
)

fig_es.update_layout(
    title="Emerging Subjects",
    xaxis_title="Change in Complaints"
)

base_layout(fig_es)
apply_horizontal_style(fig_es)
fig_es.update_yaxes(autorange="reversed")


# 7. Emerging states
est = emerging_states.head(8).sort_values("Change")

fig_est = go.Figure(
    go.Bar(
        x=est["Change"],
        y=est.index,
        orientation="h",
        hovertemplate=(
            "State: %{y}<br>"
            "Change: %{x:,}<extra></extra>"
        )
    )
)

fig_est.update_layout(
    title="Emerging States",
    xaxis_title="Change in Complaints"
)

base_layout(fig_est)
apply_horizontal_style(fig_est)
fig_est.update_yaxes(autorange="reversed")


# 8. Emerging phone numbers
ep = emerging_phones.head(10).sort_values("Change")

fig_ep = go.Figure(
    go.Bar(
        x=ep["Change"],
        y=[str(x) for x in ep.index],
        customdata=np.array([
            [
                str(x),
                int(ep.loc[x, "Previous"]),
                int(ep.loc[x, "Recent"])
            ]
            for x in ep.index
        ], dtype=object),
        orientation="h",
        hovertemplate=(
            "Phone: %{customdata[0]}<br>"
            "Previous: %{customdata[1]:,}<br>"
            "Recent: %{customdata[2]:,}<br>"
            "Change: %{x:,}<extra></extra>"
        )
    )
)

fig_ep.update_layout(
    title="Emerging Phone Numbers",
    xaxis_title="Change in Complaints"
)

base_layout(fig_ep)
apply_horizontal_style(fig_ep)
fig_ep.update_yaxes(autorange="reversed")

print("Plotly figures created.")

# WORKFLOW: Export the responsive non-overlapping Plotly dashboard to the original visualization path.

# Export a clean, non-overlapping dashboard page

figures = [
    ("daily", fig_daily),
    ("subjects", fig_subjects),
    ("robocall", fig_robocall),
    ("geo", fig_geo),
    ("phones", fig_phones),
    ("emerging-subjects", fig_es),
    ("emerging-states", fig_est),
    ("emerging-phones", fig_ep),
]

kpi_cards = f"""
<div class="kpi-grid">
  <div class="kpi"><div class="label">Total Complaints</div><div class="value">{total_complaints:,}</div></div>
  <div class="kpi"><div class="label">Robocall Reports</div><div class="value">{robocall_reports:,}</div></div>
  <div class="kpi"><div class="label">Reported Numbers</div><div class="value">{reported_phone_numbers:,}</div></div>
  <div class="kpi"><div class="label">Repeated Numbers</div><div class="value">{repeated_numbers:,}</div></div>
  <div class="kpi"><div class="label">Unique Subjects</div><div class="value">{unique_subjects:,}</div></div>
  <div class="kpi"><div class="label">States / Geographies</div><div class="value">{unique_states:,}</div></div>
</div>
"""

chart_blocks = []
for name, figure in figures:
    chart_blocks.append(
        f"""
        <section class="card {name}">
            {pio.to_html(
                figure,
                include_plotlyjs=False,
                full_html=False,
                config={
                    "responsive": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "lasso2d",
                        "select2d"
                    ]
                }
            )}
        </section>
        """
    )

dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ScamPulse AI Dashboard</title>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 24px;
    background: #f4f6f8;
    font-family: Arial, Helvetica, sans-serif;
}}

.dashboard {{
    max-width: 1500px;
    margin: 0 auto;
}}

.header {{
    text-align: center;
    margin-bottom: 24px;
}}

.header h1 {{
    margin: 0;
    font-size: 30px;
}}

.header p {{
    margin: 6px 0 0;
    font-size: 14px;
}}

.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 14px;
    margin-bottom: 18px;
}}

.kpi {{
    background: white;
    border-radius: 10px;
    padding: 18px 12px;
    text-align: center;
    min-height: 100px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}

.kpi .label {{
    font-size: 12px;
    margin-bottom: 10px;
}}

.kpi .value {{
    font-size: 25px;
    font-weight: 700;
}}

.card {{
    background: white;
    border-radius: 10px;
    padding: 8px;
    margin-bottom: 16px;
    overflow: hidden;
}}

.daily {{
    width: 100%;
}}

.subjects,
.robocall,
.geo,
.phones {{
    display: inline-block;
    vertical-align: top;
    width: calc(50% - 10px);
}}

.subjects {{
    margin-right: 16px;
}}

.emerging-subjects,
.emerging-states,
.emerging-phones {{
    display: inline-block;
    vertical-align: top;
    width: calc(33.333% - 12px);
}}

.emerging-subjects,
.emerging-states {{
    margin-right: 12px;
}}

@media (max-width: 1100px) {{
    .kpi-grid {{
        grid-template-columns: repeat(3, 1fr);
    }}

    .subjects,
    .robocall,
    .geo,
    .phones {{
        width: 100%;
        margin-right: 0;
    }}

    .emerging-subjects,
    .emerging-states,
    .emerging-phones {{
        width: 100%;
        margin-right: 0;
    }}
}}

@media (max-width: 650px) {{
    body {{
        padding: 10px;
    }}

    .kpi-grid {{
        grid-template-columns: repeat(2, 1fr);
    }}

    .header h1 {{
        font-size: 24px;
    }}
}}
</style>
</head>

<body>
<div class="dashboard">

    <div class="header">
        <h1>SCAMPULSE AI</h1>
        <p>Telecom Scam &amp; Spam Pattern Intelligence</p>
    </div>

    {kpi_cards}

    {"".join(chart_blocks)}

</div>
</body>
</html>
"""

dashboard_output = VISUALIZATIONS_DIR / "ScamPulse_AI_Dashboard.html"

dashboard_output.parent.mkdir(
    parents=True,
    exist_ok=True
)

dashboard_output.write_text(
    dashboard_html,
    encoding="utf-8"
)

print(f"Dashboard saved to: {dashboard_output.resolve()}")
