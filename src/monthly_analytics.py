"""Monthly temporal analytics and intelligence module.
Enables side-by-side monthly comparisons, volatility assessments, and diurnal profile extraction.
"""

from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import MONTHLY_REPORTS_DIR, METRICS_DIR


@st.cache_data
def load_monthly_summary_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the 24-month summary table and month-hour heatmap matrix."""
    summary_path = MONTHLY_REPORTS_DIR / "monthly_summary.csv"
    heatmap_path = MONTHLY_REPORTS_DIR / "month_hour_heatmap_matrix.csv"

    summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    heatmap_df = pd.read_csv(heatmap_path, index_col=0) if heatmap_path.exists() else pd.DataFrame()

    return summary_df, heatmap_df


def compare_two_months(m1_name: str, m2_name: str, summary_df: pd.DataFrame) -> dict:
    """Computes operational differentials between two selected calendar months."""
    row1 = summary_df[summary_df["year_month"] == m1_name].iloc[0]
    row2 = summary_df[summary_df["year_month"] == m2_name].iloc[0]

    tot_diff = int(row2["total_rentals"] - row1["total_rentals"])
    pct_diff = round((tot_diff / (row1["total_rentals"] + 1e-6)) * 100, 1)

    return {
        "month_1": row1.to_dict(),
        "month_2": row2.to_dict(),
        "total_rentals_diff": tot_diff,
        "percentage_growth": pct_diff,
        "mean_hourly_diff": round(float(row2["mean_hourly_demand"] - row1["mean_hourly_demand"]), 1),
        "registered_pct_diff": round(float(row2["registered_percentage"] - row1["registered_percentage"]), 1)
    }


def plot_monthly_comparison_bars(m1_dict: dict, m2_dict: dict) -> go.Figure:
    """Renders a comparative bar chart between two selected months."""
    categories = ["Total Rentals (k)", "Mean Hourly", "Peak Hourly", "Weekday Demand", "Weekend Demand"]
    val1 = [m1_dict["total_rentals"] / 1000.0, m1_dict["mean_hourly_demand"], m1_dict["peak_hourly_demand"], m1_dict["weekday_avg_hourly"], m1_dict["weekend_avg_hourly"]]
    val2 = [m2_dict["total_rentals"] / 1000.0, m2_dict["mean_hourly_demand"], m2_dict["peak_hourly_demand"], m2_dict["weekday_avg_hourly"], m2_dict["weekend_avg_hourly"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=m1_dict["year_month"], x=categories, y=val1, marker_color="#0f52ba"))
    fig.add_trace(go.Bar(name=m2_dict["year_month"], x=categories, y=val2, marker_color="#10b981"))

    fig.update_layout(
        title=f"Head-to-Head Comparison: {m1_dict['year_month']} vs. {m2_dict['year_month']}",
        barmode="group",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig
