"""Modular Plotly visual analytics library for the Streamlit dashboard.
Provides responsive, high-fidelity business intelligence charts.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import ALL_PEAK_HOURS, TARGET_COL, CASUAL_COL, REGISTERED_COL


def plot_hourly_time_series(pred_df: pd.DataFrame, selected_model: str, show_p90: bool = False, q_df: pd.DataFrame = None) -> go.Figure:
    """Renders interactive actual vs. predicted demand time series with rangeslider."""
    fig = go.Figure()
    # Actual demand
    fig.add_trace(go.Scatter(
        x=pred_df.index,
        y=pred_df["actual"],
        name="Actual Rentals",
        line=dict(color="#1e293b", width=2.2),
        opacity=0.9
    ))
    # Model prediction
    if selected_model in pred_df.columns:
        fig.add_trace(go.Scatter(
            x=pred_df.index,
            y=pred_df[selected_model],
            name=f"Forecast ({selected_model})",
            line=dict(color="#0f52ba", width=2, dash="dash"),
            opacity=0.95
        ))
    # P90 safety buffer
    if show_p90 and q_df is not None and "p90_conservative_calibrated" in q_df.columns:
        sub_q = q_df.reindex(pred_df.index)
        fig.add_trace(go.Scatter(
            x=sub_q.index,
            y=sub_q["p90_conservative_calibrated"],
            name="Calibrated P90 Buffer (95% Coverage)",
            line=dict(color="#10b981", width=1.5, dash="dot"),
            opacity=0.85
        ))

    fig.update_layout(
        title="Hourly Completed Rentals: Actual vs. Forecast",
        xaxis_title="Timeline",
        yaxis_title="Hourly Rentals",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig


def plot_hour_weekday_heatmap(df: pd.DataFrame) -> go.Figure:
    """Renders a 24-hour by 7-day demand intensity heatmap."""
    pivot = df.pivot_table(index="weekday", columns="hr", values=TARGET_COL, aggfunc="mean").round(1)
    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    pivot.index = [day_names[i] for i in pivot.index]

    fig = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Mean Rentals"),
        x=[f"{h:02d}:00" for h in range(24)],
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    fig.update_layout(
        title="Demand Concentration Matrix: Hour of Day vs. Day of Week",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig


def plot_weekday_weekend_curves(df: pd.DataFrame) -> go.Figure:
    """Visualizes the bimodal commuter profile vs the unimodal leisure profile."""
    hourly_wd = df[df["workingday"] == 1].groupby("hr")[TARGET_COL].mean()
    hourly_we = df[df["workingday"] == 0].groupby("hr")[TARGET_COL].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_wd.index,
        y=hourly_wd.values,
        name="Working Days (Commuters)",
        line=dict(color="#0f52ba", width=3),
        mode="lines+markers"
    ))
    fig.add_trace(go.Scatter(
        x=hourly_we.index,
        y=hourly_we.values,
        name="Non-Working Days (Leisure)",
        line=dict(color="#f59e0b", width=3),
        mode="lines+markers"
    ))

    # Highlight commute windows
    fig.add_vrect(x0=6.5, x1=9.5, fillcolor="#0f52ba", opacity=0.1, line_width=0, annotation_text="Morning Rush", annotation_position="top left")
    fig.add_vrect(x0=16.5, x1=19.5, fillcolor="#0f52ba", opacity=0.1, line_width=0, annotation_text="Evening Rush", annotation_position="top left")

    fig.update_layout(
        title="Diurnal Demand Curves: Working Days vs. Weekends / Holidays",
        xaxis_title="Hour of Day",
        yaxis_title="Average Hourly Rentals",
        template="plotly_white",
        height=400,
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig


def plot_rider_composition(pareto_data: dict) -> go.Figure:
    """Renders user segmentation pie chart."""
    r_comp = pareto_data.get("rider_composition", {})
    fig = px.pie(
        values=[r_comp.get("registered_rentals", 2672780), r_comp.get("casual_rentals", 619899)],
        names=["Registered (Commuters)", "Casual (Tourists)"],
        color_discrete_sequence=["#0f52ba", "#f59e0b"],
        hole=0.45,
        title="Rider Mix: Registered Subscribers vs Casual Users"
    )
    fig.update_layout(template="plotly_white", height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_residuals(actual: pd.Series, predicted: pd.Series) -> tuple[go.Figure, go.Figure]:
    """Generates residual histogram and residual vs actual scatter plot."""
    residuals = predicted - actual
    
    # Histogram
    fig_hist = px.histogram(
        residuals,
        nbins=60,
        title="Error Distribution (Predicted - Actual)",
        labels=dict(value="Residual Error (Rentals/Hour)"),
        color_discrete_sequence=["#0f52ba"]
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#ef4444")
    fig_hist.update_layout(template="plotly_white", height=380)

    # Scatter
    sample_df = pd.DataFrame({"actual": actual, "residual": residuals}).sample(n=min(1200, len(actual)), random_state=42)
    fig_scat = px.scatter(
        sample_df,
        x="actual",
        y="residual",
        title="Residual Error vs Actual Demand",
        labels=dict(actual="Actual Rentals", residual="Residual Error"),
        opacity=0.6,
        color_discrete_sequence=["#f59e0b"]
    )
    fig_scat.add_hline(y=0, line_dash="dash", line_color="#ef4444")
    fig_scat.update_layout(template="plotly_white", height=380)

    return fig_hist, fig_scat
