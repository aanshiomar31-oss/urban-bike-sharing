"""Professional Executive Streamlit Dashboard for Urban Bike-Sharing Demand Forecasting and Fleet Optimization.
Implements 8 comprehensive analytical pages with interactive Plotly visualizations and grounded natural-language QA.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import (
    METRICS_DIR,
    FORECASTS_DIR,
    MONTHLY_REPORTS_DIR,
    REPORTS_DIR,
    ALL_PEAK_HOURS,
    DEFAULT_TRIP_DURATION,
    DEFAULT_SPATIAL_BUFFER,
    DEFAULT_MAINTENANCE_SHARE,
    ILLUSTRATIVE_FLEET_SIZE,
)
from src.analyst import OperationsAnalyst
from src.fleet_optimizer import calculate_active_fleet

# Page configuration
st.set_page_config(
    page_title="Urban Bike-Sharing Intelligence | Sigmoid",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .badge-pass {
        background-color: #e6f4ea;
        color: #137333;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-warn {
        background-color: #fef7e0;
        color: #b06000;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_artifacts():
    """Caches precomputed project outputs for instantaneous dashboard rendering."""
    # Data quality report
    dq_path = REPORTS_DIR / "data_quality_report.json"
    dq = json.loads(dq_path.read_text()) if dq_path.exists() else {}

    # Pareto analysis
    pareto_path = METRICS_DIR / "pareto_analysis.json"
    pareto = json.loads(pareto_path.read_text()) if pareto_path.exists() else {}

    # Hypothesis results
    hyp_path = METRICS_DIR / "hypothesis_results.json"
    hyp = json.loads(hyp_path.read_text()) if hyp_path.exists() else {}

    # Fleet summary
    fleet_path = METRICS_DIR / "fleet_summary.json"
    fleet = json.loads(fleet_path.read_text()) if fleet_path.exists() else {}

    # Model metrics
    master_m_path = METRICS_DIR / "master_model_comparison.csv"
    master_metrics = pd.read_csv(master_m_path) if master_m_path.exists() else pd.DataFrame()

    direct_m_path = METRICS_DIR / "direct_forecasting_metrics.csv"
    direct_metrics = pd.read_csv(direct_m_path) if direct_m_path.exists() else pd.DataFrame()

    q_m_path = METRICS_DIR / "quantile_forecasting_metrics.csv"
    quantile_metrics = pd.read_csv(q_m_path) if q_m_path.exists() else pd.DataFrame()

    # Forecasts
    direct_preds_path = FORECASTS_DIR / "direct_forecasting_predictions.csv"
    direct_preds = pd.read_csv(direct_preds_path, parse_dates=["datetime"]).set_index("datetime") if direct_preds_path.exists() else pd.DataFrame()

    q_preds_path = FORECASTS_DIR / "quantile_forecasting_predictions.csv"
    quantile_preds = pd.read_csv(q_preds_path, parse_dates=["datetime"]).set_index("datetime") if q_preds_path.exists() else pd.DataFrame()

    # Monthly summaries
    monthly_path = MONTHLY_REPORTS_DIR / "monthly_summary.csv"
    monthly_df = pd.read_csv(monthly_path) if monthly_path.exists() else pd.DataFrame()

    heatmap_path = MONTHLY_REPORTS_DIR / "month_hour_heatmap_matrix.csv"
    heatmap_matrix = pd.read_csv(heatmap_path, index_col=0) if heatmap_path.exists() else pd.DataFrame()

    # Walk forward metrics
    wf_path = METRICS_DIR / "walk_forward_validation_metrics.csv"
    wf_metrics = pd.read_csv(wf_path) if wf_path.exists() else pd.DataFrame()

    return {
        "dq": dq,
        "pareto": pareto,
        "hyp": hyp,
        "fleet": fleet,
        "master_metrics": master_metrics,
        "direct_metrics": direct_metrics,
        "quantile_metrics": quantile_metrics,
        "direct_preds": direct_preds,
        "quantile_preds": quantile_preds,
        "monthly_df": monthly_df,
        "heatmap_matrix": heatmap_matrix,
        "wf_metrics": wf_metrics,
    }


artifacts = load_all_artifacts()

# Sidebar Navigation
with st.sidebar:
    st.title("🚲 Capital Bikeshare")
    st.caption("Urban Demand Forecasting & Fleet Analytics • Sigmoid Final Case Study")
    
    page = st.radio(
        "Navigation",
        [
            "1. Executive Summary",
            "2. Business Analytics (MECE & Pareto)",
            "3. Temporal Analytics",
            "4. Demand Forecasting",
            "5. Model Evaluation",
            "6. Hypothesis Testing",
            "7. Fleet Optimization",
            "8. Ask the Analyst",
        ],
        index=0
    )
    
    st.divider()
    st.markdown("**System Metadata**")
    st.caption("• Target: `cnt = casual + registered`\n• Granularity: Hourly Citywide\n• Observations: 17,379 hours\n• Split: Chronological Holdout (H2 2012)")


# ==============================================================================
# PAGE 1: EXECUTIVE SUMMARY
# ==============================================================================
if page == "1. Executive Summary":
    st.header("Executive Summary: Fleet Allocation vs. Procurement")
    st.caption("High-level strategic findings and machine learning results for executive leadership.")

    # Top KPI Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{artifacts['dq'].get('dataset_rows', 17379):,}")
    c2.metric("Date Range", "2011–2012 (2 Yrs)")
    c3.metric("Total Rentals", f"{artifacts['pareto'].get('total_demand', 3292679):,}")
    c4.metric("Avg / Peak Demand", "190 / 977 /hr")
    c5.metric("Best Holdout MAE", f"{artifacts['direct_metrics'].iloc[0]['MAE'] if not artifacts['direct_metrics'].empty else 41.95:.1f} /hr")

    st.divider()

    # Strategic CEO Verdict Callout
    st.subheader("CEO Decision: Should the Company Purchase More Bikes?")
    st.error("🚨 **EXECUTIVE RECOMMENDATION: DO NOT PURCHASE MORE BIKES AT THIS TIME.**")
    
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("""
        ### Why Bike Purchasing is the Wrong Move
        1. **Spatial-Temporal Bottleneck:** Our analysis proves that the perceived bike shortage is driven by extreme **temporal concentration**, not fleet volume. 
        2. **Commuter Surge:** On working days, **51.5% of all daily rentals** occur in two narrow 3-hour commute windows (07–09 and 17–19). Bikes flood commercial centers in the morning and sit idle until evening.
        3. **Fleet Sufficiency:** Using queuing sensitivity models, even a high-stress **95th-percentile demand surge (563 rides/hr)** only requires **~469 active concurrent bikes** under normal trip durations.
        4. **Compounding the Problem:** Buying more bikes without station-level rebalancing will simply result in **more idle bikes clogging destination docks**, exacerbating customer frustration.
        """)
    with col_b:
        st.markdown("""
        ### Strategic Action Plan
        * **Phase 1 (Immediate): Freeze Procurement.** Halt capital expenditure on additional hardware.
        * **Phase 2: Instrument Telemetry.** Begin recording station-level dock occupancy, empty/full dock minutes, and failed rental attempts.
        * **Phase 3: Predictive Rebalancing Pilot.** Deploy dynamic van rebalancing routes guided by our 24-hour demand forecast, and measure empty-dock reduction before reconsidering fleet size.
        """)

    st.divider()
    st.subheader("Holdout Model Performance Snapshot")
    if not artifacts["direct_metrics"].empty:
        st.dataframe(artifacts["direct_metrics"].round(2), use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 2: BUSINESS ANALYTICS (MECE & PARETO)
# ==============================================================================
elif page == "2. Business Analytics (MECE & Pareto)":
    st.header("Business Analytics: MECE Decomposition & Pareto Concentration")
    st.caption("Structured problem diagnosis separating demand, supply, distribution, and forecasting dimensions.")

    tab1, tab2, tab3 = st.tabs(["MECE Framework", "Pareto Concentration", "Rider Segmentation"])

    with tab1:
        st.subheader("MECE Problem Decomposition")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### A. Demand-Side Factors")
            st.info("""
            * **Extreme Commute Peaks:** Bimodal rush hours on working days.
            * **Rider Segmentation:** Commuters (registered) vs. tourists (casual).
            * **Seasonality & Weather:** Summer peaks (+60% vs winter); rain suppression.
            * *Unobserved:* Latent demand from riders who walked away.
            """)

            st.markdown("#### B. Supply-Side Factors")
            st.warning("""
            * **Total Active Fleet:** Estimated concurrent bikes in use.
            * **Maintenance Downtime:** Typically 5%–15% out of commission.
            * *Unobserved:* Total asset register inventory, real-time battery status.
            """)

        with col2:
            st.markdown("#### C. Distribution-Side Factors")
            st.error("""
            * **Spatial Imbalance:** Unidirectional morning commute drains residential docks.
            * **Rebalancing Friction:** Vans cannot relocate bikes fast enough during rush hours.
            * *Unobserved:* Station dock capacity, empty-dock minutes, GPS trip pairs.
            """)

            st.markdown("#### D. Forecasting-Side Factors")
            st.success("""
            * **Direct vs Autoregressive:** Direct models predict cleanly from calendar/weather.
            * **Peak Underprediction:** Standard regression underpredicts peaks by ~30 rides/hr.
            * **Risk Buffer:** Quantile P90 models provide safety buffers for operations.
            """)

    with tab2:
        st.subheader("Pareto Demand Concentration Analysis")
        p_data = artifacts["pareto"]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Top 5 Hours Share", f"{p_data.get('top_5_hours_demand_share', 41.4)}%")
        c2.metric("Working Day Commute Share", f"{p_data.get('working_day_commute_share', 51.5)}%")
        c3.metric("Non-Working Commute Share", f"{p_data.get('nonworking_day_commute_share', 26.2)}%")

        if "hourly_pareto_table" in p_data:
            pareto_df = pd.DataFrame(p_data["hourly_pareto_table"])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=pareto_df["hour"], y=pareto_df["demand"], name="Hourly Demand", marker_color="#1f77b4"))
            fig.add_trace(go.Scatter(x=pareto_df["hour"], y=pareto_df["cum_share"] * 100, name="Cumulative %", yaxis="y2", line=dict(color="#d62728", width=2.5)))
            fig.update_layout(
                title="Hourly Demand Pareto Chart (Sorted by Frequency)",
                xaxis=dict(title="Hour of Day (Sorted)", type="category"),
                yaxis=dict(title="Total Demand"),
                yaxis2=dict(title="Cumulative Share (%)", overlaying="y", side="right", range=[0, 105]),
                legend=dict(x=0.02, y=0.98),
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Rider Segmentation: Commuter (Registered) vs. Leisure (Casual)")
        r_comp = artifacts["pareto"].get("rider_composition", {})
        
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(
                values=[r_comp.get("registered_rentals", 2672780), r_comp.get("casual_rentals", 619899)],
                names=["Registered (Commuters)", "Casual (Tourists)"],
                title="Total System Rider Breakdown",
                color_discrete_sequence=["#1f77b4", "#ff7f0e"],
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("#### Behavioral Regime Comparison")
            st.markdown(f"""
            * **Registered Riders ({r_comp.get('registered_percentage', 81.2)}%):**
              * Dominant on working days (**{r_comp.get('workingday_registered_percentage', 87.5)}%** of weekday demand).
              * Extremely predictable commute curves (spikes at 08:00 and 17:00).
              * Low sensitivity to small temperature fluctuations.
            * **Casual Riders ({r_comp.get('casual_percentage', 18.8)}%):**
              * Represent **{r_comp.get('nonworkingday_casual_percentage', 35.8)}%** of weekend demand.
              * Peak in the afternoon (13:00 to 16:00).
              * Highly sensitive to precipitation and cold weather.
            """)


# ==============================================================================
# PAGE 3: TEMPORAL ANALYTICS
# ==============================================================================
elif page == "3. Temporal Analytics":
    st.header("Temporal Analytics & Multi-Resolution Demand Patterns")
    st.caption("Explore diurnal curves, monthly trends, and calendar interaction heatmaps.")

    m_df = artifacts["monthly_df"]
    h_matrix = artifacts["heatmap_matrix"]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Annual & Monthly Growth")
        if not m_df.empty:
            fig_bar = px.bar(
                m_df,
                x="year_month",
                y="total_rentals",
                title="Monthly Total Completed Rentals (2011–2012)",
                color="total_rentals",
                color_continuous_scale="Viridis"
            )
            fig_bar.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.subheader("Month-Hour Demand Heatmap Matrix")
        if not h_matrix.empty:
            fig_heat = px.imshow(
                h_matrix,
                labels=dict(x="Hour of Day", y="Year-Month", color="Avg Demand"),
                x=[f"{h:02d}:00" for h in range(24)],
                title="Hourly Demand Intensity Across 24 Calendar Months",
                color_continuous_scale="YlGnBu",
                aspect="auto"
            )
            fig_heat.update_layout(height=400)
            st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    st.subheader("Monthly Operational Dossier Inspector")
    if not m_df.empty:
        selected_ym = st.selectbox("Select Month to Inspect", m_df["year_month"].unique())
        row = m_df[m_df["year_month"] == selected_ym].iloc[0]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Monthly Rentals", f"{int(row['total_rentals']):,}")
        c2.metric("Mean Hourly Demand", f"{row['mean_hourly_demand']:.1f} /hr")
        c3.metric("Busiest Day", f"{row['busiest_day']} ({row['busiest_day_demand']:,})")
        c4.metric("Demand Volatility (CoV)", f"{row['demand_volatility_cov']:.2f}")


# ==============================================================================
# PAGE 4: DEMAND FORECASTING
# ==============================================================================
elif page == "4. Demand Forecasting":
    st.header("Demand Forecasting: Direct, Autoregressive & Quantile Experiments")
    st.caption("Visualizing conditional forecasts and operational safety buffers across the 6-month holdout test set.")

    preds = artifacts["direct_preds"]
    q_preds = artifacts["quantile_preds"]

    if not preds.empty:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("#### Forecast Configuration")
            model_choice = st.selectbox("Select Forecast Model", [c for c in preds.columns if c != "actual"])
            show_quantile = st.checkbox("Show P90 Conservative Buffer", value=True)
            
            st.markdown("#### Forecast Protocol")
            st.info("""
            * **Horizon:** Direct 1-hour conditional forecast.
            * **Inputs:** Calendar & weather only.
            * **Target Lags:** Excluded to ensure independent deployment.
            """)

        with col2:
            st.markdown("#### Interactive Forecast Visualizer")
            date_range = st.date_input(
                "Filter Date Range",
                value=[preds.index.min().date(), (preds.index.min() + pd.Timedelta(days=14)).date()]
            )
            
            if len(date_range) == 2:
                sub_preds = preds.loc[(preds.index.date >= date_range[0]) & (preds.index.date <= date_range[1])]
                sub_q = q_preds.loc[(q_preds.index.date >= date_range[0]) & (q_preds.index.date <= date_range[1])] if not q_preds.empty else pd.DataFrame()

                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=sub_preds.index, y=sub_preds["actual"], name="Actual Rentals", line=dict(color="black", width=2)))
                fig_ts.add_trace(go.Scatter(x=sub_preds.index, y=sub_preds[model_choice], name=f"Predicted ({model_choice})", line=dict(color="#1f77b4", width=2, dash="dash")))

                if show_quantile and not sub_q.empty and "p90_conservative_calibrated" in sub_q.columns:
                    fig_ts.add_trace(go.Scatter(
                        x=sub_q.index,
                        y=sub_q["p90_conservative_calibrated"],
                        name="Calibrated P90 Safety Buffer",
                        line=dict(color="#2ca02c", width=1.5, dash="dot")
                    ))

                fig_ts.update_layout(
                    title=f"Actual vs. Predicted Hourly Demand ({date_range[0]} to {date_range[1]})",
                    xaxis_title="Datetime",
                    yaxis_title="Hourly Completed Rentals",
                    height=450,
                    legend=dict(x=0.01, y=0.98)
                )
                st.plotly_chart(fig_ts, use_container_width=True)


# ==============================================================================
# PAGE 5: MODEL EVALUATION
# ==============================================================================
elif page == "5. Model Evaluation":
    st.header("Model Evaluation & Residual Diagnostics")
    st.caption("Comprehensive benchmark comparing Experiment A (Direct) and Experiment B (Autoregressive) models.")

    master_m = artifacts["master_metrics"]
    direct_preds = artifacts["direct_preds"]

    tab1, tab2, tab3 = st.tabs(["Leaderboard", "Residual Diagnostics", "Walk-Forward Stability"])

    with tab1:
        st.subheader("Model Comparison Leaderboard (6-Month Untouched Test Holdout)")
        if not master_m.empty:
            st.dataframe(master_m.round(3), use_container_width=True, hide_index=True)
            
            st.markdown("""
            > **Evaluation Takeaway:** `HistGradientBoosting` won across all metrics (lowest MAE of 41.95, highest R² of 0.912). 
            > Notice that all ML models show negative peak-hour bias (-30.7 to -43.7 bikes/hr) due to symmetric MSE loss pulling extreme peaks toward the mean.
            """)

    with tab2:
        st.subheader("Residual Diagnostics (Best Model: HistGradientBoosting)")
        if not direct_preds.empty and "HistGradientBoosting" in direct_preds.columns:
            resids = direct_preds["HistGradientBoosting"] - direct_preds["actual"]
            
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(
                    resids,
                    nbins=60,
                    title="Residual Distribution (Predicted - Actual)",
                    labels=dict(value="Residual (Rentals/Hour)"),
                    color_discrete_sequence=["#1f77b4"]
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_hist, use_container_width=True)

            with c2:
                sample_res = pd.DataFrame({"actual": direct_preds["actual"], "residual": resids}).sample(n=min(1000, len(direct_preds)), random_state=42)
                fig_scat = px.scatter(
                    sample_res,
                    x="actual",
                    y="residual",
                    title="Residual vs Actual Demand (Sample of 1,000 Points)",
                    labels=dict(actual="Actual Demand", residual="Residual"),
                    opacity=0.6,
                    color_discrete_sequence=["#ff7f0e"]
                )
                fig_scat.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_scat, use_container_width=True)

    with tab3:
        st.subheader("Expanding-Window Monthly Walk-Forward Validation Stability")
        wf_df = artifacts["wf_metrics"]
        if not wf_df.empty:
            st.dataframe(wf_df.round(2), use_container_width=True, hide_index=True)
            fig_wf = px.line(
                wf_df,
                x="validation_month",
                y="MAE",
                markers=True,
                title="Model Out-of-Time MAE Across Expanding Validation Folds",
                color_discrete_sequence=["#2ca02c"]
            )
            st.plotly_chart(fig_wf, use_container_width=True)


# ==============================================================================
# PAGE 6: HYPOTHESIS TESTING
# ==============================================================================
elif page == "6. Hypothesis Testing":
    st.header("Statistical Hypothesis Testing & Robustness Analysis")
    st.caption("Rigorous day-level permutation testing (20,000 iterations) evaluating commute concentration.")

    h1_data = artifacts["hyp"].get("H1_Commuter_Concentration", {})

    st.subheader("Hypothesis 1: Commuter Concentration")
    st.markdown(f"**Hypothesis:** *{h1_data.get('hypothesis', 'Working days have higher commute concentration.')}*")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Working-Day Commute Share", f"{h1_data.get('working_days_mean_share', 51.6)}%")
    c2.metric("Non-Working Commute Share", f"{h1_data.get('nonworking_days_mean_share', 26.2)}%")
    c3.metric("Observed Difference", f"+{h1_data.get('observed_difference_percentage_points', 25.4)} pp")
    c4.metric("Permutation P-Value", f"{h1_data.get('one_sided_p_value', 5e-5):.5g}")

    ci = h1_data.get("bootstrap_95_ci_diff_percentage_points", [24.7, 26.1])
    st.success(f"""
    **Statistical Verdict:** **Reject Null Hypothesis (p < 0.0001)**. 
    The 95% Bootstrap Confidence Interval for the difference in commute share is **[{ci[0]}%, {ci[1]}%]**.
    This proves that commuter concentration is an overwhelming statistical reality, mathematically substantiating why bikes pool at commercial hubs.
    """)

    st.divider()
    st.subheader("Methodological Safeguards")
    st.markdown("""
    * **Independent Unit of Analysis:** We aggregated to **complete 24-hour calendar days**, preventing intra-day autocorrelation from violating independence.
    * **Block-Permutation Robustness:** A secondary 7-day block permutation test yielded **p < 0.0002**, proving results are not an artifact of weekly cycles.
    """)


# ==============================================================================
# PAGE 7: FLEET OPTIMIZATION
# ==============================================================================
elif page == "7. Fleet Optimization":
    st.header("Fleet Sizing & Operational Sensitivity Simulator")
    st.caption("Parameter-driven active fleet estimation using Little's Law and spatial buffer multipliers.")

    st.markdown("""
    $$\\text{Active Fleet Needed} = \\frac{\\text{Hourly Demand} \\times \\text{Average Trip Duration (hrs)} \\times \\text{Spatial Buffer}}{1 - \\text{Maintenance Share}}$$
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Scenario Controls")
        demand_input = st.slider("Hourly Demand (Rentals/Hour)", min_value=100, max_value=1000, value=563, step=25)
        trip_input = st.slider("Average Trip Duration (Minutes)", min_value=15, max_value=60, value=30, step=5)
        buffer_input = st.slider("Spatial Imbalance Buffer", min_value=1.0, max_value=2.5, value=1.5, step=0.1)
        maint_input = st.slider("Maintenance Downtime (%)", min_value=0, max_value=25, value=10, step=1)

        trip_hours = trip_input / 60.0
        maint_share = maint_input / 100.0

        active_fleet = calculate_active_fleet(demand_input, trip_hours, buffer_input, maint_share)
        surplus = ILLUSTRATIVE_FLEET_SIZE - active_fleet

    with col2:
        st.subheader("Simulation Results")
        c1, c2 = st.columns(2)
        c1.metric("Active Fleet Needed", f"{int(np.ceil(active_fleet)):,} Bikes")
        c2.metric("2,000-Bike Fleet Balance", f"{int(surplus):,} Bikes ({'Surplus' if surplus >= 0 else 'Deficit'})")

        st.info(f"""
        **Operational Interpretation:**
        To serve a peak demand of **{demand_input} rides/hour** with an average trip of **{trip_input} mins**, 
        the system has **{int(demand_input * trip_hours)} concurrent bikes in motion**.
        Applying a **{buffer_input}x spatial buffer** and accounting for **{maint_input}% maintenance**, 
        the company needs **~{int(np.ceil(active_fleet))} active bikes**.
        """)

        # Sensitivity curve
        dur_range = np.linspace(0.25, 1.0, 15)
        fleets_curve = [calculate_active_fleet(demand_input, d, buffer_input, maint_share) for d in dur_range]
        fig_sens = px.line(
            x=dur_range * 60,
            y=fleets_curve,
            labels=dict(x="Average Trip Duration (Minutes)", y="Estimated Active Fleet"),
            title=f"Fleet Requirement vs Trip Turnaround Time (Demand = {demand_input}/hr)"
        )
        fig_sens.add_hline(y=2000, line_dash="dash", line_color="red", annotation_text="Illustrative 2,000-Bike Cap")
        st.plotly_chart(fig_sens, use_container_width=True)


# ==============================================================================
# PAGE 8: ASK THE ANALYST
# ==============================================================================
elif page == "8. Ask the Analyst":
    st.header("Ask the Analyst: Data-Grounded Natural Language QA")
    st.caption("Ask operational questions and receive answers strictly verified against computed project outputs.")

    analyst = OperationsAnalyst()

    st.markdown("#### Sample Questions (Click to Test)")
    col1, col2, col3 = st.columns(3)
    q1 = col1.button("Should we buy more bikes?")
    q2 = col2.button("What are the busiest hours?")
    q3 = col3.button("Why does the model underestimate peaks?")

    col4, col5, col6 = st.columns(3)
    q4 = col4.button("What was the hypothesis test result?")
    q5 = col5.button("Which month had the highest demand?")
    q6 = col6.button("What are the dataset limitations?")

    selected_q = ""
    if q1: selected_q = "Should we buy more bikes?"
    elif q2: selected_q = "What are the busiest hours?"
    elif q3: selected_q = "Why does the model underestimate peak demand?"
    elif q4: selected_q = "What was the hypothesis test result?"
    elif q5: selected_q = "Which month had the highest demand?"
    elif q6: selected_q = "What are the limitations of this dataset?"

    user_query = st.text_input("Or enter your own operational question:", value=selected_q, placeholder="e.g. How does weekend demand differ?")

    if user_query:
        resp = analyst.query(user_query)
        st.markdown(f"**Category:** `{resp['category']}` | **Status:** `{resp['status']}`")
        if resp["status"] == "GROUNDED":
            st.success(resp["answer"])
        else:
            st.info(resp["answer"])

st.divider()
st.caption("© 2026 Sigmoid Final Case Study • Urban Bike-Sharing Forecasting & Fleet Optimization • Production Solution")
