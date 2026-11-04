"""BIKESHARE INTELLIGENCE: Demand Forecasting & Operations Assistant
A deployable Streamlit application empowering operations managers to understand demand,
anticipate peak shortages, simulate fleet sizing, and query system intelligence in plain English.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import time
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
    TARGET_COL,
    CASUAL_COL,
    REGISTERED_COL,
)
from src.data_loader import load_raw_data
from src.model_loader import load_production_models, predict_single_hour
from src.query_engine import QueryEngine
from src.monthly_analytics import load_monthly_summary_data, compare_two_months, plot_monthly_comparison_bars
from src.visualizations import (
    plot_hourly_time_series,
    plot_hour_weekday_heatmap,
    plot_weekday_weekend_curves,
    plot_rider_composition,
    plot_residuals,
)
from src.fleet_optimizer import calculate_active_fleet

# ------------------------------------------------------------------------------
# 1. APPLICATION SETUP & THEMING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Bikeshare Intelligence | Operations Assistant",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom executive CSS
st.markdown("""
<style>
    .brand-header {
        font-size: 26px;
        font-weight: 800;
        color: #0f52ba;
        letter-spacing: -0.5px;
        margin-bottom: 0px;
    }
    .brand-tagline {
        font-size: 14px;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 20px;
    }
    .kpi-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #0f52ba;
    }
    .kpi-label {
        font-size: 13px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .callout-box {
        background-color: #eff6ff;
        border-left: 5px solid #0f52ba;
        padding: 16px;
        border-radius: 4px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .qa-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. CACHED DATA & ARTIFACTS
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading operational data assets...")
def get_dashboard_data():
    raw_df = load_raw_data()
    summary_df, heatmap_matrix = load_monthly_summary_data()

    pareto_path = METRICS_DIR / "pareto_analysis.json"
    pareto = json.loads(pareto_path.read_text()) if pareto_path.exists() else {}

    hyp_path = METRICS_DIR / "hypothesis_results.json"
    hyp = json.loads(hyp_path.read_text()) if hyp_path.exists() else {}

    fleet_path = METRICS_DIR / "fleet_summary.json"
    fleet = json.loads(fleet_path.read_text()) if fleet_path.exists() else {}

    master_m_path = METRICS_DIR / "master_model_comparison.csv"
    master_metrics = pd.read_csv(master_m_path) if master_m_path.exists() else pd.DataFrame()

    direct_m_path = METRICS_DIR / "direct_forecasting_metrics.csv"
    direct_metrics = pd.read_csv(direct_m_path) if direct_m_path.exists() else pd.DataFrame()

    q_m_path = METRICS_DIR / "quantile_forecasting_metrics.csv"
    quantile_metrics = pd.read_csv(q_m_path) if q_m_path.exists() else pd.DataFrame()

    direct_preds_path = FORECASTS_DIR / "direct_forecasting_predictions.csv"
    direct_preds = pd.read_csv(direct_preds_path, parse_dates=["datetime"]).set_index("datetime") if direct_preds_path.exists() else pd.DataFrame()

    q_preds_path = FORECASTS_DIR / "quantile_forecasting_predictions.csv"
    quantile_preds = pd.read_csv(q_preds_path, parse_dates=["datetime"]).set_index("datetime") if q_preds_path.exists() else pd.DataFrame()

    wf_path = METRICS_DIR / "walk_forward_validation_metrics.csv"
    wf_metrics = pd.read_csv(wf_path) if wf_path.exists() else pd.DataFrame()

    dq_path = REPORTS_DIR / "data_quality_report.json"
    dq = json.loads(dq_path.read_text()) if dq_path.exists() else {}

    return {
        "raw_df": raw_df,
        "summary_df": summary_df,
        "heatmap_matrix": heatmap_matrix,
        "pareto": pareto,
        "hyp": hyp,
        "fleet": fleet,
        "master_metrics": master_metrics,
        "direct_metrics": direct_metrics,
        "quantile_metrics": quantile_metrics,
        "direct_preds": direct_preds,
        "quantile_preds": quantile_preds,
        "wf_metrics": wf_metrics,
        "dq": dq,
    }


data_bundle = get_dashboard_data()
query_engine = QueryEngine()


# ------------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & SYSTEM METADATA
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-header">🚲 BIKESHARE INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tagline">Demand Forecasting & Operations Assistant</div>', unsafe_allow_html=True)

    page = st.radio(
        "Operational Views",
        [
            "Executive Overview",
            "Demand Explorer",
            "Demand Forecasting",
            "Monthly Temporal Intelligence",
            "Fleet Optimization",
            "Model Performance & Explainability",
            "Ask the Analyst (AI Assistant)",
        ],
        index=0
    )

    st.divider()
    st.markdown("### System Context")
    st.caption("""
    * **System:** Capital Bikeshare (Washington, D.C.)
    * **Target:** `cnt = casual + registered`
    * **Period:** 2011–2012 (17,379 Hourly Records)
    * **Model:** HistGradientBoosting (MAE: 41.95, R²: 0.912)
    * **Evaluation:** Out-of-Time Chronological Holdout (H2 2012)
    """)
    st.caption("Deployment: Streamlit Community Cloud Ready • Production Architecture")


# ==============================================================================
# PAGE 1: EXECUTIVE OPERATIONS OVERVIEW
# ==============================================================================
if page == "Executive Overview":
    st.title("Executive Operations Overview")
    st.caption("High-level system KPIs, empirical demand patterns, and strategic procurement recommendations.")

    # KPI Top Bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-container"><div class="kpi-value">{data_bundle["dq"].get("dataset_rows", 17379):,}</div><div class="kpi-label">Hourly Observations</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-container"><div class="kpi-value">{data_bundle["pareto"].get("total_demand", 3292679):,}</div><div class="kpi-label">Total Completed Rentals</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-container"><div class="kpi-value">190 / 977</div><div class="kpi-label">Avg / Peak Demand (hr)</div></div>', unsafe_allow_html=True)
    with c4:
        best_mae = data_bundle["direct_metrics"].iloc[0]["MAE"] if not data_bundle["direct_metrics"].empty else 41.95
        st.markdown(f'<div class="kpi-container"><div class="kpi-value">{best_mae:.1f} /hr</div><div class="kpi-label">Champion Model MAE (R²: 0.91)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Executive CEO Recommendation Callout
    st.markdown("""
    <div class="callout-box">
        <h3 style="margin-top:0; color:#0f52ba;">CEO Decision Brief: Should the Company Purchase More Bikes?</h3>
        <p style="font-size:16px; font-weight:600; color:#b91c1c;">
            🚨 RECOMMENDATION: FREEZE FLEET EXPANSION. DO NOT PURCHASE MORE BIKES.
        </p>
        <p>
            Riders' inability to find bikes during rush hours is driven by <strong>extreme temporal concentration and directional tidal flows</strong>, 
            not a shortage of total bikes. On working days, <strong>51.5% of all daily rentals</strong> occur in two narrow 3-hour windows (07–09 and 17–19). 
            Under Little's Law, even a 95th-percentile demand surge requires only <strong>~469 active concurrent bikes</strong>. 
            Purchasing additional hardware without station rebalancing will only create more idle bikes clogging downtown docks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic Insights Section
    st.subheader("Dynamic Operational Intelligence")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        * **Peak Commute Contribution:** Working days generate **51.53%** of their total volume in peak commute hours (vs. 26.14% on weekends).
        * **Top Busiest Hours:** 17:00 (5 PM), 18:00 (6 PM), and 08:00 (8 AM) generate **41.37%** of all daily throughput.
        * **User Segmentation:** Registered commuters account for **81.2%** of total volume, peaking sharply on weekdays.
        """)
    with col_b:
        st.markdown(f"""
        * **Peak Seasonal Surge:** Monthly demand peaks in late summer/early autumn (**September 2012** saw 218,573 completed rentals).
        * **Model Rush-Hour Bias:** Primary ML model underestimates peak windows by an average of **-30.69 bikes/hour**.
        * **Operational Buffer:** Operations must deploy a calibrated P90 buffer (+31.4 bikes/hr) to absorb peak surges.
        """)


# ==============================================================================
# PAGE 2: DEMAND EXPLORER
# ==============================================================================
elif page == "Demand Explorer":
    st.title("Interactive Demand Explorer")
    st.caption("Filter and inspect hourly rental patterns across dates, calendar periods, and rider segments.")

    raw_df = data_bundle["raw_df"].copy()

    # Filter Bar
    f1, f2, f3, f4 = st.columns(4)
    sel_year = f1.selectbox("Filter Year", ["All Years", 2011, 2012])
    sel_daytype = f2.selectbox("Day Classification", ["All Days", "Working Days Only", "Weekends Only"])
    sel_peak = f3.selectbox("Commute Window", ["All Hours", "Commute Peak (7-9 & 17-19)", "Off-Peak Only"])
    sel_segment = f4.selectbox("Rider Type", ["Total (cnt)", "Registered Only", "Casual Only"])

    # Apply Filters
    filtered_df = raw_df
    if sel_year != "All Years":
        filtered_df = filtered_df[filtered_df.index.year == sel_year]
    if sel_daytype == "Working Days Only":
        filtered_df = filtered_df[filtered_df["workingday"] == 1]
    elif sel_daytype == "Weekends Only":
        filtered_df = filtered_df[filtered_df["is_weekend"] == 1]
    if sel_peak == "Commute Peak (7-9 & 17-19)":
        filtered_df = filtered_df[filtered_df.index.hour.isin(ALL_PEAK_HOURS)]
    elif sel_peak == "Off-Peak Only":
        filtered_df = filtered_df[~filtered_df.index.hour.isin(ALL_PEAK_HOURS)]

    # Metrics on filtered data
    target_metric_col = "cnt" if sel_segment == "Total (cnt)" else ("registered" if sel_segment == "Registered Only" else "casual")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered Records", f"{len(filtered_df):,}")
    m2.metric("Total Segment Rentals", f"{int(filtered_df[target_metric_col].sum()):,}")
    m3.metric("Average Hourly Demand", f"{filtered_df[target_metric_col].mean():.1f} /hr")
    m4.metric("Max Recorded Hour", f"{filtered_df[target_metric_col].max():,} /hr")

    st.divider()

    # Visualizations
    tab1, tab2, tab3 = st.tabs(["Diurnal Patterns", "Concentration Heatmap", "User Mix"])
    with tab1:
        st.plotly_chart(plot_weekday_weekend_curves(filtered_df), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_hour_weekday_heatmap(filtered_df), use_container_width=True)
    with tab3:
        st.plotly_chart(plot_rider_composition(data_bundle["pareto"]), use_container_width=True)

    # CSV Download Button
    csv_data = filtered_df[["dteday", "hr", "weathersit", "temp", "hum", "windspeed", "casual", "registered", "cnt"]].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Dataset (CSV)",
        data=csv_data,
        file_name="filtered_bikeshare_demand.csv",
        mime="text/csv"
    )


# ==============================================================================
# PAGE 3: DEMAND FORECASTING
# ==============================================================================
elif page == "Demand Forecasting":
    st.title("Demand Forecasting & Live Scenario Inference")
    st.caption("Direct conditional forecasting from calendar and weather signals with calibrated P90 safety buffers.")

    preds = data_bundle["direct_preds"]
    q_preds = data_bundle["quantile_preds"]

    tab_eval, tab_infer = st.tabs(["Holdout Forecast Inspection", "Live Single-Hour Prediction"])

    with tab_eval:
        st.subheader("Holdout Test Predictions (H2 2012)")
        c1, c2 = st.columns([1, 3])
        with c1:
            model_opt = st.selectbox("Select Model", [c for c in preds.columns if c != "actual"], index=0)
            show_p90_band = st.checkbox("Show Calibrated P90 Safety Buffer", value=True)
            st.info("""
            * **Evaluation Period:** 2012-07-01 to 2012-12-31
            * **Horizon:** Direct 1-hour ahead conditional forecast.
            * **Feature Policy:** Target lags strictly excluded to ensure independence from future actuals.
            """)
        with c2:
            st.plotly_chart(plot_hourly_time_series(preds.iloc[:336], model_opt, show_p90_band, q_preds), use_container_width=True)

    with tab_infer:
        st.subheader("Generate Real-Time Prediction for Future Operational Window")
        st.caption("Calculates expected demand using strictly features available at prediction time.")

        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            sel_date = st.date_input("Target Date", value=pd.Timestamp.now().date())
            sel_hour = st.slider("Target Hour", 0, 23, 17)
        with p_col2:
            sel_weather = st.selectbox("Forecasted Weather Situation", [
                "1: Clear / Few Clouds",
                "2: Mist / Cloudy",
                "3: Light Rain / Light Snow",
                "4: Heavy Rain / Ice / Fog"
            ], index=0)
            sel_temp = st.slider("Forecasted Temperature (°C)", -5.0, 40.0, 24.0, 0.5)
        with p_col3:
            sel_hum = st.slider("Forecasted Humidity (%)", 10, 100, 50)
            sel_wind = st.slider("Forecasted Wind Speed (km/h)", 0, 50, 15)

        weather_code = int(sel_weather.split(":")[0])
        dt_target = pd.Timestamp(f"{sel_date} {sel_hour:02d}:00:00")

        if st.button("🔮 Generate Demand Forecast"):
            result = predict_single_hour(
                dt=dt_target,
                weather_sit=weather_code,
                temp_celsius=sel_temp,
                humidity_pct=sel_hum,
                wind_speed_kmh=sel_wind
            )

            res1, res2, res3 = st.columns(3)
            res1.metric("Expected Point Demand", f"{result['predicted_demand_point']:.0f} rentals/hr")
            res2.metric("Median (P50) Demand", f"{result['predicted_demand_p50']:.0f} rentals/hr")
            res3.metric("Conservative Buffer (P90)", f"{result['predicted_demand_p90_buffer']:.0f} rentals/hr")

            if result["is_peak_hour"]:
                st.warning("⚠️ Peak Commute Hour Detected: Demand is prone to rapid spatial imbalance. Stage rebalancing vans in advance.")
            else:
                st.success("✅ Normal operating conditions predicted.")


# ==============================================================================
# PAGE 4: MONTHLY TEMPORAL INTELLIGENCE
# ==============================================================================
elif page == "Monthly Temporal Intelligence":
    st.title("Monthly Temporal Intelligence")
    st.caption("Deep seasonal analytics, monthly dossiers, and side-by-side operational comparisons.")

    summary_df = data_bundle["summary_df"]
    heatmap_matrix = data_bundle["heatmap_matrix"]

    if not summary_df.empty:
        # Month Selector
        m_list = summary_df["year_month"].tolist()
        col1, col2 = st.columns(2)
        m_a = col1.selectbox("Primary Month", m_list, index=len(m_list) - 4)
        m_b = col2.selectbox("Comparison Month", m_list, index=len(m_list) - 16)

        comp = compare_two_months(m_a, m_b, summary_df)

        st.subheader("Head-to-Head Monthly Analysis")
        st.plotly_chart(plot_monthly_comparison_bars(comp["month_1"], comp["month_2"]), use_container_width=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Total Rentals ({m_a})", f"{comp['month_1']['total_rentals']:,}")
        k2.metric(f"Total Rentals ({m_b})", f"{comp['month_2']['total_rentals']:,}", delta=f"{comp['percentage_growth']}% vs {m_a}")
        k3.metric(f"Avg Hourly ({m_b})", f"{comp['month_2']['mean_hourly_demand']:.1f}", delta=f"{comp['mean_hourly_diff']:.1f}")
        k4.metric(f"Commuter Share ({m_b})", f"{comp['month_2']['registered_percentage']:.1f}%", delta=f"{comp['registered_pct_diff']:.1f}%")

        st.divider()
        st.subheader("Month-Hour Demand Heatmap (24-Month Matrix)")
        st.plotly_chart(px.imshow(
            heatmap_matrix,
            labels=dict(x="Hour of Day", y="Year-Month", color="Avg Rentals"),
            color_continuous_scale="YlGnBu",
            aspect="auto",
            height=400
        ), use_container_width=True)


# ==============================================================================
# PAGE 5: FLEET OPTIMIZATION
# ==============================================================================
elif page == "Fleet Optimization":
    st.title("Fleet Sizing & Queuing Sensitivity Simulator")
    st.caption("Simulate active fleet requirements using Little's Law, spatial rebalancing buffers, and maintenance shares.")

    st.markdown("""
    $$\\text{Active Fleet Needed} = \\frac{\\text{Hourly Demand} \\times \\text{Average Trip Duration (hrs)} \\times \\text{Spatial Imbalance Buffer}}{1 - \\text{Maintenance Share}}$$
    """)

    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        st.subheader("Operational Controls")
        scenario_preset = st.selectbox("Scenario Presets", [
            "High Stress (P95 Demand: 563 rides/hr)",
            "Conservative Planning (P90 Demand: 457 rides/hr)",
            "Expected Commute Peak (341 rides/hr)",
            "Low Demand (P50 Median: 142 rides/hr)",
            "Custom Manual Sizing"
        ], index=0)

        if "P95" in scenario_preset: d_preset = 563
        elif "P90" in scenario_preset: d_preset = 457
        elif "Expected" in scenario_preset: d_preset = 341
        elif "P50" in scenario_preset: d_preset = 142
        else: d_preset = 500

        demand_slider = st.slider("Target Hourly Demand (Rentals/Hour)", 50, 1000, d_preset, 25)
        trip_mins = st.slider("Average Trip Duration (Minutes)", 15, 60, 30, 5)
        buffer_slider = st.slider("Spatial Imbalance Buffer", 1.0, 2.5, 1.5, 0.1)
        maint_pct = st.slider("Maintenance Downtime (%)", 0, 25, 10, 1)

        trip_hrs = trip_mins / 60.0
        maint_val = maint_pct / 100.0
        active_needed = calculate_active_fleet(demand_slider, trip_hrs, buffer_slider, maint_val)
        fleet_balance = ILLUSTRATIVE_FLEET_SIZE - active_needed

    with f_col2:
        st.subheader("Sizing Simulator Outputs")
        sc1, sc2 = st.columns(2)
        sc1.metric("Active Fleet Needed", f"{int(np.ceil(active_needed)):,} Bikes")
        sc2.metric("Balance vs 2,000-Bike Fleet", f"{int(fleet_balance):,} Bikes ({'Surplus' if fleet_balance >= 0 else 'Deficit'})")

        st.info(f"""
        **Simulation Logic:**
        To fulfill **{demand_slider} rides/hour** with an average trip of **{trip_mins} mins**, 
        there are **{int(demand_slider * trip_hrs)} concurrent bikes in motion**.
        Applying a **{buffer_slider}x spatial distribution buffer** and **{maint_pct}% maintenance downtime**, 
        the active requirement is **~{int(np.ceil(active_needed))} bikes**.
        """)

        # Sensitivity curve
        durs = np.linspace(0.25, 1.0, 15)
        fleets = [calculate_active_fleet(demand_slider, d, buffer_slider, maint_val) for d in durs]
        fig_curve = px.line(
            x=durs * 60,
            y=fleets,
            labels=dict(x="Trip Duration (Minutes)", y="Active Fleet Required"),
            title=f"Fleet Requirement vs. Trip Turnaround Speed (Demand: {demand_slider}/hr)"
        )
        fig_curve.add_hline(y=2000, line_dash="dash", line_color="red", annotation_text="2,000-Bike Fleet Benchmark")
        st.plotly_chart(fig_curve, use_container_width=True)


# ==============================================================================
# PAGE 6: MODEL PERFORMANCE AND EXPLAINABILITY
# ==============================================================================
elif page == "Model Performance & Explainability":
    st.title("Model Performance & Error Diagnostics")
    st.caption("Benchmark comparison across Direct and Autoregressive models, with residual diagnostics and explainability.")

    master_m = data_bundle["master_metrics"]
    preds = data_bundle["direct_preds"]

    tab1, tab2, tab3 = st.tabs(["Benchmark Leaderboard", "Residual Diagnostics", "Quantile Risk Intervals"])

    with tab1:
        st.subheader("Model Evaluation on Untouched 6-Month Test Holdout")
        st.dataframe(master_m.round(3), use_container_width=True, hide_index=True)
        st.markdown("""
        * **HistGradientBoosting** achieved the lowest holdout test MAE (**41.95 rentals/hr**) and highest $R^2$ (**0.912**).
        * Negative peak-hour bias (**-30.69**) occurs because symmetric MSE loss naturally smooths extreme rush-hour spikes.
        """)

    with tab2:
        st.subheader("Residual Error Analysis (Best Model)")
        if not preds.empty and "HistGradientBoosting" in preds.columns:
            fig_hist, fig_scat = plot_residuals(preds["actual"], preds["HistGradientBoosting"])
            r_col1, r_col2 = st.columns(2)
            r_col1.plotly_chart(fig_hist, use_container_width=True)
            r_col2.plotly_chart(fig_scat, use_container_width=True)

    with tab3:
        st.subheader("Calibrated P90 Quantile Risk Intervals")
        st.dataframe(data_bundle["quantile_metrics"].round(3), use_container_width=True, hide_index=True)
        st.info("Calibrating with a +31.4 residual offset lifts empirical peak-hour coverage to **94.8%**, creating a robust operational safety buffer.")


# ==============================================================================
# PAGE 7: ASK THE ANALYST (AI OPERATIONS ASSISTANT)
# ==============================================================================
elif page == "Ask the Analyst (AI Assistant)":
    st.title("Ask the Analyst: Operations Assistant")
    st.caption("Ask natural-language questions about demand, forecasting, fleet sizing, and strategic recommendations.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Clear chat button
    c_btn1, c_btn2 = st.columns([6, 1])
    with c_btn2:
        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.rerun()

    # Suggested Prompts (Clickable chips)
    st.markdown("#### Suggested Questions")
    chips_row1 = st.columns(4)
    q_clicked = None
    if chips_row1[0].button("Should we buy more bikes?"): q_clicked = "Should we buy more bikes?"
    if chips_row1[1].button("What are the busiest hours?"): q_clicked = "What are the busiest hours?"
    if chips_row1[2].button("Which month had highest demand?"): q_clicked = "Which month had the highest demand?"
    if chips_row1[3].button("Why underestimate peak demand?"): q_clicked = "Why does the model underestimate peak demand?"

    chips_row2 = st.columns(4)
    if chips_row2[0].button("How does weekend demand differ?"): q_clicked = "How does weekend demand differ from weekdays?"
    if chips_row2[1].button("How many bikes for 500/hr?"): q_clicked = "How many active bikes might be needed for a demand of 500 rentals per hour?"
    if chips_row2[2].button("What did hypothesis test show?"): q_clicked = "What did the hypothesis test establish?"
    if chips_row2[3].button("Show me P90 demand estimate"): q_clicked = "Show me the P90 demand estimate."

    # Render previous conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "chart" in msg and msg["chart"] is not None:
                st.plotly_chart(msg["chart"], use_container_width=True)

    # Handle user query
    user_input = st.chat_input("Ask an operational question...")
    query_to_process = q_clicked if q_clicked else user_input

    if query_to_process:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": query_to_process})
        with st.chat_message("user"):
            st.markdown(query_to_process)

        # Process through QueryEngine
        response = query_engine.process_query(query_to_process)

        # Construct 5-point structured answer
        structured_markdown = f"""
**Category:** `{response['category']}`

1. **Direct Answer:**  
   {response['direct_answer']}

2. **Supporting Numerical Evidence:**  
   {response['evidence']}

3. **Business Interpretation:**  
   {response['interpretation']}

4. **Recommended Operational Action:**  
   {response['action']}

5. **Relevant Limitation:**  
   {response['limitation']}

*Source: {response['data_source']}*
"""
        # Append assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": structured_markdown,
            "chart": response.get("chart")
        })

        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(structured_markdown)
            if response.get("chart") is not None:
                st.plotly_chart(response["chart"], use_container_width=True)

st.divider()
st.caption("© 2026 Sigmoid Candidate Case Study • Bikeshare Intelligence System • Production Deployment Ready")
