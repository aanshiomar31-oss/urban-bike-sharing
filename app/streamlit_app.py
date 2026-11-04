from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
st.set_page_config(page_title='Bike Sharing Demand Operations', layout='wide')
st.title('Bike-sharing demand planning prototype')
st.caption('System-wide hourly rental forecast demonstration — not station-level inventory optimization.')
metrics_path = OUT / 'model_metrics.csv'
pred_path = OUT / 'test_predictions.csv'
if not metrics_path.exists() or not pred_path.exists():
    st.error('Run `python src/analysis.py` first to generate model outputs.')
    st.stop()
metrics = pd.read_csv(metrics_path)
preds = pd.read_csv(pred_path, parse_dates=['datetime'], index_col='datetime')
st.subheader('Model comparison')
st.dataframe(metrics, use_container_width=True)
st.subheader('Final test-period forecasts')
model_col = st.selectbox('Forecast series', ['xgb_point', 'xgb_q90_raw', 'xgb_q90_calibrated', 'seasonal_naive'])
start = st.date_input('Start date', value=preds.index.min().date(), min_value=preds.index.min().date(), max_value=preds.index.max().date())
end = st.date_input('End date', value=min(preds.index.max().date(), start), min_value=preds.index.min().date(), max_value=preds.index.max().date())
view = preds.loc[(preds.index.date >= start) & (preds.index.date <= end)]
if not view.empty:
    st.line_chart(view[['actual', model_col]])
    st.dataframe(view[['actual', model_col]].head(100), use_container_width=True)
else:
    st.warning('No observations in the selected date range.')
st.subheader('Ask the prototype')
q = st.text_input('Example: What was average demand at 8 AM?')
if q:
    low = q.lower()
    if 'average' in low and 'demand' in low and 'am' in low:
        import re
        m = re.search(r'(\d{1,2})\s*am', low)
        if m:
            h = int(m.group(1)) % 12
            st.metric(f'Average observed demand at {h:02d}:00', f"{preds[preds.index.hour == h]['actual'].mean():.1f} rentals/hour")
        else:
            st.info('Please specify an hour, e.g. 8 AM.')
    elif 'peak' in low or 'highest' in low:
        by_hour = preds.groupby(preds.index.hour).actual.mean().sort_values(ascending=False)
        st.write(f"Highest average hourly demand occurs at {int(by_hour.index[0]):02d}:00, averaging {by_hour.iloc[0]:.1f} rentals/hour in this test period.")
    elif 'shortage' in low or 'station' in low or 'redistribut' in low:
        st.warning('The supplied dataset has no station IDs, inventory, station capacity, failed rental attempts or bike movements. Station shortages and redistribution quantities cannot be inferred from this dataset.')
    elif 'buy' in low or 'purchase' in low:
        st.warning('This dataset alone cannot justify a fleet purchase decision. Obtain station-level availability, failed rental attempts, capacity, return flows and redistribution cost, then run a controlled pilot.')
    else:
        st.info('Supported questions: average demand at a specified hour, peak-demand hour, station shortage/redistribution feasibility, and whether fleet purchase can be justified.')
st.caption('All displayed numerical answers are derived from the supplied dataset or saved model outputs. The Q90 calibration is a global offset and may over-cover; inspect calibration results before operational use.')
