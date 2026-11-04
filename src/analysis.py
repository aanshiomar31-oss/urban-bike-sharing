from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'hour.csv'
OUT = ROOT / 'outputs'
FIG = OUT / 'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
ALPHA = 0.90


def load_and_audit(path=DATA):
    raw = pd.read_csv(path)
    required = {'instant','dteday','hr','cnt','casual','registered'}
    missing_cols = required - set(raw.columns)
    if missing_cols: raise ValueError(f'Missing required columns: {sorted(missing_cols)}')
    raw['datetime'] = pd.to_datetime(raw['dteday'], errors='coerce') + pd.to_timedelta(raw['hr'], unit='h')
    audit = {
        'rows': int(len(raw)), 'columns': int(raw.shape[1]),
        'missing_cells': int(raw.isna().sum().sum()),
        'duplicate_rows': int(raw.duplicated().sum()),
        'duplicate_timestamps': int(raw.datetime.duplicated().sum()),
        'negative_cnt': int((raw.cnt < 0).sum()),
        'cnt_equals_casual_plus_registered': bool((raw.cnt == raw.casual + raw.registered).all()),
        'start': str(raw.datetime.min()), 'end': str(raw.datetime.max()),
        'observed_timestamps': int(raw.datetime.nunique()),
    }
    full = pd.date_range(raw.datetime.min(), raw.datetime.max(), freq='h')
    absent = full.difference(pd.DatetimeIndex(raw.datetime))
    audit['expected_hourly_timestamps'] = int(len(full))
    audit['missing_timestamps'] = int(len(absent))
    audit['missing_timestamp_examples'] = [str(x) for x in absent[:20]]
    audit['missing_timestamp_policy'] = 'Do not synthesize target values. Keep observed rows; exact-time lag features can be missing and are handled by XGBoost.'
    return raw.sort_values('datetime').set_index('datetime'), audit


def make_features(raw):
    # Use a complete hourly index only as a time scaffold. cnt remains NaN at absent timestamps.
    full_idx = pd.date_range(raw.index.min(), raw.index.max(), freq='h')
    cnt_full = raw['cnt'].reindex(full_idx)
    f = raw.copy()
    idx = f.index
    f['hour_sin'] = np.sin(2*np.pi*idx.hour/24)
    f['hour_cos'] = np.cos(2*np.pi*idx.hour/24)
    f['weekday_sin'] = np.sin(2*np.pi*idx.dayofweek/7)
    f['weekday_cos'] = np.cos(2*np.pi*idx.dayofweek/7)
    f['is_weekend'] = (idx.dayofweek >= 5).astype(int)
    f['is_peak_hour'] = idx.hour.isin([7,8,9,16,17,18]).astype(int)
    f['workingday_peak'] = f['workingday'] * f['is_peak_hour']
    # Exact elapsed-time lags, not previous row lags. These are valid for rolling one-hour-ahead forecasts
    # only when the preceding observation has arrived.
    f['lag_1h'] = cnt_full.shift(1).reindex(idx)
    f['lag_24h'] = cnt_full.shift(24).reindex(idx)
    f['lag_168h'] = cnt_full.shift(168).reindex(idx)
    # Trailing means end at t-1, never include target at t.
    f['rolling_mean_24h'] = cnt_full.shift(1).rolling(24, min_periods=18).mean().reindex(idx)
    f['rolling_mean_168h'] = cnt_full.shift(1).rolling(168, min_periods=120).mean().reindex(idx)
    f['month_sin'] = np.sin(2*np.pi*(idx.month-1)/12)
    f['month_cos'] = np.cos(2*np.pi*(idx.month-1)/12)
    f['hour'] = idx.hour
    f['dayofweek'] = idx.dayofweek
    f['month'] = idx.month
    return f

FEATURES = ['hour_sin','hour_cos','weekday_sin','weekday_cos','is_weekend','is_peak_hour','workingday_peak',
            'season','yr','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed',
            'lag_1h','lag_24h','lag_168h','rolling_mean_24h','rolling_mean_168h','month_sin','month_cos']


def pinball(y, q, alpha=ALPHA):
    e = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(alpha*e, (alpha-1)*e)))


def coverage(y, q): return float(np.mean(np.asarray(y) <= np.asarray(q)))


def run():
    raw, audit = load_and_audit()
    feat = make_features(raw)
    # Retain original observed target rows. Initial lag warm-up rows are dropped only when all
    # required history is absent; XGBoost can handle later missing lags caused by genuine gaps.
    data = feat.dropna(subset=['cnt']).copy()
    # Explicit chronological windows: final 60 days test; prior 30 days calibration; prior 30 days validation.
    end = data.index.max()
    test_start = end - pd.Timedelta(days=60) + pd.Timedelta(hours=1)
    cal_start = test_start - pd.Timedelta(days=30)
    val_start = cal_start - pd.Timedelta(days=30)
    train = data[data.index < val_start]
    val = data[(data.index >= val_start) & (data.index < cal_start)]
    cal = data[(data.index >= cal_start) & (data.index < test_start)]
    test = data[data.index >= test_start]
    if min(len(train),len(val),len(cal),len(test)) == 0: raise ValueError('One or more chronological partitions are empty.')
    Xtr, ytr = train[FEATURES], train.cnt
    Xv, yv = val[FEATURES], val.cnt
    Xc, yc = cal[FEATURES], cal.cnt
    Xt, yt = test[FEATURES], test.cnt
    # Models: simple seasonal naive, RF point model, XGB point model, XGB 90th quantile model.
    preds = {}
    preds['Seasonal naive (168h lag)'] = Xt['lag_168h'].fillna(Xt['lag_24h']).fillna(ytr.median()).to_numpy()
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.8, n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)
    preds['Random Forest'] = rf.predict(Xt)
    xgb_point = XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.03,
        max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5, random_state=42, n_jobs=-1)
    xgb_point.fit(Xtr, ytr)
    preds['XGBoost point'] = xgb_point.predict(Xt)
    xgb_q = XGBRegressor(objective='reg:quantileerror', quantile_alpha=ALPHA, n_estimators=500,
        learning_rate=0.03, max_depth=5, min_child_weight=5, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=5, random_state=42, n_jobs=-1)
    xgb_q.fit(Xtr, ytr)
    q_val = xgb_q.predict(Xv)
    q_cal = xgb_q.predict(Xc)
    q_test = xgb_q.predict(Xt)
    # Calibration offset estimated only on separate calibration window; test is untouched.
    offset = float(np.quantile(yc.to_numpy()-q_cal, ALPHA, method='higher'))
    q_calibrated = np.maximum(0, q_test + offset)
    # Metrics: point-model MAE/RMSE; quantile metrics only for quantile forecasts.
    rows=[]
    for name,p in preds.items():
        rows.append({'model':name,'MAE':mean_absolute_error(yt,p),'RMSE':np.sqrt(mean_squared_error(yt,p)),
                     'Pinball_90':np.nan,'Coverage_90':np.nan})
    rows.append({'model':'XGBoost quantile 90% (raw)','MAE':mean_absolute_error(yt,q_test),
                 'RMSE':np.sqrt(mean_squared_error(yt,q_test)),'Pinball_90':pinball(yt,q_test), 'Coverage_90':coverage(yt,q_test)})
    rows.append({'model':'XGBoost quantile 90% (calibrated)','MAE':mean_absolute_error(yt,q_calibrated),
                 'RMSE':np.sqrt(mean_squared_error(yt,q_calibrated)),'Pinball_90':pinball(yt,q_calibrated), 'Coverage_90':coverage(yt,q_calibrated)})
    metrics=pd.DataFrame(rows)
    metrics.to_csv(OUT/'model_metrics.csv',index=False)
    # Detailed hourly coverage for raw and calibrated quantile predictions.
    hr=pd.DataFrame({'hour':yt.index.hour,'actual':yt.to_numpy(),'q_raw':q_test,'q_cal':q_calibrated})
    hourly=hr.groupby('hour').apply(lambda g: pd.Series({'raw_coverage':(g.actual<=g.q_raw).mean(),
            'calibrated_coverage':(g.actual<=g.q_cal).mean(),'actual_mean':g.actual.mean(),
            'raw_quantile_mean':g.q_raw.mean(),'n':len(g)}), include_groups=False).reset_index()
    hourly.to_csv(OUT/'hourly_coverage.csv',index=False)
    # Prediction export
    result=pd.DataFrame({'actual':yt,'seasonal_naive':preds['Seasonal naive (168h lag)'],
                         'rf_prediction':preds['Random Forest'],'xgb_point':preds['XGBoost point'],
                         'xgb_q90_raw':q_test,'xgb_q90_calibrated':q_calibrated})
    result.to_csv(OUT/'test_predictions.csv')
    # EDA plots
    daily=raw['cnt'].resample('D').sum(min_count=1)
    plt.figure(figsize=(13,4)); daily.plot(); plt.title('Daily bike rentals (observed hourly records)'); plt.ylabel('Rentals'); plt.tight_layout(); plt.savefig(FIG/'daily_demand.png',dpi=150); plt.close()
    plt.figure(figsize=(10,4)); raw.groupby(raw.index.hour).cnt.mean().plot(marker='o'); plt.title('Average rentals by hour'); plt.xlabel('Hour'); plt.ylabel('Average hourly rentals'); plt.tight_layout(); plt.savefig(FIG/'hourly_profile.png',dpi=150); plt.close()
    plt.figure(figsize=(10,4)); raw.groupby('weekday').cnt.mean().plot(kind='bar'); plt.title('Average rentals by weekday (0=Sunday in source data)'); plt.xlabel('weekday code'); plt.ylabel('Average hourly rentals'); plt.tight_layout(); plt.savefig(FIG/'weekday_profile.png',dpi=150); plt.close()
    plt.figure(figsize=(13,4)); result[['actual','xgb_point','xgb_q90_calibrated']].iloc[:168].plot(ax=plt.gca()); plt.title('First 7 days of final test: actual vs forecasts'); plt.ylabel('Hourly rentals'); plt.tight_layout(); plt.savefig(FIG/'test_forecast_example.png',dpi=150); plt.close()
    plt.figure(figsize=(10,4)); plt.plot(hourly.hour,hourly.raw_coverage,marker='o',label='Raw Q90'); plt.plot(hourly.hour,hourly.calibrated_coverage,marker='o',label='Calibrated Q90'); plt.axhline(.9,ls='--',c='black',label='90% target'); plt.xticks(range(24)); plt.xlabel('Hour'); plt.ylabel('Empirical coverage'); plt.legend(); plt.tight_layout(); plt.savefig(FIG/'hourly_quantile_coverage.png',dpi=150); plt.close()
    audit.update({'train_rows':len(train),'validation_rows':len(val),'calibration_rows':len(cal),'test_rows':len(test),
                  'train_end':str(train.index.max()),'validation_start':str(val.index.min()),'validation_end':str(val.index.max()),
                  'calibration_start':str(cal.index.min()),'calibration_end':str(cal.index.max()),
                  'test_start':str(test.index.min()),'test_end':str(test.index.max()),'calibration_offset':offset,
                  'feature_columns':FEATURES,'target':'cnt','forecast_protocol':'rolling one-hour-ahead; actual previous-hour demand is assumed observed at prediction time',
                  'weather_assumption':'Historical weather fields are used as if available at forecast time; operational deployment must use weather forecasts or remove these predictors.'})
    (OUT/'audit_and_design.json').write_text(json.dumps(audit,indent=2))
    # Summary markdown
    summary=f'''# Bike-sharing hourly demand analysis — results\n\n## Data audit\n- Original rows: {audit['rows']:,}; columns: {audit['columns']}\n- Missing cells: {audit['missing_cells']}; duplicate rows: {audit['duplicate_rows']}; duplicate timestamps: {audit['duplicate_timestamps']}\n- Missing hourly timestamps in the continuous calendar: {audit['missing_timestamps']} (not imputed; target values are not fabricated).\n- `cnt = casual + registered` for every supplied row: {audit['cnt_equals_casual_plus_registered']}. `casual` and `registered` are excluded from predictors to prevent direct target leakage.\n\n## Experimental design\nChronological partitions: training through {audit['train_end']}; validation {audit['validation_start']} to {audit['validation_end']}; calibration {audit['calibration_start']} to {audit['calibration_end']}; final test {audit['test_start']} to {audit['test_end']}. Final test is the last 60 calendar days. The evaluation is rolling one-hour-ahead: lag values are derived from prior observed target values.\n\n## Model metrics\n\n{metrics.to_markdown(index=False,floatfmt='.3f')}\n\nQuantile coverage is the fraction of test actuals at or below the predicted conditional 90th percentile. Calibration offset was {offset:.3f}, estimated only on the calibration partition. Calibration coverage is marginal and does not guarantee 90% coverage by hour or station.\n\n## Important interpretation and limitations\n- This dataset supports system-wide hourly rental-count forecasting. It has no station identifier, station inventory, station capacity, failed rental attempts, bike returns by station, or redistribution cost. It cannot directly identify station shortages or calculate the fleet size required.\n- Rentals (`cnt`) are observed transactions, not necessarily unconstrained latent demand: if bikes were unavailable, unmet demand may be unrecorded.\n- Weather predictors are historical observations. A real future forecast needs weather forecasts available at prediction time.\n- The lagged-demand protocol assumes the preceding actual hourly count is available for each rolling one-hour-ahead prediction. Multi-hour/day-ahead forecasting requires recursive forecasting or a different feature design.\n- The `instant` column is an index and is not used as a predictor. `casual` and `registered` are excluded because they sum exactly to `cnt`.\n\n## Business recommendation\nDo not recommend purchasing bikes solely from these system-wide rental forecasts. First instrument station-level inventory, failed rental attempts, returns, capacity, and rebalancing operations. Use this model as a system-level demand-planning baseline; pilot demand-informed redistribution only after station-level data becomes available.\n'''
    (OUT/'findings.md').write_text(summary)
    # Persist model and feature list for reproducibility
    import joblib
    joblib.dump({'model':xgb_q,'features':FEATURES,'alpha':ALPHA,'calibration_offset':offset}, ROOT/'models_xgb_quantile.joblib')
    print('AUDIT',json.dumps(audit,indent=2))
    print('\nMETRICS\n',metrics.to_string(index=False))
    print('\nHOURLY COVERAGE\n',hourly.to_string(index=False))
    print('\nCALIBRATION OFFSET',offset)

if __name__=='__main__': run()
