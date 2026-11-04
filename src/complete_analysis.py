from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'hour.csv'; OUT=ROOT/'outputs'; FIG=OUT/'figures'; MODELS=ROOT/'models'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)


def load_data(path=DATA):
    d=pd.read_csv(path)
    d['datetime']=pd.to_datetime(d['dteday'],errors='coerce')+pd.to_timedelta(d['hr'],unit='h')
    d=d.sort_values('datetime').set_index('datetime')
    full=pd.date_range(d.index.min(),d.index.max(),freq='h')
    audit={'rows':len(d),'columns':len(d.columns),'missing_cells':int(d.isna().sum().sum()),'duplicate_rows':int(d.reset_index().duplicated().sum()),'duplicate_timestamps':int(d.index.duplicated().sum()),'negative_cnt':int((d.cnt<0).sum()),'cnt_equals_casual_plus_registered':bool((d.cnt==d.casual+d.registered).all()),'start':str(d.index.min()),'end':str(d.index.max()),'expected_hourly_timestamps':len(full),'observed_timestamps':d.index.nunique(),'missing_timestamps':len(full.difference(d.index))}
    return d,audit


def add_features(d):
    x=d.copy(); i=x.index
    x['hour_sin']=np.sin(2*np.pi*i.hour/24); x['hour_cos']=np.cos(2*np.pi*i.hour/24)
    x['weekday_sin']=np.sin(2*np.pi*i.dayofweek/7); x['weekday_cos']=np.cos(2*np.pi*i.dayofweek/7)
    x['month_sin']=np.sin(2*np.pi*(i.month-1)/12); x['month_cos']=np.cos(2*np.pi*(i.month-1)/12)
    x['is_weekend']=(i.dayofweek>=5).astype(int); x['is_peak_hour']=i.hour.isin([7,8,9,17,18,19]).astype(int)
    x['workingday_peak']=x['workingday']*x['is_peak_hour']
    # Exclude IDs and target-derived columns. Weather is historical observed weather (perfect-forecast assumption).
    return x

FEATURES=['hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos','is_weekend','is_peak_hour','workingday_peak','season','yr','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed']

def metrics(y,p): return {'MAE':mean_absolute_error(y,p),'RMSE':mean_squared_error(y,p)**.5,'R2':r2_score(y,p)}
def perm_test(a,b,iterations=20000,seed=42):
    # Two-sided permutation test for difference in independent day-level means.
    a=np.asarray(a,float); b=np.asarray(b,float); obs=abs(a.mean()-b.mean()); z=np.r_[a,b]; n=len(a); rng=np.random.default_rng(seed); hits=0
    for _ in range(iterations):
        rng.shuffle(z); diff=abs(z[:n].mean()-z[n:].mean()); hits += diff>=obs
    return float((hits+1)/(iterations+1)),float(a.mean()-b.mean())

def run():
    d,audit=load_data(); x=add_features(d)
    # Data split mirrors reference: train through Jun 30 2012; test Jul-Dec 2012.
    cut=pd.Timestamp('2012-07-01 00:00:00'); train=x.loc[x.index<cut].copy(); test=x.loc[x.index>=cut].copy()
    Xtr=train[FEATURES]; ytr=train.cnt; Xte=test[FEATURES]; yte=test.cnt
    models={
      'HistGradientBoosting':HistGradientBoostingRegressor(max_iter=250,learning_rate=.07,max_leaf_nodes=31,l2_regularization=1.0,random_state=42),
      'RandomForest':RandomForestRegressor(n_estimators=350,min_samples_leaf=2,max_features=.9,n_jobs=-1,random_state=42),
      'XGBoost':XGBRegressor(objective='reg:squarederror',n_estimators=500,learning_rate=.035,max_depth=5,min_child_weight=4,subsample=.85,colsample_bytree=.85,reg_lambda=5,random_state=42,n_jobs=-1)
    }
    # Seasonal baseline: same hour previous week, exact timestamp lookup. Fallback to previous day, then training median.
    hist=d.cnt
    base_idx=test.index-pd.Timedelta(hours=168); base=hist.reindex(base_idx).to_numpy()
    fallback=hist.reindex(test.index-pd.Timedelta(hours=24)).to_numpy(); base=np.where(np.isfinite(base),base,fallback); base=np.where(np.isfinite(base),base,float(ytr.median()))
    predictions={'Seasonal baseline':base}; fitted={}
    for name,m in models.items(): m.fit(Xtr,ytr); predictions[name]=m.predict(Xte); fitted[name]=m
    rows=[]
    for name,p in predictions.items(): rows.append({'model':name,**metrics(yte,p),'peak_MAE':mean_absolute_error(yte[test.index.hour.isin([7,8,9,17,18,19])],np.asarray(p)[test.index.hour.isin([7,8,9,17,18,19])]),'peak_bias_pred_minus_actual':float(np.mean(np.asarray(p)[test.index.hour.isin([7,8,9,17,18,19])]-yte.to_numpy()[test.index.hour.isin([7,8,9,17,18,19])]))})
    met=pd.DataFrame(rows).sort_values('MAE'); met.to_csv(OUT/'reference_comparable_model_metrics.csv',index=False)
    best=fitted['HistGradientBoosting']; joblib.dump({'model':best,'features':FEATURES,'model_name':'HistGradientBoosting','train_end':str(train.index.max())},MODELS/'primary_demand_model.joblib')
    pr=pd.DataFrame({'actual':yte,**{k:v for k,v in predictions.items()}},index=test.index); pr.to_csv(OUT/'reference_comparable_test_predictions.csv',index_label='datetime')
    # H1: per-day commute-window share, complete days only; compare working vs non-working days.
    daily=d.groupby(d.index.normalize()).agg(total=('cnt','sum'),workingday=('workingday','first'),n_hours=('cnt','count'))
    daily['commute']=d[d.index.hour.isin([7,8,9,17,18,19])].groupby(d[d.index.hour.isin([7,8,9,17,18,19])].index.normalize()).cnt.sum()
    daily['commute_share']=daily.commute/daily.total
    daily=daily[(daily.n_hours==24)&daily.total.gt(0)].dropna(subset=['commute_share'])
    a=daily.loc[daily.workingday==1,'commute_share']; b=daily.loc[daily.workingday==0,'commute_share']
    p_h1,diff_h1=perm_test(a,b)
    h1={'hypothesis':'Working days have a higher share of daily rentals in commute windows (07-09 and 17-19) than non-working days.','unit':'one complete day','workingday_mean_share':float(a.mean()),'nonworkingday_mean_share':float(b.mean()),'difference_percentage_points':float(diff_h1*100),'permutation_p_value':p_h1,'n_working_days':len(a),'n_nonworking_days':len(b),'note':'Commuting window includes hours 07,08,09,17,18,19; observational association, not causal.'}
    # H3 exploratory threshold: daily mean actual temperature >=32C (temp normalized to 41C max).
    daily_temp=d.assign(temp_c=d.temp*41).groupby(d.index.normalize()).agg(demand=('cnt','sum'),temp_c=('temp_c','mean'),n_hours=('cnt','count'))
    daily_temp=daily_temp[daily_temp.n_hours==24]
    hot=daily_temp.loc[daily_temp.temp_c>=32,'demand']; mild=daily_temp.loc[daily_temp.temp_c<32,'demand']
    if len(hot)>0 and len(mild)>0:
        mw=mannwhitneyu(hot,mild,alternative='two-sided'); h3={'hypothesis':'Exploratory: complete-day rental totals differ on days with mean temperature >=32C versus <32C.','threshold_celsius':32,'hot_day_n':len(hot),'other_day_n':len(mild),'hot_day_median_demand':float(hot.median()),'other_day_median_demand':float(mild.median()),'mann_whitney_u':float(mw.statistic),'p_value':float(mw.pvalue),'note':'Exploratory observational comparison; season and calendar confounding remain. Threshold is not causal and should be pre-specified before confirmatory testing.'}
    else: h3={'note':'Insufficient complete-day observations on one side of 32C threshold; test not estimable.'}
    (OUT/'hypothesis_results.json').write_text(json.dumps({'H1':h1,'H3':h3},indent=2))
    # Fleet scenarios: assumptions explicit; use observed hourly demand P95 as reference, not demand model forecast.
    p95=float(d.cnt.quantile(.95)); p99=float(d.cnt.quantile(.99)); peak_mean=float(d.loc[d.index.hour.isin([7,8,9,17,18,19]),'cnt'].mean())
    assumptions={'trip_duration_hours':[0.4,0.6,0.8],'spatial_buffer':[1.2,1.5,2.0],'maintenance_share':[0.05,0.10,0.15],'illustrative_fleet_size':2000}
    scen=[]
    for trip in assumptions['trip_duration_hours']:
      for buf in assumptions['spatial_buffer']:
       for maint in assumptions['maintenance_share']:
        fleet=p95*trip*buf/(1-maint)
        scen.append({'demand_basis':'observed hourly 95th percentile','p95_hourly_rentals':p95,'trip_duration_hours_assumed':trip,'spatial_buffer_assumed':buf,'maintenance_share_assumed':maint,'scenario_fleet_estimate':fleet,'illustrative_2000_fleet_minus_estimate':2000-fleet})
    pd.DataFrame(scen).to_csv(OUT/'fleet_scenarios.csv',index=False)
    fleet_summary={'observed_hourly_demand_p95':p95,'observed_hourly_demand_p99':p99,'observed_peak_window_mean_hourly_rentals':peak_mean,'formula':'fleet = hourly demand × assumed trip duration (hours) × spatial buffer ÷ (1 - maintenance share)','assumptions':assumptions,'warning':'Scenario arithmetic only. Demand counts are completed rentals, not concurrent bikes in use. The formula is a simplified illustrative scenario, not a validated fleet requirement. 2,000 bikes is fictional per supplied reference.'}
    (OUT/'fleet_summary.json').write_text(json.dumps(fleet_summary,indent=2))
    # Visuals
    plt.figure(figsize=(11,4)); d.groupby(d.index.hour).cnt.mean().plot(marker='o'); plt.xlabel('Hour'); plt.ylabel('Mean hourly rentals'); plt.title('Average hourly rentals'); plt.tight_layout(); plt.savefig(FIG/'01_hourly_profile.png',dpi=150); plt.close()
    plt.figure(figsize=(10,4)); daily.groupby('workingday').commute_share.mean().plot(kind='bar'); plt.xticks([0,1],['Non-working day','Working day'],rotation=0); plt.ylabel('Share of daily rentals in commute windows'); plt.title('H1: commute-window demand share'); plt.tight_layout(); plt.savefig(FIG/'02_commute_share.png',dpi=150); plt.close()
    plt.figure(figsize=(11,4)); d.groupby(d.season).cnt.mean().plot(kind='bar'); plt.xlabel('Season code'); plt.ylabel('Mean hourly rentals'); plt.title('Mean hourly rentals by season'); plt.tight_layout(); plt.savefig(FIG/'03_seasonal_demand.png',dpi=150); plt.close()
    plt.figure(figsize=(12,4)); pr[['actual','HistGradientBoosting','RandomForest','XGBoost']].iloc[:168].plot(ax=plt.gca()); plt.ylabel('Hourly rentals'); plt.title('First seven days of Jul-Dec 2012 test period'); plt.tight_layout(); plt.savefig(FIG/'04_test_forecasts.png',dpi=150); plt.close()
    audit.update({'train_rows':len(train),'test_rows':len(test),'train_end':str(train.index.max()),'test_start':str(test.index.min()),'test_end':str(test.index.max()),'features':FEATURES,'excluded_features':['instant','casual','registered','cnt'],'weather_assumption':'Recorded historical weather is treated as known at prediction time (perfect weather forecast assumption). Deployment must use weather forecasts or omit weather predictors.','forecast_protocol':'Direct one-hour conditional prediction from calendar and weather features; no target lags. Not a multi-day autoregressive forecast.'})
    (OUT/'reference_comparable_audit.json').write_text(json.dumps(audit,indent=2))
    report=f'''# Bike Sharing: More Bikes or Better Allocation?\n\n## Executive summary\nThis analysis forecasts citywide hourly completed rentals in Washington, D.C. It does not observe unmet demand or station inventory. The appropriate decision is to investigate demand-informed redistribution and fleet-health operations before purchasing more bikes, then validate through a station-level operational pilot. Redistribution is not proven as the root cause by this dataset.\n\n## Data and quality\n- Rows: {audit['rows']:,}; time range: {audit['start']} to {audit['end']}\n- Missing cells: {audit['missing_cells']}; duplicate rows: {audit['duplicate_rows']}; duplicate timestamps: {audit['duplicate_timestamps']}\n- Missing hourly timestamps: {audit['missing_timestamps']}; no artificial target values were imputed.\n- `cnt = casual + registered`: {audit['cnt_equals_casual_plus_registered']}. `casual`, `registered`, `instant` excluded from features.\n\n## Methodology\nTrain through 2012-06-30 and test on 2012-07-01 through 2012-12-31. Predictors are calendar, cyclical time, season and weather variables. No lagged target features are used, making the model suitable for a direct hourly conditional forecast when calendar/weather inputs are known. Historical weather is treated as a perfect weather forecast assumption.\n\n## Model results\n{met.to_markdown(index=False,floatfmt='.2f')}\n\n## Hypothesis results\n### H1: commuter concentration\nWorking-day mean commute-window share: {a.mean()*100:.1f}%; non-working-day: {b.mean()*100:.1f}%; difference: {diff_h1*100:.1f} percentage points; day-level permutation p={p_h1:.3g}. The commute windows are 07:00-09:59 and 17:00-19:59. This is an association, not causal evidence.\n\n### H3: temperature\n{json.dumps(h3,indent=2)}\n\n## Fleet scenario\nObserved hourly rental P95 is {p95:.1f}; P99 is {p99:.1f}. See `fleet_scenarios.csv` for assumptions and scenario calculations. The simplified formula is a sensitivity exercise, not a fleet requirement: completed hourly rentals are not simultaneous bikes in use, and actual trip duration, spatial imbalance, maintenance share and fleet size are not provided by this dataset.\n\n## Recommendation\nDo not buy bikes based solely on these citywide completed-rental forecasts. Instrument station-level dock/bike availability, trip origins/destinations and duration, failed/abandoned rental attempts, bike health, and rebalancing logs. Then pilot forecast-guided rebalancing against the current schedule, measuring empty-station minutes during commute peaks, unserved demand, completed rentals and operating cost.\n\n## Limitations\n1. No station stock, station capacity, origin/destination, trip duration, failed rentals, fleet size or rebalancing records.\n2. Completed rentals can understate latent demand when bikes are unavailable.\n3. Weather uses recorded observations, an optimistic availability assumption.\n4. 2011-2012 historical Washington, D.C. patterns may not transfer to another city or period.\n5. The 2,000-bike fleet in the reference is fictional; scenario outputs are not operational requirements.\n'''
    (OUT/'final_findings.md').write_text(report)
    print('AUDIT',json.dumps(audit,indent=2)); print('\nMETRICS\n',met.to_string(index=False)); print('\nH1',h1); print('\nH3',h3); print('\nFLEET',fleet_summary)

if __name__=='__main__': run()
