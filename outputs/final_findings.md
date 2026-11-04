# Bike Sharing: More Bikes or Better Allocation?

## Executive summary
This analysis forecasts citywide hourly completed rentals in Washington, D.C. It does not observe unmet demand or station inventory. The appropriate decision is to investigate demand-informed redistribution and fleet-health operations before purchasing more bikes, then validate through a station-level operational pilot. Redistribution is not proven as the root cause by this dataset.

## Data and quality
- Rows: 17,379; time range: 2011-01-01 00:00:00 to 2012-12-31 23:00:00
- Missing cells: 0; duplicate rows: 0; duplicate timestamps: 0
- Missing hourly timestamps: 165; no artificial target values were imputed.
- `cnt = casual + registered`: True. `casual`, `registered`, `instant` excluded from features.

## Methodology
Train through 2012-06-30 and test on 2012-07-01 through 2012-12-31. Predictors are calendar, cyclical time, season and weather variables. No lagged target features are used, making the model suitable for a direct hourly conditional forecast when calendar/weather inputs are known. Historical weather is treated as a perfect weather forecast assumption.

## Model results
| model                |   MAE |   RMSE |   R2 |   peak_MAE |   peak_bias_pred_minus_actual |
|:---------------------|------:|-------:|-----:|-----------:|------------------------------:|
| HistGradientBoosting | 42.46 |  65.03 | 0.91 |      71.84 |                        -30.69 |
| XGBoost              | 45.82 |  68.44 | 0.90 |      77.50 |                        -36.55 |
| RandomForest         | 47.89 |  72.68 | 0.89 |      84.09 |                        -43.69 |
| Seasonal baseline    | 63.60 | 108.81 | 0.75 |     105.69 |                         12.61 |

## Hypothesis results
### H1: commuter concentration
Working-day mean commute-window share: 51.6%; non-working-day: 26.2%; difference: 25.4 percentage points; day-level permutation p=5e-05. The commute windows are 07:00-09:59 and 17:00-19:59. This is an association, not causal evidence.

### H3: temperature
{
  "hypothesis": "Exploratory: complete-day rental totals differ on days with mean temperature >=32C versus <32C.",
  "threshold_celsius": 32,
  "hot_day_n": 25,
  "other_day_n": 630,
  "hot_day_median_demand": 5463.0,
  "other_day_median_demand": 4678.0,
  "mann_whitney_u": 9041.5,
  "p_value": 0.20889845838439325,
  "note": "Exploratory observational comparison; season and calendar confounding remain. Threshold is not causal and should be pre-specified before confirmatory testing."
}

## Fleet scenario
Observed hourly rental P95 is 563.1; P99 is 782.2. See `fleet_scenarios.csv` for assumptions and scenario calculations. The simplified formula is a sensitivity exercise, not a fleet requirement: completed hourly rentals are not simultaneous bikes in use, and actual trip duration, spatial imbalance, maintenance share and fleet size are not provided by this dataset.

## Recommendation
Do not buy bikes based solely on these citywide completed-rental forecasts. Instrument station-level dock/bike availability, trip origins/destinations and duration, failed/abandoned rental attempts, bike health, and rebalancing logs. Then pilot forecast-guided rebalancing against the current schedule, measuring empty-station minutes during commute peaks, unserved demand, completed rentals and operating cost.

## Limitations
1. No station stock, station capacity, origin/destination, trip duration, failed rentals, fleet size or rebalancing records.
2. Completed rentals can understate latent demand when bikes are unavailable.
3. Weather uses recorded observations, an optimistic availability assumption.
4. 2011-2012 historical Washington, D.C. patterns may not transfer to another city or period.
5. The 2,000-bike fleet in the reference is fictional; scenario outputs are not operational requirements.
