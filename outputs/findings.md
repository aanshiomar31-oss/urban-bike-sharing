# Bike-sharing hourly demand analysis — results

## Data audit
- Original rows: 17,379; columns: 18
- Missing cells: 0; duplicate rows: 0; duplicate timestamps: 0
- Missing hourly timestamps in the continuous calendar: 165 (not imputed; target values are not fabricated).
- `cnt = casual + registered` for every supplied row: True. `casual` and `registered` are excluded from predictors to prevent direct target leakage.

## Experimental design
Chronological partitions: training through 2012-09-02 23:00:00; validation 2012-09-03 00:00:00 to 2012-10-02 23:00:00; calibration 2012-10-03 00:00:00 to 2012-11-01 23:00:00; final test 2012-11-02 00:00:00 to 2012-12-31 23:00:00. Final test is the last 60 calendar days. The evaluation is rolling one-hour-ahead: lag values are derived from prior observed target values.

## Model metrics

| model                             |    MAE |    RMSE |   Pinball_90 |   Coverage_90 |
|:----------------------------------|-------:|--------:|-------------:|--------------:|
| Seasonal naive (168h lag)         | 73.648 | 120.637 |      nan     |       nan     |
| Random Forest                     | 25.022 |  40.186 |      nan     |       nan     |
| XGBoost point                     | 24.295 |  37.824 |      nan     |       nan     |
| XGBoost quantile 90% (raw)        | 43.104 |  63.589 |        7.147 |         0.846 |
| XGBoost quantile 90% (calibrated) | 68.650 |  84.451 |        7.837 |         0.966 |

Quantile coverage is the fraction of test actuals at or below the predicted conditional 90th percentile. Calibration offset was 30.208, estimated only on the calibration partition. Calibration coverage is marginal and does not guarantee 90% coverage by hour or station.

## Important interpretation and limitations
- This dataset supports system-wide hourly rental-count forecasting. It has no station identifier, station inventory, station capacity, failed rental attempts, bike returns by station, or redistribution cost. It cannot directly identify station shortages or calculate the fleet size required.
- Rentals (`cnt`) are observed transactions, not necessarily unconstrained latent demand: if bikes were unavailable, unmet demand may be unrecorded.
- Weather predictors are historical observations. A real future forecast needs weather forecasts available at prediction time.
- The lagged-demand protocol assumes the preceding actual hourly count is available for each rolling one-hour-ahead prediction. Multi-hour/day-ahead forecasting requires recursive forecasting or a different feature design.
- The `instant` column is an index and is not used as a predictor. `casual` and `registered` are excluded because they sum exactly to `cnt`.

## Business recommendation
Do not recommend purchasing bikes solely from these system-wide rental forecasts. First instrument station-level inventory, failed rental attempts, returns, capacity, and rebalancing operations. Use this model as a system-level demand-planning baseline; pilot demand-informed redistribution only after station-level data becomes available.
