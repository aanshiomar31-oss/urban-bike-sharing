# Final Execution Report: Urban Bike-Sharing Demand Forecasting

**Status:** SUCCESS | **Runtime:** 56.64s | **Dataset Rows:** 17,379

## 1. Key Modeling Results (Experiment A - Direct Forecasting)
- **Champion Model:** `HistGradientBoosting`
- **Test MAE:** 41.95 rentals/hour
- **Test RMSE:** 65.25
- **Test R²:** 0.9118
- **Commute Peak-Hour MAE:** 71.68
- **Peak-Hour Bias:** -18.40 (Under-prediction during rush hour)

## 2. Statistical Findings
- **Hypothesis 1 (Commuter Concentration):** Working-day commute share = 51.63% vs Non-working day = 26.25% (+25.39% difference, p = 4.9998e-05).
- Over 51% of all weekday demand is compressed into the 6 peak commute hours.

## 3. Fleet Optimization & Strategic Verdict
- **Estimated Active Fleet for P95 Demand:** ~470 bikes.
- **Surplus under 2,000-bike fleet:** ~1530 bikes.
- **Executive Recommendation:** DO NOT BUY MORE BIKES. Allocate budget to instrument station dock telemetry and execute predictive van rebalancing.
