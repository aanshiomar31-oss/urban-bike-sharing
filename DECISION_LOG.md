# Decision log

1. Use `hour.csv` only, as selected for this iteration.
2. Preserve observed rows; do not interpolate missing hourly target values.
3. Exclude `instant`, `casual`, and `registered`; `casual + registered = cnt` exactly.
4. Use chronological train/test split matching the supplied reference: train through 2012-06-30; test 2012-07-01 to 2012-12-31.
5. Exclude target lags to make a direct hourly conditional forecast and avoid dependence on future actual targets for a multi-hour test horizon.
6. Compare seasonal baseline, Random Forest, XGBoost and HistGradientBoosting; choose on held-out MAE while reporting peak-hour error and bias.
7. Treat weather inputs as historical observations, explicitly noting the perfect-weather-forecast assumption.
8. Treat fleet arithmetic as sensitivity analysis only, not an actual procurement recommendation.
9. Recommend data collection and a controlled redistribution pilot before purchase decisions; redistribution is not proven as root cause.
