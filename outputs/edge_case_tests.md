# Edge-case test coverage

Automated checks in `tests/test_pipeline.py` verify:
- Original row count and key data-quality conditions.
- Detection of 165 absent hourly timestamps.
- Identity `cnt == casual + registered`.
- Exclusion of target and target-component columns from predictors.
- Exact elapsed-hour lag construction, including preserving missing lag values when a timestamp is absent.

Additional operational safeguards in the pipeline:
- Negative target values are counted in the audit; no negative values are present in this dataset.
- Missing timestamps are not imputed with fabricated target values.
- XGBoost handles missing lag-feature values natively.
- The Streamlit prototype gives a clear limitation response to station-level and fleet-purchase questions.

Not yet covered by automated integration tests: model-serving failures, out-of-distribution weather inputs, unseen station IDs (station IDs do not exist in this dataset), and live weather API failures. These are deployment-stage tests, not claims of completed validation.
