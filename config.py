"""Central configuration module for Urban Bike-Sharing Demand Forecasting and Fleet Optimization.
Contains all directory paths, feature lists, temporal partitions, and random seeds.
"""

from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "hour.csv"
if not RAW_DATA_PATH.exists():
    RAW_DATA_PATH = BASE_DIR / "hour.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
SQLITE_DB_PATH = DATA_DIR / "bikeshare.db"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MONTHLY_REPORTS_DIR = OUTPUTS_DIR / "monthly_reports"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Ensure runtime directories exist
for folder in [
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MONTHLY_REPORTS_DIR,
    FORECASTS_DIR,
    REPORTS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Commute and temporal window definitions
MORNING_PEAK_HOURS = [7, 8, 9]
EVENING_PEAK_HOURS = [17, 18, 19]
ALL_PEAK_HOURS = MORNING_PEAK_HOURS + EVENING_PEAK_HOURS
NIGHT_HOURS = [0, 1, 2, 3, 4, 5, 22, 23]
BUSINESS_HOURS = [9, 10, 11, 12, 13, 14, 15, 16, 17]

# Chronological partition dates
TRAIN_END = "2012-06-30 23:00:00"
TEST_START = "2012-07-01 00:00:00"
TEST_END = "2012-12-31 23:00:00"

# Target variables and exclusions
TARGET_COL = "cnt"
CASUAL_COL = "casual"
REGISTERED_COL = "registered"
ID_COL = "instant"
EXCLUDED_COLS = [ID_COL, CASUAL_COL, REGISTERED_COL, TARGET_COL, "dteday", "datetime"]

# Weather features
WEATHER_FEATURES = ["weathersit", "temp", "atemp", "hum", "windspeed"]

# Temporal features generated across the hierarchy (Year -> Quarter -> Month -> Week -> Day -> Hour)
HOURLY_TEMPORAL_FEATURES = [
    "hour",
    "hour_sin",
    "hour_cos",
    "morning_peak",
    "evening_peak",
    "is_peak_hour",
    "is_off_peak",
    "night_period",
    "business_hours",
]

DAILY_TEMPORAL_FEATURES = [
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "is_weekend",
    "is_workingday",
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
]

WEEKLY_TEMPORAL_FEATURES = [
    "week_of_year",
    "week_of_month",
    "weekday_index",
    "weekend_indicator",
]

MONTHLY_TEMPORAL_FEATURES = [
    "month",
    "month_sin",
    "month_cos",
    "quarter",
    "is_month_start",
    "is_month_end",
    "days_in_month",
]

ANNUAL_SEASONAL_FEATURES = [
    "year",
    "season",
    "annual_trend",
]

INTERACTION_FEATURES = [
    "hour_x_workingday",
    "hour_x_weekend",
    "month_x_hour",
    "season_x_hour",
    "weather_x_workingday",
    "temp_x_weekend",
    "peak_hour_x_workingday",
]

# Combined direct forecasting features (Experiment A)
DIRECT_FEATURES = (
    HOURLY_TEMPORAL_FEATURES
    + DAILY_TEMPORAL_FEATURES
    + WEEKLY_TEMPORAL_FEATURES
    + MONTHLY_TEMPORAL_FEATURES
    + ANNUAL_SEASONAL_FEATURES
    + INTERACTION_FEATURES
    + WEATHER_FEATURES
)

# Autoregressive features (Experiment B)
LAG_HOURS = [1, 2, 3, 24, 48, 168, 336]
LAG_COLUMNS = [f"lag_{h}" for h in LAG_HOURS]

ROLLING_WINDOWS = [3, 6, 24, 72, 168]
ROLLING_COLUMNS = (
    [f"rolling_mean_{w}h" for w in ROLLING_WINDOWS]
    + ["rolling_std_24h", "rolling_std_168h", "rolling_median_24h", "rolling_max_24h", "rolling_min_24h"]
)

AUTOREGRESSIVE_FEATURES = DIRECT_FEATURES + LAG_COLUMNS + ROLLING_COLUMNS

# Fleet sensitivity defaults
DEFAULT_TRIP_DURATION = 0.5  # 30 minutes
DEFAULT_SPATIAL_BUFFER = 1.5
DEFAULT_MAINTENANCE_SHARE = 0.10
ILLUSTRATIVE_FLEET_SIZE = 2000
