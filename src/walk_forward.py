"""Expanding-window chronological walk-forward monthly validation module.
Evaluates model stability, peak-hour bias, and error distributions across sequential monthly folds.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import DIRECT_FEATURES, TARGET_COL, METRICS_DIR, RANDOM_SEED
from src.evaluation import calculate_regression_metrics


def run_walk_forward_validation(df: pd.DataFrame, start_validation_month: str = "2012-01", end_validation_month: str = "2012-06") -> pd.DataFrame:
    """Executes chronological monthly expanding-window validation from start_month through end_month.
    
    Fold Architecture:
    - Fold 1: Train [2011-01 .. 2011-12] -> Validate on 2012-01
    - Fold 2: Train [2011-01 .. 2012-01] -> Validate on 2012-02
    ...
    - Preserves historical continuity and evaluates performance on true out-of-time monthly blocks.
    """
    df_sorted = df.sort_index()
    # List of unique year-months in validation range
    ym_series = df_sorted.index.to_period("M").astype(str)
    unique_months = sorted(ym_series.unique())
    val_months = [m for m in unique_months if start_validation_month <= m <= end_validation_month]

    results = []

    for val_m in val_months:
        train_mask = ym_series < val_m
        val_mask = ym_series == val_m

        train_fold = df_sorted[train_mask]
        val_fold = df_sorted[val_mask]

        if len(train_fold) < 1000 or len(val_fold) < 100:
            continue

        X_tr = train_fold[DIRECT_FEATURES]
        y_tr = train_fold[TARGET_COL]
        X_val = val_fold[DIRECT_FEATURES]
        y_val = val_fold[TARGET_COL]

        model = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=35,
            random_state=RANDOM_SEED
        )
        model.fit(X_tr, y_tr)
        pred = np.clip(model.predict(X_val), 0, None)

        m = calculate_regression_metrics(y_val, pred, val_fold.index.hour)
        results.append({
            "validation_month": val_m,
            "train_observations": len(train_fold),
            "val_observations": len(val_fold),
            **m
        })

    wf_df = pd.DataFrame(results)
    wf_df.to_csv(METRICS_DIR / "walk_forward_validation_metrics.csv", index=False)
    return wf_df
