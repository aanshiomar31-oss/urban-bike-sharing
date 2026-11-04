"""Historical demand and autoregressive feature engineering module.
Implements leak-free lag features (lag_1..336) and rolling statistics (rolling_mean_3h..168h)
using continuous hourly calendar scaffolds and strict shift(1) prior-observation constraints.
"""

import numpy as np
import pandas as pd
from config import TARGET_COL, LAG_HOURS, ROLLING_WINDOWS


def create_lag_features(df: pd.DataFrame, target_series: pd.Series = None) -> pd.DataFrame:
    """Computes exact elapsed-time lag and rolling features without future-target leakage.
    
    Leakage Prevention Design:
    1. A complete, un-interrupted hourly calendar scaffold is used to compute exact elapsed
       hour shifts (e.g., lag_24h is strictly the value 24 clock hours ago, not 24 rows ago).
    2. All rolling statistics apply .shift(1) so the current hour's target is NEVER included:
       rolling_mean_24h(t) = Mean(cnt[t-24 : t-1]).
    3. Missing timestamps are treated as NaN; no future observations are backfilled.
    """
    out = df.copy()
    idx = out.index

    # Build continuous chronological scaffold
    full_idx = pd.date_range(idx.min(), idx.max(), freq="h")
    
    if target_series is None:
        target_series = out[TARGET_COL]

    # Reindex target onto continuous hourly grid
    cnt_scaffold = target_series.reindex(full_idx)

    # ----------------------------------------------------
    # 1. Exact Elapsed-Hour Lag Features
    # ----------------------------------------------------
    for h in LAG_HOURS:
        # shift(h) on full grid gives the observation exactly h hours prior
        shifted = cnt_scaffold.shift(h)
        out[f"lag_{h}"] = shifted.reindex(idx).to_numpy()

    # ----------------------------------------------------
    # 2. Rolling Statistics (Computed on shifted prior observations)
    # ----------------------------------------------------
    prior_scaffold = cnt_scaffold.shift(1)  # Strictly previous hours only

    # Rolling means across windows
    for w in ROLLING_WINDOWS:
        roll_mean = prior_scaffold.rolling(window=w, min_periods=max(1, w // 3)).mean()
        out[f"rolling_mean_{w}h"] = roll_mean.reindex(idx).to_numpy()

    # Rolling standard deviations
    out["rolling_std_24h"] = prior_scaffold.rolling(24, min_periods=4).std().reindex(idx).to_numpy()
    out["rolling_std_168h"] = prior_scaffold.rolling(168, min_periods=12).std().reindex(idx).to_numpy()

    # Rolling median, max, and min (24h)
    out["rolling_median_24h"] = prior_scaffold.rolling(24, min_periods=4).median().reindex(idx).to_numpy()
    out["rolling_max_24h"] = prior_scaffold.rolling(24, min_periods=4).max().reindex(idx).to_numpy()
    out["rolling_min_24h"] = prior_scaffold.rolling(24, min_periods=4).min().reindex(idx).to_numpy()

    return out
