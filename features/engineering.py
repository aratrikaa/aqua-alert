"""
features/engineering.py
-----------------------
Transforms raw monthly groundwater / rainfall time-series into a
feature matrix suitable for classification.

Features computed per district (snapshot at the most-recent month):
  - rainfall_lag_30d   : rolling 1-month rainfall sum
  - rainfall_lag_60d   : rolling 2-month rainfall sum
  - rainfall_lag_90d   : rolling 3-month rainfall sum
  - premonsoon_depth   : mean water depth in Mar-May (worst case)
  - postmonsoon_depth  : mean water depth in Oct-Nov (after recharge)
  - monsoon_recovery   : postmonsoon_depth - premonsoon_depth (negative = recovery)
  - annual_trend       : linear trend slope (m/year) over last 3 years
  - extraction_rate    : normalised pumping proxy (0-1)
  - current_depth      : most recent water table depth (mbgl)
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress


def _annual_trend(series: pd.Series) -> float:
    """
    Fit a linear trend to the last 36 months of depth readings.
    Returns slope in metres per year (positive = declining).
    """
    s = series.dropna().tail(36)
    if len(s) < 6:
        return 0.0
    x = np.arange(len(s))
    slope, *_ = linregress(x, s.values)
    return round(slope * 12, 4)   # convert months → years


def _label_from_profile(district_id: str, districts_meta: list[dict]) -> str:
    """Look up the ground-truth risk profile from metadata (used for training)."""
    profile_map = {"critical": "Critical", "moderate": "Moderate", "safe": "Safe"}
    for d in districts_meta:
        if d["id"] == district_id:
            return profile_map[d["risk_profile"]]
    return "Moderate"


def build_feature_matrix(df: pd.DataFrame,
                         districts_meta: list[dict]) -> pd.DataFrame:
    """
    Given the full monthly time-series DataFrame, collapse each district
    into a single feature row and attach a risk label for training.

    Returns a DataFrame with one row per district.
    """
    rows = []

    for dist_id, grp in df.groupby("district_id"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # ── Rainfall lags ────────────────────────────────────────────────────
        rainfall_lag_30 = grp["rainfall_mm"].iloc[-1:].sum()
        rainfall_lag_60 = grp["rainfall_mm"].iloc[-2:].sum()
        rainfall_lag_90 = grp["rainfall_mm"].iloc[-3:].sum()

        # ── Seasonal aggregates ──────────────────────────────────────────────
        grp["month"] = grp["date"].dt.month
        premonsoon  = grp[grp["month"].isin([3, 4, 5])]["water_depth_mbgl"].mean()
        postmonsoon = grp[grp["month"].isin([10, 11])]["water_depth_mbgl"].mean()
        monsoon_recovery = postmonsoon - premonsoon   # negative = table rose = good

        # ── Trend ────────────────────────────────────────────────────────────
        annual_trend = _annual_trend(grp["water_depth_mbgl"])

        # ── Latest readings ──────────────────────────────────────────────────
        current_depth    = grp["water_depth_mbgl"].iloc[-1]
        extraction_rate  = grp["extraction_rate"].iloc[-1]

        # ── Label ────────────────────────────────────────────────────────────
        label = _label_from_profile(dist_id, districts_meta)

        rows.append({
            "district_id":       dist_id,
            "district_name":     grp["district_name"].iloc[0],
            "state":             grp["state"].iloc[0],
            "rainfall_lag_30d":  round(rainfall_lag_30, 2),
            "rainfall_lag_60d":  round(rainfall_lag_60, 2),
            "rainfall_lag_90d":  round(rainfall_lag_90, 2),
            "premonsoon_depth":  round(premonsoon, 3),
            "postmonsoon_depth": round(postmonsoon, 3),
            "monsoon_recovery":  round(monsoon_recovery, 3),
            "annual_trend":      annual_trend,
            "extraction_rate":   extraction_rate,
            "current_depth":     round(current_depth, 3),
            "risk_label":        label,
        })

    return pd.DataFrame(rows)


# Feature columns used by the model (order matters for SHAP alignment)
FEATURE_COLS = [
    "rainfall_lag_30d",
    "rainfall_lag_60d",
    "rainfall_lag_90d",
    "premonsoon_depth",
    "postmonsoon_depth",
    "monsoon_recovery",
    "annual_trend",
    "extraction_rate",
    "current_depth",
]

# Human-readable labels for SHAP explanation text
FEATURE_LABELS = {
    "rainfall_lag_30d":  "last-month rainfall",
    "rainfall_lag_60d":  "2-month cumulative rainfall",
    "rainfall_lag_90d":  "3-month cumulative rainfall",
    "premonsoon_depth":  "pre-monsoon water table depth",
    "postmonsoon_depth": "post-monsoon water table depth",
    "monsoon_recovery":  "monsoon recharge recovery",
    "annual_trend":      "annual depletion trend",
    "extraction_rate":   "groundwater extraction rate",
    "current_depth":     "current water table depth",
}
