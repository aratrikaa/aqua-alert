"""
data/generate_data.py
--------------------
Generates realistic synthetic groundwater level and rainfall time-series
data for Indian districts. The patterns mimic real CGWB / IMD observations:
  - Monsoon season (Jun-Sep) brings water table recovery.
  - Pre-monsoon (Mar-May) shows maximum depletion.
  - High extraction zones (Punjab, Haryana) show secular decline trend.
  - Rajasthan has arid patterns with low recharge.
"""

import numpy as np
import pandas as pd
import json
import os

# ── District Metadata ──────────────────────────────────────────────────────────
DISTRICTS = [
    # Punjab (high extraction, critical groundwater stress)
    {"id": "PB_LDH", "name": "Ludhiana",      "state": "Punjab",       "lat": 30.9010, "lon": 75.8573, "risk_profile": "critical"},
    {"id": "PB_BTD", "name": "Bathinda",       "state": "Punjab",       "lat": 30.2110, "lon": 74.9455, "risk_profile": "critical"},
    {"id": "PB_SAS", "name": "SAS Nagar",      "state": "Punjab",       "lat": 30.7046, "lon": 76.7179, "risk_profile": "moderate"},
    {"id": "PB_AMR", "name": "Amritsar",       "state": "Punjab",       "lat": 31.6340, "lon": 74.8723, "risk_profile": "critical"},
    {"id": "PB_FZR", "name": "Fazilka",        "state": "Punjab",       "lat": 30.4024, "lon": 74.0296, "risk_profile": "critical"},

    # Haryana (over-exploited aquifers, dark zones)
    {"id": "HR_KRN", "name": "Karnal",         "state": "Haryana",      "lat": 29.6857, "lon": 76.9905, "risk_profile": "critical"},
    {"id": "HR_KTL", "name": "Kaithal",        "state": "Haryana",      "lat": 29.8010, "lon": 76.3986, "risk_profile": "critical"},
    {"id": "HR_AMB", "name": "Ambala",         "state": "Haryana",      "lat": 30.3782, "lon": 76.7767, "risk_profile": "moderate"},
    {"id": "HR_FRD", "name": "Faridabad",      "state": "Haryana",      "lat": 28.4089, "lon": 77.3178, "risk_profile": "moderate"},
    {"id": "HR_MEW", "name": "Mewat",          "state": "Haryana",      "lat": 28.1000, "lon": 77.0000, "risk_profile": "safe"},

    # Rajasthan (arid, low recharge)
    {"id": "RJ_JPR", "name": "Jaipur",         "state": "Rajasthan",    "lat": 26.9124, "lon": 75.7873, "risk_profile": "moderate"},
    {"id": "RJ_JDH", "name": "Jodhpur",        "state": "Rajasthan",    "lat": 26.2389, "lon": 73.0243, "risk_profile": "critical"},
    {"id": "RJ_AJM", "name": "Ajmer",          "state": "Rajasthan",    "lat": 26.4499, "lon": 74.6399, "risk_profile": "moderate"},
    {"id": "RJ_BKN", "name": "Bikaner",        "state": "Rajasthan",    "lat": 28.0229, "lon": 73.3119, "risk_profile": "critical"},
    {"id": "RJ_ALW", "name": "Alwar",          "state": "Rajasthan",    "lat": 27.5530, "lon": 76.6346, "risk_profile": "moderate"},

    # Maharashtra (mixed — drought-prone interior vs. wetter coast)
    {"id": "MH_PNE", "name": "Pune",           "state": "Maharashtra",  "lat": 18.5204, "lon": 73.8567, "risk_profile": "safe"},
    {"id": "MH_NGP", "name": "Nagpur",         "state": "Maharashtra",  "lat": 21.1458, "lon": 79.0882, "risk_profile": "safe"},
    {"id": "MH_LTR", "name": "Latur",          "state": "Maharashtra",  "lat": 18.4088, "lon": 76.5604, "risk_profile": "moderate"},
    {"id": "MH_AHM", "name": "Ahmednagar",     "state": "Maharashtra",  "lat": 19.0952, "lon": 74.7496, "risk_profile": "moderate"},
    {"id": "MH_OSM", "name": "Osmanabad",      "state": "Maharashtra",  "lat": 18.1860, "lon": 76.0390, "risk_profile": "critical"},

    # Gujarat (coastal and semi-arid stress)
    {"id": "GJ_AHM", "name": "Ahmedabad",      "state": "Gujarat",      "lat": 23.0225, "lon": 72.5714, "risk_profile": "moderate"},
    {"id": "GJ_RJK", "name": "Rajkot",         "state": "Gujarat",      "lat": 22.3039, "lon": 70.8022, "risk_profile": "critical"},
    {"id": "GJ_SRT", "name": "Surat",          "state": "Gujarat",      "lat": 21.1702, "lon": 72.8311, "risk_profile": "safe"},
    {"id": "GJ_BVN", "name": "Banaskantha",    "state": "Gujarat",      "lat": 24.1743, "lon": 72.4381, "risk_profile": "critical"},
    {"id": "GJ_AND", "name": "Anand",          "state": "Gujarat",      "lat": 22.5645, "lon": 72.9289, "risk_profile": "moderate"},

    # Uttar Pradesh (Indo-Gangetic plain, mixed)
    {"id": "UP_LKO", "name": "Lucknow",        "state": "Uttar Pradesh","lat": 26.8467, "lon": 80.9462, "risk_profile": "moderate"},
    {"id": "UP_AGR", "name": "Agra",           "state": "Uttar Pradesh","lat": 27.1767, "lon": 78.0081, "risk_profile": "critical"},
    {"id": "UP_MRT", "name": "Meerut",         "state": "Uttar Pradesh","lat": 28.9845, "lon": 77.7064, "risk_profile": "critical"},
    {"id": "UP_VNS", "name": "Varanasi",       "state": "Uttar Pradesh","lat": 25.3176, "lon": 82.9739, "risk_profile": "safe"},
    {"id": "UP_GZB", "name": "Ghaziabad",      "state": "Uttar Pradesh","lat": 28.6692, "lon": 77.4538, "risk_profile": "moderate"},
]

# Profile parameters: (base_depth_m, trend_m_per_yr, monsoon_recovery_m, noise_std)
RISK_PROFILES = {
    "critical": {
        "base_depth":      18.0,   # depth to water table (mbgl) — higher = deeper = worse
        "trend_per_year":   1.2,   # annual decline in metres
        "monsoon_recovery": 1.8,   # recovery during monsoon
        "noise_std":        0.5,
        "annual_rainfall":  450,   # mm
        "extraction_rate":  0.85,  # normalised 0-1
    },
    "moderate": {
        "base_depth":      10.0,
        "trend_per_year":   0.5,
        "monsoon_recovery": 2.5,
        "noise_std":        0.4,
        "annual_rainfall":  700,
        "extraction_rate":  0.55,
    },
    "safe": {
        "base_depth":       5.0,
        "trend_per_year":  -0.1,   # slight improvement
        "monsoon_recovery": 3.5,
        "noise_std":        0.35,
        "annual_rainfall": 1000,
        "extraction_rate":  0.30,
    },
}


def _monsoon_factor(month: int) -> float:
    """Return a seasonal multiplier for water table depth (0 = dry, 1 = peak wet)."""
    # Peak monsoon: July-August  |  driest: April-May
    seasonal = {1: 0.3, 2: 0.2, 3: 0.1, 4: 0.0, 5: 0.05,
                6: 0.4, 7: 0.85, 8: 1.0, 9: 0.9, 10: 0.6, 11: 0.4, 12: 0.3}
    return seasonal[month]


def _rainfall_for_month(annual_mm: float, month: int, rng: np.random.Generator) -> float:
    """Distribute annual rainfall into monthly values with monsoon skew."""
    monthly_share = {
        1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.02,
        6: 0.12, 7: 0.28, 8: 0.30, 9: 0.14, 10: 0.05,
        11: 0.03, 12: 0.02
    }
    base = annual_mm * monthly_share[month]
    return max(0.0, rng.normal(base, base * 0.25))


def generate_time_series(district: dict, start_year: int = 2019,
                         end_year: int = 2024) -> pd.DataFrame:
    """
    Build a monthly time series of groundwater depth (mbgl) and rainfall (mm)
    for a single district over [start_year, end_year].
    """
    rng = np.random.default_rng(seed=hash(district["id"]) % 2**31)
    profile = RISK_PROFILES[district["risk_profile"]]

    records = []
    base = profile["base_depth"]

    for year in range(start_year, end_year + 1):
        years_elapsed = year - start_year
        for month in range(1, 13):
            # Secular declining trend
            trend_component = profile["trend_per_year"] * (years_elapsed + (month - 1) / 12)

            # Seasonal: monsoon recharges the aquifer → depth decreases
            seasonal_component = -profile["monsoon_recovery"] * _monsoon_factor(month)

            # Noise
            noise = rng.normal(0, profile["noise_std"])

            depth = base + trend_component + seasonal_component + noise
            depth = max(1.0, depth)  # physical lower bound

            rainfall = _rainfall_for_month(profile["annual_rainfall"], month, rng)

            records.append({
                "date":             pd.Timestamp(year=year, month=month, day=1),
                "district_id":      district["id"],
                "district_name":    district["name"],
                "state":            district["state"],
                "water_depth_mbgl": round(depth, 3),   # metres below ground level
                "rainfall_mm":      round(rainfall, 2),
                "extraction_rate":  profile["extraction_rate"],
            })

    return pd.DataFrame(records)


def generate_all_data(output_dir: str = "data") -> tuple[pd.DataFrame, list[dict]]:
    """
    Generate time-series for all districts, save to CSV/JSON, and return
    the combined DataFrame plus district metadata list.
    """
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    for d in DISTRICTS:
        df = generate_time_series(d)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "groundwater_timeseries.csv"), index=False)

    with open(os.path.join(output_dir, "districts.json"), "w") as f:
        json.dump(DISTRICTS, f, indent=2)

    print(f"[data] Generated {len(combined)} monthly records for {len(DISTRICTS)} districts.")
    return combined, DISTRICTS


if __name__ == "__main__":
    generate_all_data()
