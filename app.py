# -*- coding: utf-8 -*-
"""
app.py  --  AquaAlert main entry point
=======================================
Run with:  python app.py

The startup sequence:
  1. Generate (or load) synthetic groundwater / rainfall time-series.
  2. Build the feature matrix.
  3. Train (or load cached) classifier + compute SHAP explanations.
  4. Store results in Flask app config so API routes can serve them.
  5. Start the development server on http://localhost:5000
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Force UTF-8 output on Windows so Unicode chars in sub-modules don't crash
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from flask import Flask, send_from_directory

# -- Ensure project root is on sys.path ------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.generate_data   import generate_all_data, DISTRICTS
from features.engineering import build_feature_matrix, FEATURE_COLS
from model.train          import train, load_model, MODEL_PATH, ENCODER_PATH, LABEL_ORDER
from model.explain        import compute_shap_explanation
from api.routes           import api_bp

# -- Flask app -------------------------------------------------------------
app = Flask(__name__, static_folder="frontend", static_url_path="")
app.register_blueprint(api_bp)


# -- Startup pipeline -------------------------------------------------------

def _build_results(feature_df, model, le, ts_df):
    """
    Run predictions + SHAP explanations for every district and
    return a list of result dicts ready to be served via the API.
    """
    results = []

    # Build district coordinate map
    coord_map = {d["id"]: {"lat": d["lat"], "lon": d["lon"]} for d in DISTRICTS}

    X = feature_df[FEATURE_COLS].values

    for row_idx, (_, row) in enumerate(feature_df.iterrows()):
        dist_id = row["district_id"]
        x_row   = X[row_idx]

        # Predict
        proba      = model.predict_proba(x_row.reshape(1, -1))[0]
        class_idx  = int(np.argmax(proba))
        label      = LABEL_ORDER[class_idx]
        confidence = round(float(proba[class_idx]) * 100, 1)
        risk_score = round(float(proba[2]), 4)   # P(Critical)

        # SHAP explanation
        explanation = compute_shap_explanation(model, x_row, class_idx)

        coords = coord_map.get(dist_id, {"lat": 20.0, "lon": 78.0})

        results.append({
            "district_id":   dist_id,
            "district_name": row["district_name"],
            "state":         row["state"],
            "lat":           coords["lat"],
            "lon":           coords["lon"],
            "risk_label":    label,
            "confidence":    confidence,
            "risk_score":    risk_score,
            "features": {
                col: round(float(row[col]), 3) for col in FEATURE_COLS
            },
            "explanation":   explanation,
        })

    return results


def startup():
    """One-time initialisation: data -> features -> model -> results."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  AquaAlert  |  Groundwater Early-Warning System")
    print(sep)

    # 1. Data
    data_path = os.path.join(ROOT, "data", "groundwater_timeseries.csv")
    if os.path.exists(data_path):
        print("[startup] Loading cached time-series ...")
        ts_df = pd.read_csv(data_path, parse_dates=["date"])
        with open(os.path.join(ROOT, "data", "districts.json")) as f:
            districts_meta = json.load(f)
    else:
        print("[startup] Generating synthetic time-series ...")
        ts_df, districts_meta = generate_all_data(os.path.join(ROOT, "data"))

    # 2. Features
    print("[startup] Engineering features ...")
    feature_df = build_feature_matrix(ts_df, districts_meta)

    # 3. Model
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        print("[startup] Loading cached model ...")
        model, le = load_model()
    else:
        print("[startup] Training classifier ...")
        model, le = train(feature_df)

    # 4. Results + SHAP
    print("[startup] Computing predictions & SHAP explanations ...")
    results = _build_results(feature_df, model, le, ts_df)

    # Store in app config
    ts_groups = {did: grp.reset_index(drop=True)
                 for did, grp in ts_df.groupby("district_id")}

    app.config["AQUA_RESULTS"]    = results
    app.config["AQUA_TIMESERIES"] = ts_groups

    critical_count = sum(1 for r in results if r["risk_label"] == "Critical")
    print(f"\n[startup] Ready -- {len(results)} districts classified "
          f"({critical_count} Critical)")
    print("[startup] Dashboard -> http://localhost:5000\n")


# -- Static file serving ----------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


# -- __init__ files so Python treats directories as packages ----------------

def _ensure_init_files():
    for pkg in ["data", "features", "model", "api"]:
        init = os.path.join(ROOT, pkg, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()


# -- Entry point ------------------------------------------------------------

# -- Initialise on import (needed when gunicorn loads this module) ----------
_ensure_init_files()
startup()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False)
