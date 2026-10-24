"""
api/routes.py
-------------
Flask blueprint that exposes the AquaAlert REST API.

Endpoints:
  GET /api/health              → readiness probe (200 = ready, 503 = warming up)
  GET /api/districts           → list of all districts with risk classification
  GET /api/district/<id>       → detailed info + SHAP explanation for one district
  GET /api/alerts              → Critical-tier districts sorted by risk score
  GET /api/chart/<id>          → time-series data for a district (for mini chart)
  GET /api/feature-importance  → global SHAP feature importance across all districts
"""

from flask import Blueprint, jsonify, current_app

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/health")
def health():
    """Readiness probe — returns 200 when the model pipeline is loaded."""
    results = current_app.config.get("AQUA_RESULTS")
    if results is None:
        return jsonify({"status": "warming_up"}), 503
    return jsonify({"status": "ok", "districts": len(results)}), 200


@api_bp.route("/districts")
def get_districts():
    """Return all districts with classification results and coordinates."""
    return jsonify(current_app.config["AQUA_RESULTS"])


@api_bp.route("/district/<dist_id>")
def get_district(dist_id: str):
    """Return full detail (features + SHAP explanation) for one district."""
    results = current_app.config["AQUA_RESULTS"]
    detail  = next((r for r in results if r["district_id"] == dist_id), None)
    if detail is None:
        return jsonify({"error": "District not found"}), 404
    return jsonify(detail)


@api_bp.route("/alerts")
def get_alerts():
    """
    Return only Critical-tier districts, sorted by descending risk_score
    (probability of being Critical).
    """
    results  = current_app.config["AQUA_RESULTS"]
    critical = [r for r in results if r["risk_label"] == "Critical"]
    critical.sort(key=lambda r: r["risk_score"], reverse=True)
    return jsonify(critical)


@api_bp.route("/chart/<dist_id>")
def get_chart(dist_id: str):
    """
    Return monthly water depth + rainfall for the past 24 months
    for rendering a sparkline / mini chart on the detail panel.
    """
    ts_map = current_app.config["AQUA_TIMESERIES"]
    if dist_id not in ts_map:
        return jsonify({"error": "District not found"}), 404

    df = ts_map[dist_id].tail(24)
    return jsonify({
        "labels":      df["date"].dt.strftime("%b %Y").tolist(),
        "depth":       df["water_depth_mbgl"].round(2).tolist(),
        "rainfall":    df["rainfall_mm"].round(1).tolist(),
    })


@api_bp.route("/feature-importance")
def get_feature_importance():
    """
    Return mean absolute SHAP values across all districts
    (global feature importance for the About tab chart).
    """
    results = current_app.config.get("AQUA_RESULTS", [])
    if not results:
        return jsonify({"error": "Not ready"}), 503

    from features.engineering import FEATURE_COLS, FEATURE_LABELS
    import numpy as np

    # Sum absolute SHAP values per feature across all districts
    totals = {col: 0.0 for col in FEATURE_COLS}
    for r in results:
        for feat in r.get("explanation", {}).get("top_features", []):
            if feat["name"] in totals:
                totals[feat["name"]] += abs(feat["shap_value"])

    max_val = max(totals.values()) or 1.0
    items = [
        {
            "name":    col,
            "label":   FEATURE_LABELS[col],
            "value":   round(totals[col], 4),
            "pct":     round(totals[col] / max_val * 100, 1),
        }
        for col in FEATURE_COLS
    ]
    items.sort(key=lambda x: x["value"], reverse=True)
    return jsonify(items)


