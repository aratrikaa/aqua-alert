"""
api/routes.py
-------------
Flask blueprint that exposes the AquaAlert REST API.

Endpoints:
  GET /api/districts          → list of all districts with risk classification
  GET /api/district/<id>      → detailed info + SHAP explanation for one district
  GET /api/alerts             → Critical-tier districts sorted by risk score
  GET /api/chart/<id>         → time-series data for a district (for mini chart)
"""

from flask import Blueprint, jsonify, current_app

api_bp = Blueprint("api", __name__, url_prefix="/api")


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
