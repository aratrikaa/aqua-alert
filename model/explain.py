"""
model/explain.py
----------------
Computes SHAP values for a single district prediction and converts
the top driving factors into human-readable plain-language sentences.
"""

import numpy as np
import shap
from features.engineering import FEATURE_COLS, FEATURE_LABELS


def _get_explainer(model):
    """
    Return a SHAP TreeExplainer for tree-based models (RF / XGBoost).
    """
    return shap.TreeExplainer(model)


def compute_shap_explanation(
    model,
    feature_row: np.ndarray,
    predicted_class_idx: int,
    top_n: int = 3,
) -> dict:
    """
    Compute SHAP values for a single sample and return the top-N
    contributing features with their direction and plain-English reason.

    Parameters
    ----------
    model            : trained sklearn / XGBoost model
    feature_row      : 1-D numpy array of shape (n_features,)
    predicted_class_idx : index of the predicted class (0=Safe,1=Moderate,2=Critical)
    top_n            : number of top features to surface

    Returns
    -------
    dict with keys:
      - top_features : list of {name, label, shap_value, direction, readable}
      - summary      : one-sentence plain-English summary
    """
    explainer = _get_explainer(model)
    X = feature_row.reshape(1, -1)

    shap_values = explainer.shap_values(X)

    # shap_values shape differs between RF (list of arrays) and XGB (2-D array)
    if isinstance(shap_values, list):
        # Random Forest: shap_values[class_idx] has shape (1, n_features)
        sv = shap_values[predicted_class_idx][0]
    else:
        # XGBoost multi-class: shape (1, n_features, n_classes) OR (n_classes, 1, n_features)
        if shap_values.ndim == 3:
            # (1, n_features, n_classes)
            sv = shap_values[0, :, predicted_class_idx]
        else:
            sv = shap_values[0]

    # Rank by absolute contribution
    abs_sv = np.abs(sv)
    top_indices = np.argsort(abs_sv)[::-1][:top_n]

    top_features = []
    for idx in top_indices:
        fname  = FEATURE_COLS[idx]
        flabel = FEATURE_LABELS[fname]
        sv_val = float(sv[idx])
        direction = "high" if sv_val > 0 else "low"

        top_features.append({
            "name":      fname,
            "label":     flabel,
            "shap_value": round(sv_val, 4),
            "direction": direction,
            "raw_value": round(float(feature_row[idx]), 3),
        })

    # Build a plain-language summary
    summary = _build_summary(predicted_class_idx, top_features, feature_row)

    return {"top_features": top_features, "summary": summary}


# ── Plain-language templates ───────────────────────────────────────────────────

_CLASS_WORDS = {0: "Safe", 1: "Moderate", 2: "Critical"}

_FEATURE_PHRASES = {
    # (direction, feature_name) → readable phrase
    ("high", "current_depth"):       "very deep water table",
    ("low",  "current_depth"):       "shallow water table",
    ("high", "annual_trend"):        "rapid annual depletion trend",
    ("low",  "annual_trend"):        "improving annual trend",
    ("high", "extraction_rate"):     "high groundwater extraction",
    ("low",  "extraction_rate"):     "low extraction pressure",
    ("low",  "monsoon_recovery"):    "weak monsoon recharge",
    ("high", "monsoon_recovery"):    "strong monsoon recharge",
    ("low",  "rainfall_lag_90d"):    "low 3-month cumulative rainfall",
    ("high", "rainfall_lag_90d"):    "above-average recent rainfall",
    ("low",  "rainfall_lag_60d"):    "below-average 2-month rainfall",
    ("high", "rainfall_lag_60d"):    "above-average 2-month rainfall",
    ("low",  "rainfall_lag_30d"):    "low recent rainfall",
    ("high", "rainfall_lag_30d"):    "high recent rainfall",
    ("high", "premonsoon_depth"):    "severe pre-monsoon water table drop",
    ("low",  "premonsoon_depth"):    "mild pre-monsoon depletion",
    ("high", "postmonsoon_depth"):   "inadequate post-monsoon recovery",
    ("low",  "postmonsoon_depth"):   "good post-monsoon water levels",
}


def _build_summary(class_idx: int, top_features: list, feature_row: np.ndarray) -> str:
    class_word = _CLASS_WORDS.get(class_idx, "Unknown")

    phrases = []
    for feat in top_features:
        key = (feat["direction"], feat["name"])
        phrase = _FEATURE_PHRASES.get(key, feat["label"])
        phrases.append(phrase)

    if not phrases:
        return f"Classified as {class_word} based on overall hydrological indicators."

    if len(phrases) == 1:
        reason = phrases[0]
    elif len(phrases) == 2:
        reason = f"{phrases[0]} and {phrases[1]}"
    else:
        reason = f"{phrases[0]}, {phrases[1]}, and {phrases[2]}"

    return f"Flagged {class_word} primarily due to {reason}."
