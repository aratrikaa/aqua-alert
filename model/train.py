"""
model/train.py
--------------
Trains a Random Forest (baseline) and XGBoost classifier on the
feature matrix, picks the better one by cross-validation accuracy,
and serialises both the model and label encoder to disk.

Classes: Safe | Moderate | Critical
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
import xgboost as xgb

from features.engineering import FEATURE_COLS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "aqua_model.joblib")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.joblib")

# Ordered so that 0=Safe, 1=Moderate, 2=Critical (used for confidence display)
LABEL_ORDER = ["Safe", "Moderate", "Critical"]


def train(feature_df: pd.DataFrame) -> tuple:
    """
    Train and persist the best classifier.

    Parameters
    ----------
    feature_df : DataFrame with FEATURE_COLS + 'risk_label' column.

    Returns
    -------
    (model, label_encoder)
    """
    X = feature_df[FEATURE_COLS].values
    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)
    y = le.transform(feature_df["risk_label"].values)

    # ── Random Forest baseline ───────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    )

    # ── XGBoost ─────────────────────────────────────────────────────────────
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=min(3, len(np.unique(y))), shuffle=True, random_state=42)

    # With only 12 districts the dataset is tiny; use leave-one-out-style CV
    rf_scores  = cross_val_score(rf,  X, y, cv=cv, scoring="accuracy")
    xgb_scores = cross_val_score(xgb_clf, X, y, cv=cv, scoring="accuracy")

    print(f"[model] RF  CV accuracy : {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
    print(f"[model] XGB CV accuracy : {xgb_scores.mean():.3f} ± {xgb_scores.std():.3f}")

    # Pick the better model
    if xgb_scores.mean() >= rf_scores.mean():
        best = xgb_clf
        print("[model] Selecting XGBoost as the primary model.")
    else:
        best = rf
        print("[model] Selecting Random Forest as the primary model.")

    best.fit(X, y)

    # Full-dataset report
    y_pred = best.predict(X)
    print("\n[model] Training set classification report:")
    print(classification_report(y, y_pred, target_names=le.classes_))

    # Persist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    joblib.dump(le,   ENCODER_PATH)
    print(f"[model] Saved model  → {MODEL_PATH}")
    print(f"[model] Saved encoder → {ENCODER_PATH}")

    return best, le


def load_model() -> tuple:
    """Load persisted model and label encoder."""
    model = joblib.load(MODEL_PATH)
    le    = joblib.load(ENCODER_PATH)
    return model, le
