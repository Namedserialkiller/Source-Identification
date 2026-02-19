"""
Prediction pipeline: preprocess raw data, load artifacts, return predictions.
"""

import os
import joblib
import pandas as pd
import numpy as np

from config import MODEL_PATH, ENCODER_PATH, SCALER_PATH, ARTIFACTS_DIR
from preprocessing import preprocess, get_feature_columns


def load_artifacts():
    """
    Load saved model, label encoder, and scaler from artifacts/.

    Returns:
        Tuple of (model, encoder, scaler). Scaler may be None if not saved.
    """
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(ENCODER_PATH):
        raise FileNotFoundError(
            f"Artifacts not found in {ARTIFACTS_DIR}. Train the model first."
        )
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.isfile(SCALER_PATH) else None
    return model, encoder, scaler


def predict(raw_df: pd.DataFrame, fill_missing: bool = True) -> np.ndarray:
    """
    Preprocess raw data, load model and encoder, return predicted source labels.

    Raw data must contain: date, PM (pm2_5/PM25, pm10/PM10), gaseous pollutants
    (co, no, no2, o3, so2, nh3). Optional: wind_dir, wind_speed, fire_count.
    Applies the same preprocessing as training (cyclical month, wind u/v,
    pm_ratio, nitrogen_ratio, gaseous normalization).

    Args:
        raw_df: DataFrame with required and optional columns.
        fill_missing: If True, fill NaN in features with 0 before prediction.
                      If False, rows with NaN are dropped from the output alignment.

    Returns:
        Array of predicted labels (string: stubble, traffic, industry, mixed).
    """
    model, encoder, scaler = load_artifacts()
    df, _ = preprocess(raw_df, scaler=scaler)
    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols]
    if fill_missing:
        X = X.fillna(0)
    else:
        valid = X.notna().all(axis=1)
        X = X.loc[valid]
    y_encoded = model.predict(X)
    labels = encoder.inverse_transform(y_encoded)
    if not fill_missing:
        out = np.full(len(raw_df), np.nan, dtype=object)
        out[valid.values] = labels
        return out
    return labels


def predict_proba(raw_df: pd.DataFrame, fill_missing: bool = True) -> np.ndarray:
    """
    Return predicted class probabilities for each row of raw data.

    Args:
        raw_df: Same as in predict().
        fill_missing: Same as in predict().

    Returns:
        Array of shape (n_samples, n_classes) with probabilities.
    """
    model, encoder, scaler = load_artifacts()
    df, _ = preprocess(raw_df, scaler=scaler)
    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols].fillna(0) if fill_missing else X[feature_cols].dropna()
    return model.predict_proba(X)
