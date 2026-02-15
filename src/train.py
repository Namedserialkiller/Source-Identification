"""
Training script for the pollution source classifier.

Trains an XGBClassifier with sample weights for class imbalance,
encodes labels with LabelEncoder, and exports model, encoder, and scaler
to artifacts/.
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import MODEL_PATH, ENCODER_PATH, SCALER_PATH, ensure_artifacts_dir
from preprocessing import get_feature_columns


def train(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    **xgb_kwargs,
) -> tuple:
    """
    Train XGBClassifier with encoded labels and sample weights for imbalance.

    Args:
        X: Feature matrix (columns must match get_feature_columns()).
        y: String labels (stubble, traffic, industry, mixed).
        random_state: Random seed for reproducibility.
        **xgb_kwargs: Optional arguments passed to XGBClassifier.

    Returns:
        Tuple of (fitted XGBClassifier, fitted LabelEncoder).
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.astype(str))
    class_weights = compute_sample_weight(class_weight="balanced", y=y_encoded)
    default_kwargs = {
        "random_state": random_state,
        "eval_metric": "mlogloss",
        "verbosity": 1,
    }
    default_kwargs.update(xgb_kwargs)
    model = XGBClassifier(**default_kwargs)
    model.fit(X, y_encoded, sample_weight=class_weights, verbose=True)
    return model, encoder


def save_artifacts(model: XGBClassifier, encoder: LabelEncoder, scaler=None) -> None:
    """
    Save model, label encoder, and scaler to the centralized artifacts/ folder.

    Args:
        model: Fitted XGBClassifier.
        encoder: Fitted LabelEncoder.
        scaler: Fitted StandardScaler for gaseous pollutants (optional).
    """
    ensure_artifacts_dir()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    if scaler is not None:
        joblib.dump(scaler, SCALER_PATH)


def _check_feature_correlation(X: pd.DataFrame, y: pd.Series, encoder) -> None:
    """Print warning if pm_ratio or fire_count have near-perfect correlation with target."""
    for col in ["pm_ratio", "fire_count"]:
        if col not in X.columns:
            continue
        x_vals = X[col].fillna(0).values
        if np.std(x_vals) < 1e-10:
            continue
        y_enc = encoder.transform(y.astype(str))
        corr = np.corrcoef(x_vals, y_enc)[0, 1]
        if not np.isnan(corr) and abs(corr) > 0.95:
            print(f"  [WARNING] {col} has very high correlation ({corr:.3f}) with target.")

def run_training(
    df: pd.DataFrame,
    target_col: str = "source",
    test_size: float = 0.2,
    random_state: int = 42,
    scaler=None,
    exclude_features: list = None,
    save_artifacts_flag: bool = True,
) -> tuple:
    """
    Run full training pipeline: feature matrix, encode labels, train, save.

    Splits data into train/test for evaluation. Returns test set for
    downstream evaluation and visualization. Includes chemical tracers
    (co, no, no2, o3, so2, nh3) and nitrogen_ratio in features.

    Args:
        df: Preprocessed DataFrame with features and label column.
        target_col: Name of the label column.
        test_size: Fraction of data for test set (default 0.2).
        random_state: Random seed for train/test split reproducibility.
        scaler: Fitted StandardScaler (from preprocessing); saved for prediction.

    Returns:
        Tuple of (model, encoder, X_test, y_test).
    """
    feature_cols = get_feature_columns(exclude=exclude_features)
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[available].copy()
    y = df[target_col]
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].fillna(0).reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    model, encoder = train(X_train, y_train, random_state=random_state)
    _check_feature_correlation(X_train, y_train, encoder)
    if save_artifacts_flag:
        save_artifacts(model, encoder, scaler=scaler)
    return model, encoder, X_test, y_test
