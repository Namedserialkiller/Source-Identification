"""
Evaluation utilities: classification report and feature importance plot.
"""

import os
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from config import MODEL_PATH, ENCODER_PATH, ARTIFACTS_DIR, PLOTS_DIR
from preprocessing import get_feature_columns

# Feature categories for visualization (chemical tracers vs PM)
CHEMICAL_TRACERS = {"co", "no", "no2", "o3", "so2", "nh3", "nitrogen_ratio"}
PM_FEATURES = {"PM25", "PM10", "pm_ratio"}


def load_model_and_encoder():
    """
    Load the saved model and label encoder from artifacts/.

    Returns:
        Tuple of (model, encoder).
    """
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(ENCODER_PATH):
        raise FileNotFoundError(
            f"Artifacts not found in {ARTIFACTS_DIR}. Run train.py first."
        )
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


def classification_report_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    encoder: LabelEncoder = None,
) -> str:
    """
    Generate sklearn classification report; optionally decode labels.

    Args:
        y_true: True labels (encoded or string).
        y_pred: Predicted labels (encoded or string).
        encoder: If provided, inverse_transform is used for display.

    Returns:
        String classification report.
    """
    if encoder is not None:
        try:
            y_true = encoder.inverse_transform(y_true.astype(int))
            y_pred = encoder.inverse_transform(y_pred.astype(int))
        except (ValueError, TypeError):
            pass
    return classification_report(y_true, y_pred)


def run_classification_report(
    X: pd.DataFrame,
    y: pd.Series,
    model=None,
    encoder=None,
) -> str:
    """
    Compute and return classification report for given features and labels.

    Args:
        X: Feature matrix (same columns as get_feature_columns()).
        y: True labels (string or encoded).
        model: Fitted classifier; if None, loaded from artifacts.
        encoder: Fitted LabelEncoder; if None, loaded from artifacts.

    Returns:
        Classification report string.
    """
    if model is None or encoder is None:
        model, encoder = load_model_and_encoder()
    y_encoded = encoder.transform(y.astype(str))
    y_pred = model.predict(X)
    return classification_report_from_arrays(y_encoded, y_pred, encoder=encoder)


def _get_feature_category(name: str) -> str:
    """Return category for coloring: chemical, PM, or other."""
    if name in CHEMICAL_TRACERS:
        return "chemical"
    if name in PM_FEATURES:
        return "pm"
    return "other"


def plot_feature_importance(
    model,
    feature_names: list = None,
    save_path: str = None,
    top_n: int = None,
) -> None:
    """
    Plot and optionally save feature importance from tree-based model.

    Color-codes features by category: chemical tracers (SO2, NO2, NH3, etc.)
    vs PM features (PM25, PM10, pm_ratio) to show how much chemical tracers
    contribute to source identification compared to PM levels.

    Args:
        model: Fitted XGBClassifier (or model with feature_importances_).
        feature_names: List of feature names; default from get_feature_columns().
        save_path: If set, save figure to this path (e.g. artifacts/plots/feature_importance.png).
        top_n: If set, show only top_n features by importance.
    """
    import matplotlib.patches as mpatches

    if feature_names is None:
        feature_names = get_feature_columns()
    imp = model.feature_importances_
    if len(feature_names) != len(imp):
        feature_names = [f"f{i}" for i in range(len(imp))]
    idx = np.argsort(imp)[::-1]
    if top_n is not None:
        idx = idx[:top_n]
    names = [feature_names[i] for i in idx]
    values = imp[idx]
    colors = []
    for n in names:
        cat = _get_feature_category(n)
        if cat == "chemical":
            colors.append("#2ecc71")
        elif cat == "pm":
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")
    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.4)))
    ax.barh(range(len(names)), values, align="center", color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance", fontsize=11)
    ax.set_title(
        "Feature importance: chemical tracers vs PM (chemical=green, PM=red, other=blue)",
        fontsize=11,
    )
    legend = [
        mpatches.Patch(color="#2ecc71", label="Chemical tracers (SO2, NO2, NH3, etc.)"),
        mpatches.Patch(color="#e74c3c", label="PM features (PM2.5, PM10, ratio)"),
        mpatches.Patch(color="#3498db", label="Other (time, wind)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot: {save_path}")
    plt.close()


def evaluate_on_data(
    X: pd.DataFrame,
    y: pd.Series,
    save_importance_path: str = None,
    save_with_timestamp: bool = False,
) -> str:
    """
    Run full evaluation: classification report and feature importance plot.

    Args:
        X: Feature matrix.
        y: True labels (string).
        save_importance_path: Path to save feature importance plot;
            default artifacts/plots/feature_importance.png.
        save_with_timestamp: If True, append timestamp to filename to avoid overwrite.

    Returns:
        Classification report string.
    """
    model, encoder = load_model_and_encoder()
    report = run_classification_report(X, y, model=model, encoder=encoder)
    if save_importance_path is None:
        base = "feature_importance"
        if save_with_timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{base}_{ts}"
        save_importance_path = os.path.join(PLOTS_DIR, f"{base}.png")
    plot_feature_importance(model, save_path=save_importance_path)
    return report
