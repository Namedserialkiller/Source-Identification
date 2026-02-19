try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import os
import numpy as np
import joblib
from datetime import datetime

from config import MODEL_PATH, ENCODER_PATH, SCALER_PATH, PLOTS_DIR


def load_model_and_encoder():
    """
    Load the trained model and label encoder from artifacts.

    Returns:
        Tuple of (model, encoder).
    """
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


def run_classification_report(X_test, y_test, model=None, encoder=None):
    """
    Generate and return classification report.

    Args:
        X_test: Test features.
        y_test: True labels.
        model: Trained model (if None, load from artifacts).
        encoder: Label encoder (if None, load from artifacts).

    Returns:
        Classification report string.
    """
    if model is None or encoder is None:
        model, encoder = load_model_and_encoder()
    y_pred_encoded = model.predict(X_test)
    y_pred = encoder.inverse_transform(y_pred_encoded)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)
    return report


def evaluate_on_data(X_test, y_test, save_with_timestamp=False):
    """
    Evaluate model on test data and save plots.

    Args:
        X_test: Test features.
        y_test: True labels.
        save_with_timestamp: If True, save plots with timestamp.
    """
    model, encoder = load_model_and_encoder()
    y_pred_encoded = model.predict(X_test)
    y_pred = encoder.inverse_transform(y_pred_encoded)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)
    print("\n--- Classification Report ---")
    print(report)

    # Macro F1
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(14, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap='Blues', xticks_rotation=45, ax=ax)
    plt.title('Confusion Matrix: Pollution Source Categories')
    plt.tight_layout()

    if save_with_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_path = os.path.join(PLOTS_DIR, f'confusion_matrix_{timestamp}.png')
    else:
        cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # SHAP explainability
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
            if save_with_timestamp:
                shap_path = os.path.join(PLOTS_DIR, f'shap_summary_{timestamp}.png')
            else:
                shap_path = os.path.join(PLOTS_DIR, 'shap_summary.png')
            plt.savefig(shap_path, bbox_inches='tight')
            plt.close()
            print(f"SHAP summary plot saved to {shap_path}")
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
    else:
        print("Skipping SHAP analysis (shap not installed)")


def run_evaluation(model, X_test, y_test, encoder, plots_dir):
    """
    Comprehensive evaluation suite including SHAP and metrics.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True labels.
        encoder: Label encoder.
        plots_dir: Directory to save plots.

    Returns:
        Tuple of (classification_report, macro_f1).
    """
    os.makedirs(plots_dir, exist_ok=True)

    # Predictions
    y_pred_encoded = model.predict(X_test)
    y_pred = encoder.inverse_transform(y_pred_encoded)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)
    print("\n--- Classification Report ---")
    print(report)

    # Macro F1
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(14, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap='Blues', xticks_rotation=45, ax=ax)
    plt.title('Confusion Matrix: Pollution Source Categories')
    plt.tight_layout()
    cm_plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_plot_path}")

    # SHAP
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
            shap_plot_path = os.path.join(plots_dir, 'shap_summary.png')
            plt.savefig(shap_plot_path, bbox_inches='tight')
            plt.close()
            print(f"SHAP summary plot saved to {shap_plot_path}")
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
    else:
        print("Skipping SHAP analysis (shap not installed)")

    return report, macro_f1