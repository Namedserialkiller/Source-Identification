"""
Centralized paths and configuration for the pollution source identification pipeline.
"""

import os

# Directory for saved models and encoders
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "encoder.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")


def ensure_artifacts_dir() -> str:
    """
    Create artifacts directory if it does not exist.

    Returns:
        Path to the artifacts directory.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return ARTIFACTS_DIR
