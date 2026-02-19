"""
Entry point to run the full pipeline: load data, preprocess, label, train, evaluate.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from config import ensure_artifacts_dir
from preprocessing import preprocess, get_feature_columns
from label_generation import add_labels
from train import run_training
from evaluate import evaluate_on_data, run_classification_report, load_model_and_encoder

try:
    from fetch_live_data import fetch_delhi_aqi, fetch_latest_single_reading
    LIVE_DATA_AVAILABLE = True
except ImportError:
    LIVE_DATA_AVAILABLE = False
    print("Warning: Live data fetching not available. Install openaq: pip install openaq")


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV. Expected columns: date, PM (pm2_5/PM25, pm10/PM10),
    gaseous pollutants (co, no, no2, o3, so2, nh3). Optional: wind_dir,
    wind_speed, fire_count.

    Args:
        path: Path to CSV file.

    Returns:
        DataFrame with raw observations.
    """
    df = pd.read_csv(path)
    required = ["date"]
    pm_cols = ["pm2_5", "PM25", "pm25"]
    pm10_cols = ["pm10", "PM10"]
    has_pm = any(c in df.columns for c in pm_cols) and any(c in df.columns for c in pm10_cols)
    if not has_pm:
        raise ValueError("CSV must have PM columns (pm2_5/PM25 and pm10/PM10)")
    return df


def run_pipeline(
    data_path: str = None,
    evaluate: bool = True,
    use_live_data: bool = False,
) -> pd.DataFrame:
    """
    Run full pipeline: load -> preprocess -> generate labels -> train -> evaluate.

    Args:
        data_path: Path to CSV with date, PM25, PM10, wind_dir, wind_speed,
                   fire_count. Ignored if use_live_data=True.
        evaluate: If True, run evaluation (classification report + importance plot).
        use_live_data: If True, fetch data from OpenAQ API instead of CSV.

    Returns:
        Preprocessed DataFrame with features and labels (for inspection).
    """
    ensure_artifacts_dir()
    if use_live_data:
        if not LIVE_DATA_AVAILABLE:
            raise ImportError(
                "Live data fetching requires openaq package. "
                "Install with: pip install openaq python-dotenv"
            )
        print("\n=== Fetching Live Data from OpenAQ API ===")
        try:
            df = fetch_delhi_aqi(hours_back=168, use_historical_fallback=False)
            print(f"Fetched {len(df)} records from OpenAQ\n")
        except Exception as e:
            print(f"Error fetching live data: {e}")
            print("Falling back to CSV data...")
            if data_path is None:
                raise ValueError(
                    "Live data fetch failed and no CSV path provided. "
                    "Please provide data_path or fix API configuration."
                )
            df = load_data(data_path)
    else:
        if data_path is None:
            raise ValueError("data_path is required when use_live_data=False")
        df = load_data(data_path)
    df, scaler = preprocess(df)
    df = add_labels(df)
    model, encoder, X_test, y_test = run_training(
        df, target_col="source", scaler=scaler
    )
    if evaluate:
        y_pred = model.predict(X_test)
        report = run_classification_report(
            X_test, y_test, model=model, encoder=encoder
        )
        print("\n--- Classification Report (Test Set) ---")
        print(report)
        accuracy = accuracy_score(y_test, encoder.inverse_transform(y_pred))
        print(f"\nFinal Model Accuracy: {accuracy:.2%}")

        if accuracy >= 0.999:
            print("\n--- Feature Ablation: Training without fire_count ---")
            model_ablation, enc_ablation, X_test_ab, y_test_ab = run_training(
                df,
                target_col="source",
                scaler=scaler,
                exclude_features=["fire_count"],
                save_artifacts_flag=False,
            )
            y_pred_ab = model_ablation.predict(X_test_ab)
            report_ab = run_classification_report(
                X_test_ab, y_test_ab, model=model_ablation, encoder=enc_ablation
            )
            print(report_ab)
            acc_ab = accuracy_score(
                y_test_ab, enc_ablation.inverse_transform(y_pred_ab)
            )
            print(f"Ablation (no fire_count) Accuracy: {acc_ab:.2%}")

        evaluate_on_data(X_test, y_test, save_with_timestamp=True)
    return df


def main():
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Air pollution source identification: train and evaluate."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default=None,
        help="Path to CSV with date, PM25, PM10, wind_dir, wind_speed, fire_count.",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip evaluation after training.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data and run pipeline (for testing).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch live data from OpenAQ API instead of CSV file.",
    )
    args = parser.parse_args()

    if args.synthetic:
        np.random.seed(42)
        n = 500
        months = np.random.choice([1, 2, 5, 6, 10, 11], size=n)
        dates = pd.to_datetime(
            [f"2020-{m:02d}-{(d % 28) + 1}" for m, d in zip(months, np.random.randint(1, 28, n))]
        )
        df = pd.DataFrame({
            "date": dates,
            "PM25": np.clip(np.random.lognormal(2, 1, n), 10, 500),
            "PM10": np.clip(np.random.lognormal(2.5, 1, n), 20, 600),
            "wind_dir": np.random.uniform(0, 360, n),
            "wind_speed": np.random.uniform(0, 15, n),
            "fire_count": np.random.poisson(3, n),
            "co": np.clip(np.random.lognormal(7, 1, n), 500, 5000),
            "no": np.clip(np.random.lognormal(1, 1.5, n), 0.1, 100),
            "no2": np.clip(np.random.lognormal(3, 0.8, n), 10, 150),
            "o3": np.clip(np.random.lognormal(2, 1, n), 1, 100),
            "so2": np.clip(np.random.lognormal(2.5, 1, n), 5, 200),
            "nh3": np.clip(np.random.lognormal(1.5, 1.2, n), 1, 60),
        })
        # Adjust for weak supervision: stubble (Oct/Nov + fires + high NH3)
        df.loc[months >= 10, "fire_count"] += 8
        df.loc[months >= 10, "nh3"] *= 1.5
        synthetic_path = os.path.join(ensure_artifacts_dir(), "synthetic_data.csv")
        df.to_csv(synthetic_path, index=False)
        data_path = synthetic_path
    elif args.live:
        data_path = None
    elif args.data_path is None:
        print("Provide data_path, use --synthetic, or use --live for OpenAQ API.", file=sys.stderr)
        sys.exit(1)
    else:
        data_path = args.data_path

    run_pipeline(
        data_path=data_path,
        evaluate=not args.no_evaluate,
        use_live_data=args.live,
    )


if __name__ == "__main__":
    main()
