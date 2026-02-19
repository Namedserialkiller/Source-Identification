"""
Real-time prediction using live OpenAQ data.

Fetches latest Delhi AQI measurements and predicts pollution source.
"""

import sys
import pandas as pd
import numpy as np

try:
    from fetch_live_data import fetch_latest_single_reading, fetch_delhi_aqi
    from predict import predict, predict_proba
    LIVE_PREDICT_AVAILABLE = True
except ImportError as e:
    LIVE_PREDICT_AVAILABLE = False
    print(f"Error importing required modules: {e}")
    print("Ensure openaq is installed: pip install openaq python-dotenv")


def predict_live_delhi(
    hours_back: int = 1,
    show_probabilities: bool = True,
) -> dict:
    """
    Fetch latest Delhi AQI data and predict pollution source.

    Args:
        hours_back: Hours to look back for measurements (default 1 for most recent).
        show_probabilities: If True, include class probabilities in output.

    Returns:
        Dictionary with predictions, probabilities (if requested), and metadata.
    """
    if not LIVE_PREDICT_AVAILABLE:
        raise ImportError(
            "Live prediction requires openaq package. "
            "Install with: pip install openaq python-dotenv"
        )

    print("Fetching latest Delhi AQI data from OpenAQ...")
    try:
        if hours_back <= 1:
            df = fetch_latest_single_reading()
        else:
            df = fetch_delhi_aqi(hours_back=hours_back, use_historical_fallback=True)
            df = df.tail(1).reset_index(drop=True)

        if df.empty:
            raise ValueError("No data fetched from OpenAQ API")

        print(f"\nLatest reading timestamp: {df['date'].iloc[0]}")
        print(f"Parameters: {[c for c in df.columns if c != 'date']}")

        # Predict
        labels = predict(df, fill_missing=True)
        result = {
            "timestamp": str(df["date"].iloc[0]),
            "predicted_source": labels[0] if len(labels) > 0 else "unknown",
            "measurements": df.iloc[0].to_dict(),
        }

        if show_probabilities:
            probs = predict_proba(df, fill_missing=True)
            if len(probs) > 0:
                # Get class names from encoder
                from predict import load_artifacts
                _, encoder, _ = load_artifacts()
                class_names = encoder.classes_
                result["probabilities"] = {
                    cls: float(prob) for cls, prob in zip(class_names, probs[0])
                }

        return result

    except Exception as e:
        print(f"Error in live prediction: {e}")
        raise


def main():
    """CLI entry point for live predictions."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict pollution source from live Delhi AQI data."
    )
    parser.add_argument(
        "--hours-back",
        type=int,
        default=1,
        help="Hours to look back for measurements (default: 1)",
    )
    parser.add_argument(
        "--no-probabilities",
        action="store_true",
        help="Don't show class probabilities",
    )
    args = parser.parse_args()

    try:
        result = predict_live_delhi(
            hours_back=args.hours_back,
            show_probabilities=not args.no_probabilities,
        )

        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        print(f"Timestamp: {result['timestamp']}")
        print(f"Predicted Source: {result['predicted_source'].upper()}")
        if "probabilities" in result:
            print("\nClass Probabilities:")
            for cls, prob in sorted(
                result["probabilities"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {cls:12s}: {prob:.2%}")
        print("\nMeasurements:")
        for key, val in result["measurements"].items():
            if key != "date":
                print(f"  {key:12s}: {val}")
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()