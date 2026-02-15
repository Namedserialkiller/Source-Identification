"""
Live data fetching from OpenAQ API for Delhi air quality measurements.

Fetches real-time measurements and maps them to the format expected by
the preprocessing pipeline.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

try:
    from openaq import OpenAQ
    OPENAQ_AVAILABLE = True
except ImportError:
    OPENAQ_AVAILABLE = False
    print("Warning: openaq package not installed. Install with: pip install openaq")

# Define custom exceptions if openaq is not available
if OPENAQ_AVAILABLE:
    try:
        from openaq.exceptions import ApiKeyMissingError, UnauthorizedError
    except ImportError:
        # Define fallback exceptions if openaq.exceptions is not available
        class ApiKeyMissingError(Exception):
            pass
        class UnauthorizedError(Exception):
            pass
else:
    class ApiKeyMissingError(Exception):
        pass
    class UnauthorizedError(Exception):
        pass


# Delhi coordinates and location IDs
DELHI_COORDS = {
    "lat": 28.6139,
    "lon": 77.2090,
    "radius": 50000,  # 50km radius
}

# Parameter mapping: OpenAQ parameter names -> our column names
PARAMETER_MAP = {
    "pm25": "pm2_5",
    "pm10": "pm10",
    "co": "co",
    "no2": "no2",
    "o3": "o3",
    "so2": "so2",
    "nh3": "nh3",
    "no": "no",
}

# Historical means for missing parameters (from Delhi dataset)
HISTORICAL_MEANS = {
    "co": 3000.0,
    "no": 20.0,
    "no2": 90.0,
    "o3": 50.0,
    "so2": 80.0,
    "pm2_5": 200.0,
    "pm10": 300.0,
    "nh3": 25.0,
}


def get_openaq_client() -> Optional[object]:
    """
    Initialize and return OpenAQ client with API key from environment.

    Returns:
        OpenAQ client instance or None if API key missing.
    """
    if not OPENAQ_AVAILABLE:
        raise ImportError(
            "openaq package not installed. Install with: pip install openaq"
        )
    # Load .env from parent directory
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(env_path)
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ApiKeyMissingError(
            "OPENAQ_API_KEY not found in .env file. "
            "Please create .env file with your OpenAQ API key."
        )
    return OpenAQ(api_key=api_key)


def fetch_delhi_aqi(
    hours_back: int = 24,
    use_historical_fallback: bool = True,
) -> pd.DataFrame:
    """
    Fetch latest air quality measurements for Delhi from OpenAQ API.

    Queries measurements within Delhi area (coordinates 28.6139, 77.2090)
    for parameters: pm25, pm10, co, no2, o3, so2, nh3, no.

    Args:
        hours_back: Number of hours to look back for measurements (default 24).
        use_historical_fallback: If True, fill missing parameters with historical means.

    Returns:
        DataFrame with columns: date, pm2_5, pm10, co, no, no2, o3, so2, nh3.
        Compatible with preprocessing.py format.

    Raises:
        ApiKeyMissingError: If API key is not configured.
        UnauthorizedError: If API key is invalid (401).
        Exception: For other API errors.
    """
    try:
        client = get_openaq_client()
    except ApiKeyMissingError:
        raise

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(hours=hours_back)

    print(f"Fetching Delhi AQI data from OpenAQ API...")
    print(f"  Date range: {start_date.isoformat()} to {end_date.isoformat()}")
    print(f"  Location: Delhi ({DELHI_COORDS['lat']}, {DELHI_COORDS['lon']})")

    try:
        # Query measurements by coordinates
        response = client.measurements.get(
            coordinates=f"{DELHI_COORDS['lat']},{DELHI_COORDS['lon']}",
            radius=DELHI_COORDS["radius"],
            date_from=start_date.isoformat(),
            date_to=end_date.isoformat(),
            limit=10000,  # Max results
        )

        if not response or not hasattr(response, "results") or not response.results:
            print("  Warning: No measurements found. Using historical fallback.")
            return _create_fallback_dataframe(use_historical_fallback)

        # Parse response into DataFrame
        # Handle different response structures (list vs object with results attribute)
        results_list = response.results if hasattr(response, "results") else response
        if isinstance(results_list, dict) and "results" in results_list:
            results_list = results_list["results"]

        measurements = []
        for result in results_list:
            # Handle different result structures
            if isinstance(result, dict):
                param = result.get("parameter", "").lower()
                date_val = result.get("date", {})
                if isinstance(date_val, dict):
                    date_val = date_val.get("utc") or date_val.get("local")
                elif isinstance(date_val, str):
                    pass  # Already a string
                else:
                    date_val = datetime.utcnow()

                value = result.get("value")
                if value is None:
                    continue

                if param in PARAMETER_MAP:
                    measurements.append({
                        "date": pd.to_datetime(date_val) if date_val else datetime.utcnow(),
                        "parameter": PARAMETER_MAP[param],
                        "value": float(value),
                        "location": result.get("location", {}).get("name", "Unknown") if isinstance(result.get("location"), dict) else str(result.get("location", "Unknown")),
                    })

        if not measurements:
            print("  Warning: No valid measurements found. Using historical fallback.")
            return _create_fallback_dataframe(use_historical_fallback)

        # Convert to DataFrame and pivot
        df_raw = pd.DataFrame(measurements)
        df = df_raw.pivot_table(
            index="date",
            columns="parameter",
            values="value",
            aggfunc="mean",  # Average if multiple measurements at same time
        ).reset_index()

        # Ensure all expected columns exist
        expected_cols = ["date", "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"]
        for col in expected_cols:
            if col not in df.columns:
                if col == "date":
                    continue
                if use_historical_fallback:
                    df[col] = HISTORICAL_MEANS.get(col, np.nan)
                    print(f"  Warning: Missing parameter {col}, using historical mean: {HISTORICAL_MEANS.get(col, 'N/A')}")
                else:
                    df[col] = np.nan

        df = df[expected_cols]
        df = df.sort_values("date").reset_index(drop=True)

        print(f"  Successfully fetched {len(df)} measurement records")
        print(f"  Parameters found: {[c for c in df.columns if c != 'date' and df[c].notna().any()]}")

        return df

    except UnauthorizedError as e:
        raise UnauthorizedError(
            "OpenAQ API returned 401 Unauthorized. Please check your API key in .env file."
        ) from e
    except Exception as e:
        print(f"  Error fetching from OpenAQ API: {e}")
        if use_historical_fallback:
            print("  Falling back to historical means...")
            return _create_fallback_dataframe(use_historical_fallback)
        raise


def _create_fallback_dataframe(use_historical: bool = True) -> pd.DataFrame:
    """
    Create a DataFrame with historical means when API fails or returns no data.

    Args:
        use_historical: If True, use historical means; else use NaN.

    Returns:
        DataFrame with single row containing current timestamp and values.
    """
    now = datetime.utcnow()
    data = {"date": [now]}
    for param, col in PARAMETER_MAP.items():
        if use_historical:
            data[col] = [HISTORICAL_MEANS.get(col, np.nan)]
        else:
            data[col] = [np.nan]
    df = pd.DataFrame(data)
    return df


def fetch_latest_single_reading() -> pd.DataFrame:
    """
    Fetch the most recent single reading for real-time prediction.

    Returns:
        DataFrame with one row containing latest measurements.
    """
    df = fetch_delhi_aqi(hours_back=1, use_historical_fallback=True)
    if len(df) > 0:
        # Return most recent reading
        return df.tail(1).reset_index(drop=True)
    return df
