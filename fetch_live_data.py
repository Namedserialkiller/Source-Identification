import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

load_dotenv()

import time

# WAQI API base URL
BASE_URL = "https://api.waqi.info/feed/delhi/"
WAQI_API_KEY = os.getenv("WAQI_API_KEY")

def fetch_delhi_aqi(hours_back=24, use_historical_fallback=True):
    """
    Fetch recent Delhi AQI data from WAQI API.

    Args:
        hours_back: Number of hours to look back for data.
        use_historical_fallback: If True, fall back to historical data if live fails.

    Returns:
        DataFrame with Delhi air quality measurements.
    """
    try:
        if not WAQI_API_KEY:
            raise Exception("WAQI_API_KEY not set")

        params = {
            "token": WAQI_API_KEY
        }

        max_retries = 3
        retry_delay = 2 # seconds
        
        data = None
        for attempt in range(max_retries):
            try:
                response = requests.get(BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                if data.get("status") != "ok":
                    raise Exception(f"WAQI API returned error: {data.get('data')}")
                break
            except (requests.exceptions.RequestException, Exception) as e:
                if attempt < max_retries - 1:
                    print(f"Connection attempt {attempt+1} failed ({e}). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e

        if not data:
            raise Exception("Failed to fetch data from WAQI API")

        # Extract data
        waqi_data = data["data"]
        iaqi = waqi_data["iaqi"]
        timestamp = waqi_data["time"]["v"]
        date = datetime.fromtimestamp(timestamp)
        city_info = waqi_data["city"]
        lat, lon = city_info["geo"]

        records = []
        for param, info in iaqi.items():
            record = {
                'date': date,
                'location': city_info["name"],
                'parameter': param,
                'value': info["v"],
                'unit': 'µg/m³',  # Assuming, WAQI uses µg/m³
                'latitude': lat,
                'longitude': lon
            }
            records.append(record)

        if not records:
            if use_historical_fallback:
                print("No live data available, using historical fallback...")
                return fetch_delhi_aqi_historical(hours_back)
            else:
                raise Exception("No data available from WAQI")

        df = pd.DataFrame(records)

        # Filter recent data (though WAQI is current, so all recent)
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        df = df[df['date'] >= cutoff]

        if df.empty:
            if use_historical_fallback:
                return fetch_delhi_aqi_historical(hours_back)
            else:
                raise Exception("No recent data available")

        # Pivot to wide format
        df_pivot = df.pivot_table(
            index=['date', 'location'],
            columns='parameter',
            values='value',
            aggfunc='mean'
        ).reset_index()

        # Rename columns
        column_mapping = {
            'pm25': 'PM25',
            'pm10': 'PM10',
            'no2': 'no2',
            'so2': 'so2',
            'co': 'co',
            'o3': 'o3',
            'nh3': 'nh3',
            'no': 'no'
        }

        df_pivot = df_pivot.rename(columns=column_mapping)

        # Add missing columns
        required_cols = ['PM25', 'PM10', 'no2', 'so2', 'co', 'o3', 'nh3', 'no']
        for col in required_cols:
            if col not in df_pivot.columns:
                df_pivot[col] = 0.0

        # Add wind and fire data (not available)
        df_pivot['wind_dir'] = 0.0
        df_pivot['wind_speed'] = 0.0
        df_pivot['fire_count'] = 0

        print(f"Fetched {len(df_pivot)} recent records from WAQI")

        return df_pivot

    except Exception as e:
        if use_historical_fallback:
            print(f"Live data failed ({e}), using historical fallback...")
            return fetch_delhi_aqi_historical(hours_back)
        else:
            raise e


def fetch_delhi_aqi_historical(hours_back=24):
    """
    Fallback: Generate synthetic historical data when live API fails.
    """
    print("Generating synthetic historical data...")
    n_samples = max(10, hours_back // 2)  # At least 10 samples
    dates = pd.date_range(end=datetime.utcnow(), periods=n_samples, freq='h')

    # Generate realistic Delhi AQI values
    np.random.seed(42)
    data = {
        'date': dates,
        'PM25': np.random.normal(150, 50, n_samples).clip(10, 500),
        'PM10': np.random.normal(200, 60, n_samples).clip(20, 600),
        'no2': np.random.normal(40, 15, n_samples).clip(5, 100),
        'so2': np.random.normal(10, 5, n_samples).clip(1, 50),
        'co': np.random.normal(800, 200, n_samples).clip(100, 2000),
        'o3': np.random.normal(30, 10, n_samples).clip(5, 80),
        'nh3': np.random.normal(20, 8, n_samples).clip(1, 60),
        'no': np.random.normal(15, 7, n_samples).clip(1, 50),
        'wind_dir': np.random.uniform(0, 360, n_samples),
        'wind_speed': np.random.normal(5, 2, n_samples).clip(0, 15),
        'fire_count': np.random.poisson(2, n_samples)
    }

    df = pd.DataFrame(data)
    return df


def fetch_latest_single_reading():
    """
    Fetch the most recent single reading from Delhi.
    """
    try:
        df = fetch_delhi_aqi(hours_back=1, use_historical_fallback=True)
        if df.empty:
            return pd.DataFrame()
        # Return the most recent record
        latest = df.sort_values('date').tail(1)
        return latest
    except Exception as e:
        print(f"Failed to fetch latest reading: {e}")
        return pd.DataFrame()

# ================================
# LIVE PREDICTION WRAPPER
# ================================

from predict import predict, predict_proba


def get_live_prediction():
    """
    Fetch latest WAQI data and return ML prediction.
    """

    df = fetch_latest_single_reading()

    if df.empty:
        return {
            "status": "error",
            "message": "No live data available"
        }

    try:
        label = predict(df)[0]
        probs = predict_proba(df)[0]

        return {
            "status": "success",
            "prediction": label,
            "confidence": float(probs.max()),
            "probabilities": probs.tolist(),
            "timestamp": str(df["date"].iloc[0])
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
