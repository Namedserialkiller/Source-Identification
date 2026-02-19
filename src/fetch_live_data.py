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

PARAMETER_MAP = {"pm25": "pm2_5", "pm10": "pm10", "co": "co", "no2": "no2", "o3": "o3", "so2": "so2", "nh3": "nh3", "no": "no"}
HISTORICAL_MEANS = {"co": 3000.0, "no": 20.0, "no2": 90.0, "o3": 50.0, "so2": 80.0, "pm2_5": 200.0, "pm10": 300.0, "nh3": 25.0}

DELHI_SENSORS = [13866, 24, 13864, 27, 28, 29, 26, 30, 33, 32, 31, 34, 392, 12234783, 12234789]

def fetch_delhi_aqi(hours_back: int = 24, use_historical_fallback: bool = True) -> pd.DataFrame:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key: return _create_fallback_dataframe(use_historical_fallback)
    
    try:
        client = OpenAQ(api_key=api_key)
        start_date = datetime.utcnow() - timedelta(hours=hours_back)
        measurements = []
        for s_id in DELHI_SENSORS:
            if s_id > 2147483647: continue
            try:
                response = client.measurements.list(s_id, "measurements", datetime_from=start_date.isoformat(), limit=5)
                # FIXED: Access results as an attribute instead of using .get()
                results = getattr(response, "results", [])
                for item in results:
                    p_obj = getattr(item, "parameter", None)
                    p_name = getattr(p_obj, "name", "").lower() if p_obj else ""
                    val = getattr(item, "value", None)
                    dt_obj = getattr(item, "date", None)
                    utc_dt = getattr(dt_obj, "utc", None) if dt_obj else None
                    if p_name in PARAMETER_MAP and val is not None:
                        measurements.append({
                            "date": pd.to_datetime(utc_dt) if utc_dt else datetime.utcnow(),
                            "parameter": PARAMETER_MAP[p_name],
                            "value": float(val)
                        })
            except Exception: pass
        if not measurements: return _create_fallback_dataframe(use_historical_fallback)
        df = pd.DataFrame(measurements).pivot_table(index="date", columns="parameter", values="value", aggfunc="mean").reset_index()
        for col in PARAMETER_MAP.values():
            if col not in df.columns: df[col] = np.nan
        return df.sort_values("date").reset_index(drop=True)
    except Exception: return _create_fallback_dataframe(use_historical_fallback)

def _create_fallback_dataframe(use_historical: bool = True):
    now = datetime.utcnow(); data = {"date": [now]}
    for col in PARAMETER_MAP.values(): data[col] = [np.nan]
    return pd.DataFrame(data)

def fetch_latest_single_reading(): return fetch_delhi_aqi(hours_back=1).tail(1)