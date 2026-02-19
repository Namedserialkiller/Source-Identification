import os
import requests
import pandas as pd
from datetime import datetime

from predict import predict, predict_proba


OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

BASE_URL = "https://api.openaq.org/v2/latest"


def fetch_delhi_data():
    if not OPENAQ_API_KEY:
        raise Exception("OPENAQ_API_KEY not set")

    headers = {
        "X-API-Key": OPENAQ_API_KEY
    }

    params = {
        "city": "Delhi",
        "limit": 1
    }

    res = requests.get(
        BASE_URL,
        headers=headers,
        params=params,
        timeout=15
    )

    if res.status_code != 200:
        raise Exception(f"OpenAQ error: {res.status_code}")

    data = res.json()

    if "results" not in data or len(data["results"]) == 0:
        raise Exception("No OpenAQ data found")

    measurements = data["results"][0]["measurements"]

    values = {}

    for m in measurements:
        values[m["parameter"]] = m["value"]

    return values


def predict_live_delhi():

    values = fetch_delhi_data()

    df = pd.DataFrame([{
        "date": datetime.utcnow(),

        "pm2_5": values.get("pm25", 0),
        "pm10": values.get("pm10", 0),
        "co": values.get("co", 0),
        "no2": values.get("no2", 0),
        "o3": values.get("o3", 0),
        "so2": values.get("so2", 0),
        "nh3": values.get("nh3", 0),
        "no": values.get("no", 0),

        "wind_speed": 0,
        "wind_dir": 0,
        "fire_count": 0,
    }])

    label = predict(df)[0]
    probs = predict_proba(df)[0]
