import os
import requests
import pandas as pd
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()

from predict import predict, predict_proba, load_artifacts


WAQI_API_KEY = os.getenv("WAQI_API_KEY")

BASE_URL = "https://api.waqi.info/feed/delhi/"


# -------------------------------
# Fetch Live Data
# -------------------------------
def fetch_delhi_data():

    if not WAQI_API_KEY:
        raise Exception("WAQI_API_KEY environment variable not set")

    params = {
        "token": WAQI_API_KEY
    }

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):

        try:
            res = requests.get(
                BASE_URL,
                params=params,
                timeout=15
            )

            res.raise_for_status()
            data = res.json()

            if data.get("status") != "ok":
                raise Exception(f"WAQI API Error: {data.get('data')}")

            data_node = data["data"]
            iaqi = data_node.get("iaqi", {})

            station_name = data_node.get("city", {}).get("name", "Unknown")
            last_updated = data_node.get("time", {}).get("s", "Unknown")

            values = {
                "station": station_name,
                "time": last_updated
            }

            param_mapping = {
                "pm25": "pm25",
                "pm10": "pm10",
                "co": "co",
                "no2": "no2",
                "o3": "o3",
                "so2": "so2",
                "nh3": "nh3",
                "no": "no"
            }

            for waqi_param, model_param in param_mapping.items():
                values[model_param] = iaqi.get(waqi_param, {}).get("v", 0)

            # Safety rule
            if values["pm10"] < values["pm25"]:
                values["pm10"] = values["pm25"]

            values["wind_speed"] = iaqi.get("w", {}).get("v", 0)
            values["wind_dir"] = iaqi.get("wd", {}).get("v", 0)

            return values


        except Exception as e:

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2

            else:
                raise Exception(f"WAQI API Failed: {str(e)}")


# -------------------------------
# Main Prediction Function (API ENTRY)
# -------------------------------
def get_live_prediction():

    try:
        values = fetch_delhi_data()
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

    # Prepare DataFrame
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

        "wind_speed": values.get("wind_speed", 0),
        "wind_dir": values.get("wind_dir", 0),

        "fire_count": 0
    }])

    # Predict
    label = predict(df)[0]
    probs = predict_proba(df)[0]

    _, encoder, _ = load_artifacts()

    confidence = f"{probs.max():.2%}"

    probabilities = {
        source: f"{prob:.2%}"
        for source, prob in zip(encoder.classes_, probs)
    }

    # Final Output
    result = {

        "status": "success",

        "station": values.get("station"),
        "last_updated": values.get("time"),
        "generated_at": datetime.utcnow().isoformat(),

        "inputs": {
            "pm25": values.get("pm25", 0),
            "pm10": values.get("pm10", 0),
            "no2": values.get("no2", 0),
            "so2": values.get("so2", 0),
            "nh3": values.get("nh3", 0),
            "co": values.get("co", 0),
            "o3": values.get("o3", 0),
            "no": values.get("no", 0)
        },

        "prediction": {
            "label": str(label),
            "confidence": confidence,
            "probabilities": probabilities
        }
    }

    return result

if __name__ == "__main__":
    import json
    result = get_live_prediction()
    print(json.dumps(result, indent=4))
