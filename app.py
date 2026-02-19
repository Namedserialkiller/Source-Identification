from fastapi import FastAPI
import pandas as pd
from datetime import datetime
from predict import predict, predict_proba
from fetch_live_data import get_live_prediction



app = FastAPI(
    title="Delhi Pollution Source Identification API",
    description="Predicts pollution source using ML",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Pollution Source API is live"
    }


@app.post("/predict")
def predict_source(data: dict):

    """
    Input JSON Example:
    {
        "pm25": 120,
        "pm10": 200,
        "no2": 35,
        "so2": 8,
        "nh3": 15,
        "co": 800,
        "o3": 20,
        "no": 10,
        "wind_speed": 3,
        "wind_dir": 90,
        "fire_count": 0
    }
    """

    try:

        df = pd.DataFrame([{
            "date": datetime.utcnow(),

            "PM25": data.get("pm25", 0),
            "PM10": data.get("pm10", 0),

            "no2": data.get("no2", 0),
            "so2": data.get("so2", 0),
            "nh3": data.get("nh3", 0),
            "co": data.get("co", 0),
            "o3": data.get("o3", 0),
            "no": data.get("no", 0),

            "wind_speed": data.get("wind_speed", 0),
            "wind_dir": data.get("wind_dir", 0),

            "fire_count": data.get("fire_count", 0)
        }])

        label = predict(df)[0]
        probs = predict_proba(df)[0]

        confidence = float(probs.max())

        return {
            "status": "success",
            "prediction": label,
            "confidence": round(confidence, 4),
            "probabilities": probs.tolist()
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }
# ------------------------------
# LIVE WAQI PREDICTION
# ------------------------------
@app.get("/predict-live")
def predict_live():

    result = get_live_prediction()
    return result
