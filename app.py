from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from predict import predict, predict_proba

app = FastAPI(title="Pollution Source Identification API")


# Input schema
class PollutionInput(BaseModel):
    pm25: float
    pm10: float
    co: float
    no: float
    no2: float
    o3: float
    so2: float
    nh3: float
    wind_speed: float
    wind_dir: float
    fire_count: int


@app.get("/")
def home():
    return {"status": "API is running successfully 🚀"}


@app.post("/predict")
def predict_source(data: PollutionInput):

    df = pd.DataFrame(
        {
            "date": [pd.Timestamp.now()],
            "PM25": [data.pm25],
            "PM10": [data.pm10],
            "co": [data.co],
            "no": [data.no],
            "no2": [data.no2],
            "o3": [data.o3],
            "so2": [data.so2],
            "nh3": [data.nh3],
            "wind_speed": [data.wind_speed],
            "wind_dir": [data.wind_dir],
            "fire_count": [data.fire_count],
        }
    )

    label = predict(df)[0]
    probs = predict_proba(df)[0]

    return {
        "prediction": label,
        "probabilities": probs.tolist(),
    }
