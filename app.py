import time
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "ctr_model.pkl"

app = FastAPI(title="velocity-ctr-service", docs_url="/docs")


class PredictionRequest(BaseModel):
    user_id: str = Field(..., description="User identifier, e.g. 'user_101'")
    ad_id: str = Field(..., description="Ad identifier")
    ad_position: str = Field(..., description="Ad position: banner/native/interstitial")


class PredictionResponse(BaseModel):
    ad_id: str
    click_probability: float
    server_latency_ms: float


# In-memory low-latency feature store (mock)
FEATURE_STORE: Dict[str, Dict[str, str]] = {
    "user_101": {"device_os": "iOS", "app_category": "Social"},
    "user_102": {"device_os": "Android", "app_category": "Gaming"},
    "user_103": {"device_os": "Other", "app_category": "News"},
}

# Default fallback for unknown users
DEFAULT_USER_FEATURES = {"device_os": "Android", "app_category": "Gaming"}


def init_model():
    """Load the trained pipeline into memory and attach it to app state."""
    if getattr(app.state, "model", None) is not None:
        return app.state.model

    try:
        model = joblib.load(MODEL_PATH)
        app.state.model = model
        print(f"Loaded model from {MODEL_PATH}")
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {exc}")


@app.on_event("startup")
def load_model():
    init_model()


@app.post("/predict_ctr", response_model=PredictionResponse)
def predict_ctr(req: PredictionRequest):
    start_ts = time.time()

    # Fetch user-level features from the in-memory feature store
    user_feats = FEATURE_STORE.get(req.user_id, DEFAULT_USER_FEATURES)

    device_os = user_feats.get("device_os", DEFAULT_USER_FEATURES["device_os"])
    app_category = user_feats.get("app_category", DEFAULT_USER_FEATURES["app_category"])

    # Current hour (0-23)
    hour_of_day = int(time.localtime().tm_hour)

    # Build single-row DataFrame matching training layout
    row = {
        "device_os": device_os,
        "ad_position": req.ad_position,
        "app_category": app_category,
        "hour_of_day": hour_of_day,
    }

    try:
        df = pd.DataFrame([row])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to construct input features: {exc}")

    # Run prediction using loaded model
    model = getattr(app.state, "model", None)
    if model is None:
        model = init_model()

    try:
        proba = model.predict_proba(df)
        click_prob = float(proba[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}")

    latency_ms = (time.time() - start_ts) * 1000.0

    resp = PredictionResponse(
        ad_id=req.ad_id,
        click_probability=round(click_prob, 4),
        server_latency_ms=round(latency_ms, 3),
    )
    return resp


if __name__ == "__main__":
    import uvicorn

    # Recommended for production: run via ASGI server with multiple workers (e.g. gunicorn + uvicorn workers).
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1, log_level="info")
