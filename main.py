import os
import joblib
import numpy as np
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI Application
app = FastAPI(
    title="Agriculture Crop Yield Prediction API",
    description="API for predicting agricultural yields and dynamically retraining regression models.",
    version="1.0.0"
)
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# ---------------------------------------------------------
# MODEL & PIPELINE LOADING
# ---------------------------------------------------------
MODEL_PATH = "best_crop_model.pkl"
SCALER_PATH = "feature_scaler.pkl"
FEATURES_PATH = "feature_names.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("Successfully loaded ML artifacts into memory.")
except FileNotFoundError:
    print("Warning: ML artifact files not found. Please run training script or upload .pkl files.")
    model, scaler, feature_names = None, None, None

# ---------------------------------------------------------
# PYDANTIC INPUT VALIDATION MODELS
# ---------------------------------------------------------
class CropPredictionRequest(BaseModel):
    Rainfall_mm: float = Field(..., ge=0.0, le=3000.0, description="Total rainfall in millimeters (Range: 0 to 3000 mm)")
    Temperature_Celsius: float = Field(..., ge=-10.0, le=60.0, description="Average ambient temperature in Celsius (Range: -10 to 60 °C)")
    Fertilizer_Used: int = Field(..., ge=0, le=1, description="Binary flag for fertilizer usage (0 = No, 1 = Yes)")
    Irrigation_Used: int = Field(..., ge=0, le=1, description="Binary flag for irrigation usage (0 = No, 1 = Yes)")
    Days_to_Harvest: float = Field(..., ge=30.0, le=365.0, description="Total crop maturation period in days (Range: 30 to 365 days)")
    Region: str = Field(..., min_length=2, max_length=50, description="Geographical agricultural region (e.g., North, East, South, West)")
    Soil_Type: str = Field(..., min_length=2, max_length=50, description="Soil classification (e.g., Loam, Clay, Sandy, Silt, Peaty)")
    Crop: str = Field(..., min_length=2, max_length=50, description="Cultivated crop name (e.g., Wheat, Maize, Rice, Barley, Cotton)")
    Weather_Condition: str = Field(..., min_length=2, max_length=50, description="Dominant weather pattern (e.g., Sunny, Rainy, Cloudy)")

    class Config:
        json_schema_extra = {
            "example": {
                "Rainfall_mm": 750.0,
                "Temperature_Celsius": 25.5,
                "Fertilizer_Used": 1,
                "Irrigation_Used": 1,
                "Days_to_Harvest": 110.0,
                "Region": "North",
                "Soil_Type": "Loam",
                "Crop": "Wheat",
                "Weather_Condition": "Sunny"
            }
        }
class RetrainRecord(CropPredictionRequest):
    Yield_tons_per_hectare: float = Field(..., ge=0.0, le=30.0, description="Actual observed crop harvest yield in tons per hectare")

class RetrainRequest(BaseModel):
    data: List[RetrainRecord] = Field(..., min_items=1, description="List of new agricultural records to update the regression model")

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def preprocess_input(record: CropPredictionRequest) -> np.ndarray:
    """Converts a raw JSON request into a Z-score scaled, one-hot encoded NumPy array matching training features."""
    if feature_names is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded on the server.")

    df_template = pd.DataFrame(0.0, index=[0], columns=feature_names)

    df_template.at[0, "Rainfall_mm"] = record.Rainfall_mm
    df_template.at[0, "Temperature_Celsius"] = record.Temperature_Celsius
    df_template.at[0, "Fertilizer_Used"] = float(record.Fertilizer_Used)
    df_template.at[0, "Irrigation_Used"] = float(record.Irrigation_Used)
    df_template.at[0, "Days_to_Harvest"] = record.Days_to_Harvest
    categorical_mappings = {
        f"Region_{record.Region}": 1.0,
        f"Soil_Type_{record.Soil_Type}": 1.0,
        f"Crop_{record.Crop}": 1.0,
        f"Weather_Condition_{record.Weather_Condition}": 1.0
    }

    for col_name, val in categorical_mappings.items():
        if col_name in df_template.columns:
            df_template.at[0, col_name] = val
    return scaler.transform(df_template.values)

# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------
@app.get("/", status_code=status.HTTP_200_OK)
def health_check():
    """Returns basic API status and confirmation of loaded models."""
    return {
        "status": "active",
        "model_loaded": model is not None,
        "docs_url": "/docs"
    }

@app.post("/predict", status_code=status.HTTP_200_OK)
def predict_yield(payload: CropPredictionRequest):
    """Takes agricultural parameters and returns the predicted crop yield in tons/hectare."""
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not initialized.")
    
    try:
        processed_matrix = preprocess_input(payload)
        prediction = model.predict(processed_matrix)[0]
        return {
            "prediction_status": "success",
            "predicted_yield_tons_per_hectare": round(float(prediction), 4),
            "input_parameters": payload.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Prediction error: {str(e)}")