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