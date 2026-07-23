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