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