from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from typing import Optional
import traceback
from tensorflow.keras.models import load_model # type: ignore

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or set to ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the trained model correctly
model = load_model("saved_model/cardio_model.h5")  

# Define input data format using Pydantic
class PatientData(BaseModel):
    age: int
    gender: int
    height: int
    weight: float
    ap_hi: int
    ap_lo: int
    smoke: int
    alco: int
    active: int
    gluc_2: int
    gluc_3: int
    cholesterol_2: Optional[int] = 0  # Optional with default value
    cholesterol_3: Optional[int] = 0  # Optional with default value

# Home route (optional, just for checking)
@app.get("/")
def home():
    return {"message": "Cardiovascular Disease Prediction API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input to NumPy array (ensure only 12 features)
        input_data = np.array([[data.age, data.gender, data.height, data.weight,
                                data.ap_hi, data.ap_lo, data.smoke, data.alco,
                                data.active, data.gluc_2, data.gluc_3]])  # Only 12 features

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return result
        return {"cardiovascular_risk": "Yes" if prediction >= 0.5 else "No"}  

    except Exception as e:
        error_message = str(e)
        traceback.print_exc()  # Print the error in the terminal
        raise HTTPException(status_code=500, detail=f"Error: {error_message}")
