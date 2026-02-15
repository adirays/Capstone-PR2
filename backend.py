import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# Load the trained model
try:
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Model file 'heart_model.pkl' not found. Please ensure the model is trained and saved.")

# Define the FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API to serve the Heart Disease Prediction Model",
    version="1.0.0"
)

# Define the input data schema
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

    class Config:
        schema_extra = {
            "example": {
                "age": 45,
                "sex": 1,
                "cp": 2,
                "trestbps": 120,
                "chol": 200,
                "fbs": 0,
                "restecg": 1,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 1,
                "ca": 0,
                "thal": 2
            }
        }

@app.get("/")
def home():
    return {"message": "Welcome to the Heart Disease Prediction API"}

@app.post("/predict")
def predict(data: HeartDiseaseInput):
    """
    Predict heart disease risk based on patient data.
    """
    try:
        # Convert input data to numpy array
        input_data = np.array([
            data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
            data.restecg, data.thalach, data.exang, data.oldpeak,
            data.slope, data.ca, data.thal
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Return result
        result = {
            "prediction": int(prediction),
            "risk_probability": float(probability),
            "high_risk": bool(prediction == 1)
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
