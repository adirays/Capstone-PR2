import requests
import json

url = "http://127.0.0.1:8000/predict"

data = {
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

try:
    response = requests.post(url, json=data)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
