import joblib
import pandas as pd

# Load trained model
model = joblib.load("../models/status_model.pkl")

# Simulated ESP32 data
sample = pd.DataFrame([{
    "temp": 38,
    "voltage": 10.5,
    "current": 0.4,
    "vibration": 13,
    "humidity": 80,
    "pressure": 1000
}])

# Predict status
prediction = model.predict(sample)

print("Predicted Turbine Status:", prediction[0])