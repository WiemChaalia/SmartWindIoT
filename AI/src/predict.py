import joblib
import pandas as pd

# Load trained model
model = joblib.load("../models/power_model.pkl")

# Simulated ESP32 data
sample = pd.DataFrame([{
    "temp": 27,
    "voltage": 12.4,
    "current": 0.9,
    "vibration": 5,
    "humidity": 58,
    "pressure": 1011
}])

# Predict power
prediction = model.predict(sample)

print("Predicted Power:", round(prediction[0], 2), "W")