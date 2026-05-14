import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("../data/smartwind_dataset.csv", sep=";")
print(df.columns)
print(df.head())

print("Dataset loaded:", df.shape)

# Features (IMPORTANT: match ESP32 structure)
features = [
    "temp",
    "voltage",
    "current",
    "vibration",
    "humidity",
    "pressure"
]

X = df[features]
y = df["power"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

# Save model
joblib.dump(model, "../models/power_model.pkl")

print("Model saved successfully!")