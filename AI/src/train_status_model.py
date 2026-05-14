import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("../data/smartwind_dataset.csv", sep=";")

print("Dataset loaded:", df.shape)

# Features
features = [
    "temp",
    "voltage",
    "current",
    "vibration",
    "humidity",
    "pressure"
]

X = df[features]

# Target
y = df["status"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create classification model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "../models/status_model.pkl")

print("Status model saved successfully!")