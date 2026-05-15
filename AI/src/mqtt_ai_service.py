import json
import numpy as np
import joblib
import paho.mqtt.client as mqtt
import os

# ── MQTT CONFIG ─────────────────────────────
BROKER = "localhost"
PORT = 1883

INPUT_TOPIC = "smartwind/data"
OUTPUT_TOPIC = "smartwind/ai"

# ── PATH ────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "power_prediction_model.joblib")

power_model = joblib.load(MODEL_PATH)

print("AI Service started... waiting for MQTT data")

# ── STATE (for delta feature) ───────────────
last_voltage = None

# ── FEATURE ENGINEERING (MATCH TRAINING) ───
def build_features(payload):
    global last_voltage

    temp = payload["temp"]
    voltage = payload["voltage"]
    current = payload["current"]
    vibration = payload["vibration"]
    humidity = payload["humidity"]
    pressure = payload["pressure"]

    # engineered features (IMPORTANT)
    power_calc = voltage * current
    efficiency = payload.get("power", power_calc) / (power_calc + 1e-6)

    if last_voltage is None:
        voltage_delta = 0
    else:
        voltage_delta = abs(voltage - last_voltage)

    last_voltage = voltage

    return np.array([[

        temp,
        voltage,
        current,
        vibration,
        humidity,
        pressure,
        power_calc,
        efficiency,
        voltage_delta

    ]])

# ── RULE-BASED ANOMALY ─────────────────────
def detect_anomaly(payload):
    vibration = payload["vibration"]
    voltage = payload["voltage"]
    temp = payload["temp"]

    if vibration > 12 or voltage < 10.5 or temp > 38:
        return "critical"
    elif vibration > 8 or voltage < 11:
        return "warning"
    else:
        return "normal"

# ── MQTT CALLBACK ──────────────────────────
def on_message(client, userdata, msg):

    payload = json.loads(msg.payload.decode())
    print("\nReceived:", payload)

    # predict
    X = build_features(payload)
    predicted_power = float(power_model.predict(X)[0])

    # anomaly
    status = detect_anomaly(payload)

    real_power = payload.get("power", None)
    deviation = abs(real_power - predicted_power) if real_power else None

    result = {
        "predicted_power": round(predicted_power, 2),
        "real_power": real_power,
        "status": status,
        "deviation": round(deviation, 3) if deviation else None
    }

    client.publish(OUTPUT_TOPIC, json.dumps(result))

    print("AI Output:", result)

# ── MQTT SETUP ─────────────────────────────
client = mqtt.Client()
client.on_message = on_message

client.connect(BROKER, PORT, 1883)
client.subscribe(INPUT_TOPIC)

client.loop_forever()