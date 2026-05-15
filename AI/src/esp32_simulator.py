"""
SmartWindIoT — ESP32 Simulator
================================
Simule les messages MQTT de l'ESP32 sans avoir le hardware.
Envoie des données capteurs réalistes toutes les 5 secondes.

Run : python esp32_simulator.py
Prérequis : pip install paho-mqtt
"""

import paho.mqtt.client as mqtt
import json
import time
import random
import math
from datetime import datetime

# ── Config MQTT ───────────────────────────────────────────────────────────────
BROKER_HOST = "localhost"       # adresse du broker Mosquitto
BROKER_PORT = 1883
TOPIC_DATA  = "smartwind/data"
TOPIC_STATUS = "smartwind/simulator/status"

# ── Paramètres de simulation ──────────────────────────────────────────────────
INTERVAL_SECONDS = 5            # intervalle entre chaque envoi
ANOMALY_CHANCE   = 0.12         # 12% de chance d'anomalie par lecture

# ── Callbacks MQTT ────────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connecté au broker {BROKER_HOST}:{BROKER_PORT} ✓")
        client.publish(TOPIC_STATUS, json.dumps({"simulator": "online"}))
    else:
        print(f"[MQTT] Échec connexion, code : {rc}")

def on_disconnect(client, userdata, rc):
    print("[MQTT] Déconnecté du broker")

# ── Générateur de données capteurs ────────────────────────────────────────────
def generate_sensor_data(tick: int) -> dict:
    """
    Génère des données capteurs réalistes avec cycle jour/nuit
    et injection aléatoire d'anomalies.
    """
    # Cycle jour/nuit (période de 288 ticks = 24h à 5 sec/tick)
    hour_sim    = (tick % 288) / 12          # heure simulée (0-24)
    day_factor  = math.sin(math.pi * hour_sim / 12) * 0.5 + 0.5

    # Décider le scénario
    scenario = "normal"
    if random.random() < ANOMALY_CHANCE:
        scenario = random.choice([
            "high_vibration",
            "overheat",
            "low_voltage",
            "power_drop"
        ])

    # Valeurs de base
    temp      = round(20 + 10 * day_factor + random.gauss(0, 1.0), 2)
    voltage   = round(11.5 + 2.0 * day_factor + random.gauss(0, 0.1), 3)
    current   = round(0.3 + 1.5 * day_factor + random.gauss(0, 0.05), 3)
    vibration = round(2 + 5 * day_factor + random.gauss(0, 0.4), 2)
    humidity  = round(65 - 20 * day_factor + random.gauss(0, 2), 2)
    pressure  = round(1013 + random.gauss(0, 4), 2)

    # Injection d'anomalie
    if scenario == "high_vibration":
        vibration = round(random.uniform(11, 15), 2)
        print(f"  ⚠ Anomalie injectée : HIGH VIBRATION ({vibration})")

    elif scenario == "overheat":
        temp = round(random.uniform(36, 40), 2)
        print(f"  ⚠ Anomalie injectée : OVERHEAT ({temp}°C)")

    elif scenario == "low_voltage":
        voltage = round(random.uniform(10.0, 10.8), 3)
        print(f"  ⚠ Anomalie injectée : LOW VOLTAGE ({voltage}V)")

    elif scenario == "power_drop":
        current = round(random.uniform(0.1, 0.3), 3)
        voltage = round(random.uniform(10.5, 11.5), 3)
        print(f"  ⚠ Anomalie injectée : POWER DROP (V={voltage}, I={current})")

    # Clamp dans les plages réelles
    temp      = max(15.0,  min(40.0,  temp))
    voltage   = max(10.0,  min(14.0,  voltage))
    current   = max(0.1,   min(2.0,   current))
    vibration = max(0.0,   min(15.0,  vibration))
    humidity  = max(30.0,  min(90.0,  humidity))
    pressure  = max(980.0, min(1035.0, pressure))
    power = round(voltage * current * random.uniform(0.85, 0.98), 3)

    return {
        "temp":      temp,
        "voltage":   voltage,
        "current":   current,
        "vibration": vibration,
        "humidity":  humidity,
        "pressure":  pressure,
        "power": power,
        "timestamp": datetime.now().isoformat()
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SmartWindIoT — ESP32 Simulator")
    print("=" * 60)
    print(f"  Broker  : {BROKER_HOST}:{BROKER_PORT}")
    print(f"  Topic   : {TOPIC_DATA}")
    print(f"  Intervalle : {INTERVAL_SECONDS}s")
    print(f"  Anomaly chance : {ANOMALY_CHANCE*100:.0f}%")
    print("  Ctrl+C pour arrêter")
    print("=" * 60)

    client = mqtt.Client(client_id="esp32_simulator")
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect

    try:
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
        client.loop_start()
        time.sleep(1)  # attendre connexion

        tick = 0
        while True:
            tick += 1
            data = generate_sensor_data(tick)

            payload = json.dumps(data)
            result  = client.publish(TOPIC_DATA, payload, qos=1)

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{ts}] Tick #{tick} → publié sur {TOPIC_DATA}")
            print(f"  temp={data['temp']}°C  voltage={data['voltage']}V  "
                  f"current={data['current']}A  vibration={data['vibration']}")
            print(f"  humidity={data['humidity']}%  pressure={data['pressure']}hPa")

            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n\n[INFO] Simulateur arrêté.")
        client.publish(TOPIC_STATUS, json.dumps({"simulator": "offline"}))
        client.loop_stop()
        client.disconnect()

    except ConnectionRefusedError:
        print("\n[ERREUR] Impossible de se connecter au broker MQTT.")
        print("  → Vérifier que Mosquitto tourne : 'mosquitto -v'")
        print(f"  → Ou changer BROKER_HOST si le broker est sur une autre machine")

if __name__ == "__main__":
    main()