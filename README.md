# Smart-Wind-IoT

# SmartWindIoT — AI Module

## Structure

```
SmartWindIoT_AI/
│
├── SmartWindIoT_dataset.csv        ← ton dataset (copier ici)
│
├── 01_data_preparation.py          ← Nettoyage + feature engineering + EDA
├── 02_power_prediction.py          ← Modèle de prédiction de puissance
├── 03_anomaly_maintenance.py       ← Détection d'anomalies + maintenance prédictive
│
├── requirements.txt
│
├── outputs/                        ← Graphiques générés automatiquement
│   ├── 01_time_series.png
│   ├── 02_correlation_matrix.png
│   ├── 03_feature_distributions.png
│   ├── 04_feature_importance_*.png
│   ├── 05_predicted_vs_actual.png
│   ├── 06_confusion_matrices.png
│   ├── 07_anomaly_feature_importance.png
│   ├── 08_maintenance_risk.png
│   └── SmartWindIoT_features.csv
│
└── models/                         ← Modèles entraînés (joblib)
    ├── power_prediction_model.joblib
    ├── anomaly_detection_model.joblib
    └── maintenance_model.joblib
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

Lancer les scripts dans l'ordre :

```bash
python 01_data_preparation.py
python 02_power_prediction.py
python 03_anomaly_maintenance.py
```

## Ce que fait chaque script

### 01 — Data Preparation
- Charge et nettoie le CSV
- Crée les features dérivées :
  - `hour`, `is_daytime` (cycle jour/nuit)
  - `efficiency` (puissance réelle / calculée)
  - `vib_rolling_mean`, `vib_rolling_std` (features MPU6050)
  - `voltage_delta` (instabilité tension)
- Génère 3 graphiques d'exploration

### 02 — Power Prediction
- Compare 3 modèles : Random Forest, Gradient Boosting, Ridge
- Sélectionne automatiquement le meilleur (R²)
- Sauvegarde le modèle pour réutilisation
- Fonction `predict_power()` pour l'intégration ESP32/MQTT

### 03 — Anomaly Detection & Predictive Maintenance
**Anomaly Detection (supervisé)**
- Classifie chaque lecture en : normal / warning / critical
- Modèles : Random Forest + SVM
- Gère le déséquilibre de classes (normal >> critical)

**Predictive Maintenance (non supervisé — MPU6050)**
- Isolation Forest entraîné sur données normales uniquement
- Score de risque de maintenance (0 = safe, 1 = urgent)
- Détecte les vibrations anormales avant qu'elles deviennent critiques
- Fonctions `detect_anomaly()` et `get_maintenance_risk()` pour ESP32/MQTT

## Intégration future (Node-RED / MQTT)

Les fonctions d'inférence sont prêtes pour recevoir les payloads MQTT :

```python
from 03_anomaly_maintenance import detect_anomaly, get_maintenance_risk

# Payload reçu via MQTT depuis l'ESP32
mqtt_data = {
    "temp": 28, "voltage": 12.3, "current": 1.1,
    "vibration": 7, "humidity": 58, "pressure": 1010, "power": 13.5
}

detect_anomaly(mqtt_data)
get_maintenance_risk(vibration=7, vib_rolling_mean=6.5, temp=28)
```

## Modèles utilisés

| Tâche | Modèle | Librairie |
|---|---|---|
| Power Prediction | Random Forest / Gradient Boosting / Ridge | scikit-learn |
| Anomaly Detection | Random Forest Classifier / SVM | scikit-learn |
| Predictive Maintenance | Isolation Forest | scikit-learn |
