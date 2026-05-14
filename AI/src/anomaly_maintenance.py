"""
SmartWindIoT — Script 03 : Anomaly Detection & Predictive Maintenance
======================================================================
Deux tâches complémentaires :

  1. Anomaly Detection (classification)
     → Classifie chaque lecture capteur en : normal / warning / critical
     → Modèles : Random Forest Classifier + SVM

  2. Predictive Maintenance (MPU6050 — vibrations)
     → Détecte les patterns de vibration anormaux AVANT qu'ils deviennent critiques
     → Modèle : Isolation Forest (détection non supervisée)
     → Score de risque de maintenance en sortie

Run : python 03_anomaly_maintenance.py
Prérequis : lancer d'abord 01_data_preparation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (classification_report, confusion_matrix,
                                      ConfusionMatrixDisplay, accuracy_score)
from sklearn.ensemble         import RandomForestClassifier, IsolationForest
from sklearn.svm              import SVC

# ── Config ───────────────────────────────────────────────────────────────────


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "outputs", "SmartWindIoT_features.csv")
OUTPUT_DIR = "outputs"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Features pour la classification d'anomalie
ANOMALY_FEATURES = [
    "temp", "voltage", "current", "vibration",
    "humidity", "pressure", "power",
    "efficiency", "voltage_delta",
    "vib_rolling_mean", "vib_rolling_std"
]

# Features spécifiques vibration (MPU6050) pour maintenance prédictive
MAINTENANCE_FEATURES = [
    "vibration", "vib_rolling_mean", "vib_rolling_std",
    "vibration_sq", "temp", "current"
]

# ── Chargement ────────────────────────────────────────────────────────────────
print("=" * 60)
print("  SmartWindIoT — Anomaly Detection & Predictive Maintenance")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)
print(f"\n[INFO] Dataset chargé : {len(df)} lignes")

# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 1 — ANOMALY DETECTION (Classification supervisée)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  PARTIE 1 : Anomaly Detection (classification)")
print("─" * 60)

X_cls = df[ANOMALY_FEATURES]
y_cls = df["status"]      # normal / warning / critical

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)
print(f"[INFO] Train: {len(X_train_c)} | Test: {len(X_test_c)}")
print(f"[INFO] Classes : {y_cls.value_counts().to_dict()}")

# Modèles de classification
classifiers = {
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",       # gère le déséquilibre normal>>critical
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ]),
}

cls_results = {}
print("\n--- Entraînement classifieurs ---")

for name, pipeline in classifiers.items():
    print(f"\n  [{name}]")
    pipeline.fit(X_train_c, y_train_c)
    y_pred = pipeline.predict(X_test_c)

    acc = accuracy_score(y_test_c, y_pred)
    cv  = cross_val_score(pipeline, X_train_c, y_train_c,
                          cv=5, scoring="accuracy", n_jobs=-1)

    cls_results[name] = {
        "pipeline": pipeline,
        "y_pred":   y_pred,
        "accuracy": acc,
        "cv_mean":  cv.mean(),
        "cv_std":   cv.std(),
    }

    print(f"    Accuracy = {acc:.4f}")
    print(f"    CV Acc   = {cv.mean():.4f} ± {cv.std():.4f}")
    print(classification_report(y_test_c, y_pred, zero_division=0))

# Meilleur classifieur
best_cls_name = max(cls_results, key=lambda k: cls_results[k]["accuracy"])
best_cls      = cls_results[best_cls_name]
print(f"\n[INFO] ★ Meilleur classifieur : {best_cls_name} "
      f"(Accuracy = {best_cls['accuracy']:.4f})")

joblib.dump(best_cls["pipeline"], f"{MODELS_DIR}/anomaly_detection_model.joblib")
print(f"[INFO] Modèle sauvegardé → {MODELS_DIR}/anomaly_detection_model.joblib")

# Matrices de confusion (côte à côte)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Anomaly Detection — Matrices de confusion", fontweight="bold")
labels = ["normal", "warning", "critical"]
colors = ["#27ae60", "#f39c12", "#e74c3c"]

for ax, (name, res) in zip(axes, cls_results.items()):
    cm = confusion_matrix(y_test_c, res["y_pred"], labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAccuracy = {res['accuracy']:.4f}")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Sauvegardé : {OUTPUT_DIR}/06_confusion_matrices.png")

# Feature importance pour le classifieur Random Forest
rf_pipe = cls_results["Random Forest"]["pipeline"]
importances = rf_pipe.named_steps["model"].feature_importances_
fi_df = pd.DataFrame({
    "feature": ANOMALY_FEATURES,
    "importance": importances
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(fi_df["feature"], fi_df["importance"],
        color="#e67e22", edgecolor="white", linewidth=0.5)
ax.set_title("Feature Importance — Anomaly Detection (RF)", fontweight="bold")
ax.set_xlabel("Importance")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_anomaly_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Sauvegardé : {OUTPUT_DIR}/07_anomaly_feature_importance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 — PREDICTIVE MAINTENANCE (MPU6050 — Isolation Forest)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  PARTIE 2 : Predictive Maintenance (MPU6050 vibrations)")
print("─" * 60)

# Entraîner uniquement sur les données "normal" pour apprendre le comportement sain
df_normal = df[df["status"] == "normal"][MAINTENANCE_FEATURES]
print(f"[INFO] Entraînement sur données normales uniquement : {len(df_normal)} lignes")

# Isolation Forest : détecte les points anormaux sans labels
iso_forest = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  IsolationForest(
        n_estimators=200,
        contamination=0.08,      # ~8% d'anomalies attendues
        random_state=42,
        n_jobs=-1
    ))
])

iso_forest.fit(df_normal)
print("[INFO] Isolation Forest entraîné ✓")

# Prédire sur tout le dataset
X_maint = df[MAINTENANCE_FEATURES]
preds   = iso_forest.predict(X_maint)          # 1=normal, -1=anomalie
scores  = iso_forest.decision_function(X_maint) # score (plus bas = plus anormal)

# Normaliser le score en "risque de maintenance" (0=safe, 1=urgent)
risk_score = 1 - (scores - scores.min()) / (scores.max() - scores.min())
df["maintenance_risk"] = risk_score
df["maintenance_flag"] = (preds == -1).astype(int)

# Stats
n_flagged = df["maintenance_flag"].sum()
print(f"[INFO] Points flaggés comme anormaux : {n_flagged} "
      f"({n_flagged/len(df)*100:.1f}%)")
print(f"[INFO] Score de risque moyen : {risk_score.mean():.3f}")

# Corrélation flag vs statut réel (validation)
print("\n--- Corrélation flag maintenance vs statut réel ---")
cross = pd.crosstab(df["status"], df["maintenance_flag"],
                    rownames=["Statut réel"], colnames=["Flag maintenance"])
print(cross)

# Visualisation : score de risque dans le temps
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("SmartWindIoT — Predictive Maintenance (MPU6050)",
             fontsize=13, fontweight="bold")

# Vibration brute
axes[0].plot(df.index, df["vibration"], color="#e67e22", linewidth=0.7, alpha=0.8)
axes[0].axhline(10, color="red", linestyle="--", linewidth=1, label="Seuil critique")
axes[0].set_ylabel("Vibration")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# Moyenne glissante vibration
axes[1].plot(df.index, df["vib_rolling_mean"], color="#9b59b6",
             linewidth=1, label="Moy. glissante (30 min)")
axes[1].fill_between(df.index,
                     df["vib_rolling_mean"] - df["vib_rolling_std"],
                     df["vib_rolling_mean"] + df["vib_rolling_std"],
                     alpha=0.2, color="#9b59b6")
axes[1].set_ylabel("Vibration (moy.)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# Score de risque maintenance
axes[2].fill_between(df.index, df["maintenance_risk"],
                     alpha=0.5, color="#e74c3c", label="Risque maintenance")
axes[2].plot(df.index, df["maintenance_risk"], color="#c0392b", linewidth=0.5)
axes[2].axhline(0.7, color="red", linestyle="--", linewidth=1, label="Seuil alerte (0.7)")
axes[2].set_ylabel("Score de risque")
axes[2].set_ylim(0, 1)
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)
axes[2].set_xlabel("Index temporel")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_maintenance_risk.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Sauvegardé : {OUTPUT_DIR}/08_maintenance_risk.png")

# Sauvegarde du modèle de maintenance
joblib.dump(iso_forest, f"{MODELS_DIR}/maintenance_model.joblib")
print(f"[INFO] Modèle sauvegardé → {MODELS_DIR}/maintenance_model.joblib")

# Sauvegarder dataset final avec scores
df.to_csv(f"{OUTPUT_DIR}/SmartWindIoT_final.csv", index=False)
print(f"[INFO] Dataset final sauvegardé → {OUTPUT_DIR}/SmartWindIoT_final.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS D'INFÉRENCE RÉUTILISABLES (à appeler avec données MQTT en temps réel)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_anomaly(mqtt_payload: dict) -> dict:
    """
    Classifie une lecture ESP32 (MQTT JSON) comme normal/warning/critical.

    Exemple d'appel :
        detect_anomaly({
            "temp": 38, "voltage": 10.5, "current": 0.9,
            "vibration": 12, "humidity": 55, "pressure": 1010, "power": 9
        })
    """
    model = joblib.load(f"{MODELS_DIR}/anomaly_detection_model.joblib")

    # Calculer les features dérivées
    power_calc    = mqtt_payload["voltage"] * mqtt_payload["current"]
    efficiency    = (mqtt_payload["power"] / power_calc) if power_calc > 0 else 0.95
    voltage_delta = 0.0   # inconnu sans historique → 0 par défaut
    vib_rolling_mean = mqtt_payload["vibration"]
    vib_rolling_std  = 0.0

    X_new = pd.DataFrame([{
        "temp":             mqtt_payload["temp"],
        "voltage":          mqtt_payload["voltage"],
        "current":          mqtt_payload["current"],
        "vibration":        mqtt_payload["vibration"],
        "humidity":         mqtt_payload["humidity"],
        "pressure":         mqtt_payload["pressure"],
        "power":            mqtt_payload["power"],
        "efficiency":       min(efficiency, 1.0),
        "voltage_delta":    voltage_delta,
        "vib_rolling_mean": vib_rolling_mean,
        "vib_rolling_std":  vib_rolling_std,
    }])

    status = model.predict(X_new)[0]
    proba  = model.predict_proba(X_new)[0]
    classes = model.classes_

    result = {
        "status":      status,
        "confidence":  float(max(proba)),
        "probabilities": dict(zip(classes, proba.round(3)))
    }
    print(f"  → Statut détecté : {status} (confiance: {result['confidence']:.1%})")
    print(f"     Probabilités  : {result['probabilities']}")
    return result


def get_maintenance_risk(vibration, vib_rolling_mean=None, vib_rolling_std=0,
                         temp=25, current=1.0) -> dict:
    """
    Calcule le score de risque de maintenance à partir des données MPU6050.

    Exemple d'appel :
        get_maintenance_risk(vibration=11.5, vib_rolling_mean=9.0, temp=32)
    """
    model = joblib.load(f"{MODELS_DIR}/maintenance_model.joblib")

    if vib_rolling_mean is None:
        vib_rolling_mean = vibration

    X_new = pd.DataFrame([{
        "vibration":        vibration,
        "vib_rolling_mean": vib_rolling_mean,
        "vib_rolling_std":  vib_rolling_std,
        "vibration_sq":     vibration ** 2,
        "temp":             temp,
        "current":          current,
    }])

    score    = model.decision_function(X_new)[0]
    flagged  = model.predict(X_new)[0] == -1

    # Score normalisé (approximation relative)
    risk = max(0.0, min(1.0, 0.5 - score))

    level = "🔴 URGENT" if risk > 0.7 else "🟡 Attention" if risk > 0.4 else "🟢 Normal"

    result = {
        "risk_score":   round(risk, 3),
        "flag":         flagged,
        "level":        level,
    }
    print(f"  → Risque maintenance : {risk:.3f}  {level}")
    return result


# ── Tests rapides ─────────────────────────────────────────────────────────────
print("\n--- Tests d'inférence temps réel ---")

print("\n[Test 1] Données normales ESP32:")
detect_anomaly({
    "temp": 25, "voltage": 12.5, "current": 1.1,
    "vibration": 4, "humidity": 55, "pressure": 1012, "power": 13.5
})

print("\n[Test 2] Données critiques ESP32:")
detect_anomaly({
    "temp": 39, "voltage": 10.2, "current": 0.4,
    "vibration": 13.5, "humidity": 80, "pressure": 998, "power": 3.8
})

print("\n[Test 3] Score de maintenance — vibration élevée:")
get_maintenance_risk(vibration=12.0, vib_rolling_mean=10.5,
                     vib_rolling_std=2.1, temp=34, current=0.8)

print("\n[Test 4] Score de maintenance — vibration normale:")
get_maintenance_risk(vibration=3.5, vib_rolling_mean=4.0,
                     vib_rolling_std=0.5, temp=25, current=1.1)

print("\n" + "=" * 60)
print("  ✓ Anomaly Detection & Predictive Maintenance terminés")
print("")
print("  Modèles sauvegardés dans /models/ :")
print("    - power_prediction_model.joblib")
print("    - anomaly_detection_model.joblib")
print("    - maintenance_model.joblib")
print("")
print("  Graphiques sauvegardés dans /outputs/")
print("=" * 60)
