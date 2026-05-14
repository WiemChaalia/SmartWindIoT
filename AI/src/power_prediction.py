"""
SmartWindIoT — Script 02 : Power Prediction
============================================
Entraîne et compare 3 modèles de régression pour prédire
la puissance générée par la turbine.

Modèles testés :
  - Random Forest Regressor    (robuste, interprétable)
  - Gradient Boosting          (haute précision)
  - Ridge Regression           (baseline linéaire)

Run : python 02_power_prediction.py
Prérequis : lancer d'abord 01_data_preparation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model    import Ridge

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "outputs", "SmartWindIoT_features.csv")
OUTPUT_DIR  = "outputs"
MODELS_DIR  = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


FEATURES = [
    "voltage", "current", "temp", "humidity",
    "pressure", "hour", "is_daytime",
    "voltage_delta", "efficiency"
]
TARGET = "power"

# ── Chargement ────────────────────────────────────────────────────────────────
print("=" * 60)
print("  SmartWindIoT — Power Prediction")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)
print(f"\n[INFO] Dataset chargé : {len(df)} lignes")

X = df[FEATURES]
y = df[TARGET]

# ── Split train / test ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

# ── Définition des modèles ────────────────────────────────────────────────────
models = {
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ))
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0))
    ]),
}

# ── Entraînement et évaluation ────────────────────────────────────────────────
results = {}
print("\n--- Entraînement des modèles ---")

for name, pipeline in models.items():
    print(f"\n  [{name}]")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # Cross-validation sur le train set
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=5, scoring="r2", n_jobs=-1)

    results[name] = {
        "pipeline": pipeline,
        "y_pred":   y_pred,
        "MAE":      mae,
        "RMSE":     rmse,
        "R2":       r2,
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std":  cv_scores.std(),
    }

    print(f"    MAE  = {mae:.3f} W")
    print(f"    RMSE = {rmse:.3f} W")
    print(f"    R²   = {r2:.4f}")
    print(f"    CV R²= {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Sélection du meilleur modèle ──────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["R2"])
best      = results[best_name]
print(f"\n[INFO] ★ Meilleur modèle : {best_name} (R² = {best['R2']:.4f})")

# Sauvegarde
model_path = f"{MODELS_DIR}/power_prediction_model.joblib"
joblib.dump(best["pipeline"], model_path)
print(f"[INFO] Modèle sauvegardé → {model_path}")

# ── Feature Importance (Random Forest ou Gradient Boosting) ──────────────────
for name in ["Random Forest", "Gradient Boosting"]:
    if name in results:
        inner_model = results[name]["pipeline"].named_steps["model"]
        importances = inner_model.feature_importances_
        fi_df = pd.DataFrame({
            "feature": FEATURES,
            "importance": importances
        }).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(fi_df["feature"], fi_df["importance"],
                       color="#3498db", edgecolor="white", linewidth=0.5)
        ax.set_title(f"Feature Importance — {name}", fontweight="bold")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, fi_df["importance"]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        safe_name = name.replace(" ", "_").lower()
        plt.savefig(f"{OUTPUT_DIR}/04_feature_importance_{safe_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Feature importance sauvegardée pour {name}")

# ── Visualisation : Prédit vs Réel ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Power Prediction — Prédit vs Réel (tous les modèles)",
             fontsize=13, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    y_pred = res["y_pred"]
    ax.scatter(y_test, y_pred, alpha=0.4, s=15, color="#3498db", edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()) - 1,
            max(y_test.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Idéal")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Puissance réelle (W)")
    ax.set_ylabel("Puissance prédite (W)")
    ax.set_title(f"{name}\nR²={res['R2']:.4f}  MAE={res['MAE']:.2f}W")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Sauvegardé : {OUTPUT_DIR}/05_predicted_vs_actual.png")

# ── Tableau comparatif ────────────────────────────────────────────────────────
print("\n--- Tableau comparatif des modèles ---")
print(f"{'Modèle':<22} {'MAE (W)':>10} {'RMSE (W)':>10} {'R²':>8} {'CV R²':>12}")
print("-" * 66)
for name, res in results.items():
    star = " ★" if name == best_name else ""
    print(f"{name:<22} {res['MAE']:>10.3f} {res['RMSE']:>10.3f} "
          f"{res['R2']:>8.4f} {res['CV_R2_mean']:>8.4f}±{res['CV_R2_std']:.3f}{star}")

# ── Fonction de prédiction réutilisable ───────────────────────────────────────
def predict_power(voltage, current, temp, humidity, pressure,
                  hour=12, is_daytime=1, voltage_delta=0.0, efficiency=0.95):
    """
    Prédit la puissance générée à partir des données ESP32.

    Exemple d'usage :
        predict_power(voltage=12.5, current=1.2, temp=28, humidity=55,
                      pressure=1012, hour=14, is_daytime=1)
    """
    model = joblib.load(f"{MODELS_DIR}/power_prediction_model.joblib")
    X_new = pd.DataFrame([{
        "voltage": voltage, "current": current, "temp": temp,
        "humidity": humidity, "pressure": pressure, "hour": hour,
        "is_daytime": is_daytime, "voltage_delta": voltage_delta,
        "efficiency": efficiency
    }])
    prediction = model.predict(X_new)[0]
    print(f"  → Puissance prédite : {prediction:.2f} W")
    return prediction

# Test rapide avec des valeurs typiques ESP32
print("\n--- Test de prédiction (données ESP32 simulées) ---")
predict_power(voltage=12.5, current=1.2, temp=28, humidity=55,
              pressure=1012, hour=14, is_daytime=1)
predict_power(voltage=11.0, current=0.5, temp=35, humidity=70,
              pressure=1005, hour=22, is_daytime=0)

print("\n" + "=" * 60)
print("  ✓ Power Prediction terminé — Lancer 03_anomaly_maintenance.py")
print("=" * 60)
