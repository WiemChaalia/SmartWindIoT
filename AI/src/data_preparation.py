"""
SmartWindIoT — Script 01 : Data Preparation & Exploration
==========================================================
Charge le dataset, nettoie les données, explore les distributions
et prépare les features pour les modèles AI.

Run : python 01_data_preparation.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Paths ─────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "data", "smartwind_dataset.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ── 1. Chargement ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  SmartWindIoT — Data Preparation")
print("=" * 60)

df = pd.read_csv(DATASET_PATH, sep=";")

df.columns = df.columns.str.strip()

df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%Y %H:%M")

print(df.head())    # supprimer doublons timestamp

print(f"\n[INFO] Dataset chargé : {len(df)} lignes, {df.shape[1]} colonnes")
print(f"[INFO] Période : {df['timestamp'].min()} → {df['timestamp'].max()}")

# ── 2. Aperçu rapide ──────────────────────────────────────────────────────────
print("\n--- Aperçu des premières lignes ---")
print(df.head())

print("\n--- Types de colonnes ---")
print(df.dtypes)

print("\n--- Valeurs manquantes ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Aucune valeur manquante ✓")

print("\n--- Statistiques descriptives ---")
print(df.describe().round(3))

# ── 3. Distribution des statuts ───────────────────────────────────────────────
print("\n--- Distribution des statuts ---")
counts = df["status"].value_counts()
for label, count in counts.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<10} {count:>5} lignes  ({pct:5.1f}%)  {bar}")

# ── 4. Feature Engineering ────────────────────────────────────────────────────
print("\n[INFO] Création des features temporelles et dérivées...")

df["hour"]         = df["timestamp"].dt.hour
df["day_of_week"]  = df["timestamp"].dt.dayofweek
df["is_daytime"]   = ((df["hour"] >= 6) & (df["hour"] <= 20)).astype(int)

# Puissance calculée vs mesurée (indicateur d'efficacité)
df["power_calc"]   = df["voltage"] * df["current"]
df["efficiency"]   = (df["power"] / df["power_calc"].replace(0, np.nan)).clip(0, 1)

# Features de vibration pour maintenance prédictive
df["vibration_sq"] = df["vibration"] ** 2          # énergie vibratoire
df["vib_rolling_mean"] = df["vibration"].rolling(window=6, min_periods=1).mean()
df["vib_rolling_std"]  = df["vibration"].rolling(window=6, min_periods=1).std().fillna(0)

# Variation de tension (instabilité)
df["voltage_delta"] = df["voltage"].diff().abs().fillna(0)

# Encodage numérique du statut (pour les modèles)
status_map = {"normal": 0, "warning": 1, "critical": 2}
df["status_code"]  = df["status"].map(status_map)

print(f"[INFO] Features créées : {list(df.columns)}")

# ── 5. Visualisations ─────────────────────────────────────────────────────────
print("\n[INFO] Génération des graphiques...")

# 5a. Série temporelle des variables clés
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle("SmartWindIoT — Séries temporelles", fontsize=14, fontweight="bold")

colors = {"normal": "#2ecc71", "warning": "#f39c12", "critical": "#e74c3c"}
color_seq = df["status"].map(colors)

for ax, col, title, color in zip(
    axes,
    ["power", "voltage", "vibration", "temp"],
    ["Puissance (W)", "Tension (V)", "Vibration", "Température (°C)"],
    ["#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
):
    ax.plot(df["timestamp"], df[col], color=color, linewidth=0.7, alpha=0.8)
    ax.set_ylabel(title, fontsize=9)
    ax.grid(alpha=0.3)

    # Surligner les anomalies
    for status, clr in [("warning", "#f39c12"), ("critical", "#e74c3c")]:
        mask = df["status"] == status
        ax.scatter(df.loc[mask, "timestamp"], df.loc[mask, col],
                   color=clr, s=8, zorder=5, alpha=0.7, label=status)

axes[0].legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_time_series.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Sauvegardé : {OUTPUT_DIR}/01_time_series.png")

# 5b. Matrice de corrélation
fig, ax = plt.subplots(figsize=(10, 8))
num_cols = ["temp", "voltage", "current", "vibration", "humidity",
            "pressure", "power", "efficiency", "vib_rolling_mean"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title("Matrice de corrélation — Features SmartWindIoT", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Sauvegardé : {OUTPUT_DIR}/02_correlation_matrix.png")

# 5c. Distribution par statut
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Distribution des features par statut", fontsize=13, fontweight="bold")
features = ["power", "voltage", "current", "vibration", "temp", "efficiency"]
palette  = {"normal": "#2ecc71", "warning": "#f39c12", "critical": "#e74c3c"}

for ax, feat in zip(axes.flat, features):
    for status, clr in palette.items():
        subset = df[df["status"] == status][feat]
        ax.hist(subset, bins=30, alpha=0.6, color=clr, label=status, density=True)
    ax.set_title(feat, fontsize=10)
    ax.set_xlabel("")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Sauvegardé : {OUTPUT_DIR}/03_feature_distributions.png")

# ── 6. Sauvegarde du dataset enrichi ─────────────────────────────────────────
enriched_path = f"{OUTPUT_DIR}/SmartWindIoT_features.csv"
df.to_csv(enriched_path, index=False)
print(f"\n[INFO] Dataset enrichi sauvegardé → {enriched_path}")

print("\n" + "=" * 60)
print("  ✓ Préparation terminée — Lancer 02_power_prediction.py")
print("=" * 60)
