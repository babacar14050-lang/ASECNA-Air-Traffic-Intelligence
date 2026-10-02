# ============================================================
# IMPORTATION DES BIBLIOTHEQUE
# ============================================================

import os
import tarfile
import gzip
import json
import warnings
import joblib

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DOSSIER = r"C:\Users\INDEX INFORMATIQUE\Desktop\Master\data\donnees_opensky"
CSV_SORTIE = r"C:\Users\INDEX INFORMATIQUE\Desktop\Master\data\dataset_ASECNA_2017_2022.csv"

DOSSIER_MODELES = r"C:\Users\INDEX INFORMATIQUE\Desktop\Master\models\XGBoost"
MODELE_SORTIE = os.path.join(DOSSIER_MODELES, "xgboost_congestion_2h.pkl")
FEATURES_SORTIE = os.path.join(DOSSIER_MODELES, "features_xgboost_2h.pkl")

INTERVALLE = "15min"
GRID = 0.5
HORIZON_2H = 8                 # 8 x 15 min = 2 heures
TRAIN_RATIO = 0.80

os.makedirs(os.path.dirname(CSV_SORTIE), exist_ok=True)
os.makedirs(DOSSIER_MODELES, exist_ok=True)

# ============================================================
# ZONE ASECNA APPROXIMATIVE
# ============================================================

ASECNA = {
    "lat_min": -90,
    "lat_max": 90.0,
    "lon_min": -180.0,
    "lon_max": 180.0
}


# ============================================================
# FILTRE ASECNA
# ============================================================

def filtrer_asecna(df):
    return df[
        df["lat"].between(ASECNA["lat_min"], ASECNA["lat_max"])
        & df["lon"].between(ASECNA["lon_min"], ASECNA["lon_max"])
    ].copy()


# ============================================================
# TRAITER UN FICHIER TAR
# ============================================================

def traiter_fichier(chemin_tar):

    print("\n" + "=" * 70)
    print("TRAITEMENT :", os.path.basename(chemin_tar))
    print("=" * 70)

    try:
        data = None

        with tarfile.open(chemin_tar, "r:*") as tar:
            for membre in tar.getmembers():

                if not membre.isfile():
                    continue

                if not (membre.name.endswith(".json.gz") or
                        membre.name.endswith(".json")):
                    continue

                fichier = tar.extractfile(membre)

                if fichier is None:
                    continue

                print("Fichier interne :", membre.name)

                if membre.name.endswith(".json.gz"):
                    with gzip.GzipFile(fileobj=fichier) as gz:
                        data = json.load(gz)
                else:
                    data = json.load(fichier)

                break

        if data is None:
            print("Aucun fichier JSON trouvé.")
            return None

        if isinstance(data, list):
            states = data
        elif isinstance(data, dict):
            states = data.get("states", [])
        else:
            states = []

        if not states:
            print("Aucune position.")
            return None

        df = pd.DataFrame(states)

        colonnes = ["time", "icao24", "lat", "lon", "velocity"]

        for c in colonnes:
            if c not in df.columns:
                print("Colonne manquante :", c)
                return None

        df = df[colonnes].copy()

        # Conversion numérique
        for c in ["time", "lat", "lon", "velocity"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Nettoyage
        df = df.dropna(subset=["time", "lat", "lon"])
        df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]

        # Zone ASECNA
        df = filtrer_asecna(df)

        if df.empty:
            print("Aucune position dans la zone ASECNA.")
            return None

        # Date UTC
        df["datetime"] = pd.to_datetime(
            df["time"], unit="s", errors="coerce", utc=True
        )
        df = df.dropna(subset=["datetime"])

        # Grille
        df["latitude"] = np.floor(df["lat"] / GRID) * GRID
        df["longitude"] = np.floor(df["lon"] / GRID) * GRID

        # Intervalle de 15 min
        df["datetime"] = df["datetime"].dt.floor(INTERVALLE)

        # Agrégation
        resultat = (
            df.groupby(["datetime", "latitude", "longitude"], observed=True)
            .agg(
                traffic_count=("icao24", "nunique"),
                adsb_observations=("icao24", "count"),
                avg_speed=("velocity", "mean"),
                speed_var=("velocity", "std"),
            )
            .reset_index()
        )

        resultat["speed_var"] = resultat["speed_var"].fillna(0)
        resultat["avg_speed"] = resultat["avg_speed"].fillna(0)

        # Densité de trafic
        resultat["density"] = resultat["traffic_count"]

        print("Positions ASECNA :", len(df))
        print("Cellules/instants :", len(resultat))

        return resultat

    except Exception as e:
        print("ERREUR :", repr(e))
        return None


# ============================================================
# 1. COLLECTE DES FICHIERS
# ============================================================

fichiers = sorted(
    f for f in os.listdir(DOSSIER)
    if f.endswith(".tar")
)

if not fichiers:
    raise FileNotFoundError(
        f"Aucun fichier .tar trouvé dans : {DOSSIER}"
    )

print("\nNombre de fichiers TAR :", len(fichiers))

resultats = []

for i, fichier in enumerate(fichiers, 1):

    print(f"\nFICHIER {i}/{len(fichiers)}")

    chemin = os.path.join(DOSSIER, fichier)
    resultat = traiter_fichier(chemin)

    if resultat is not None and not resultat.empty:
        resultats.append(resultat)


if not resultats:
    raise RuntimeError("Aucune donnée ASECNA n'a été produite.")


# ============================================================
# 2. CONCATÉNATION
# ============================================================

df = pd.concat(resultats, ignore_index=True)

del resultats

print("\n" + "=" * 70)
print("CONCATÉNATION TERMINÉE")
print("Lignes :", len(df))
print("=" * 70)


# ============================================================
# 3. AGRÉGATION DE SÉCURITÉ
# Évite les doublons si plusieurs fichiers contiennent le même
# intervalle/cellule.
# ============================================================

df = (
    df.groupby(["datetime", "latitude", "longitude"], as_index=False)
    .agg(
        traffic_count=("traffic_count", "sum"),
        adsb_observations=("adsb_observations", "sum"),
        avg_speed=("avg_speed", "mean"),
        speed_var=("speed_var", "mean"),
    )
)

df["density"] = df["traffic_count"]


# ============================================================
# 4. CONGESTION INDEX GLOBAL
#
# IMPORTANT :
# On calcule les maxima APRÈS avoir concaténé toutes les années.
# Dans l'ancien code, la normalisation était faite fichier par
# fichier, ce qui rendait l'indice incohérent dans le temps.
# ============================================================

for c in ["traffic_count", "density", "speed_var"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

max_traffic = max(df["traffic_count"].quantile(0.99), 1)
max_density = max(df["density"].quantile(0.99), 1)
max_speed_var = max(df["speed_var"].quantile(0.99), 1)

traffic_norm = (df["traffic_count"] / max_traffic).clip(0, 1)
density_norm = (df["density"] / max_density).clip(0, 1)
speed_norm = (df["speed_var"] / max_speed_var).clip(0, 1)

df["congestion_index"] = (
    0.50 * traffic_norm
    + 0.30 * density_norm
    + 0.20 * speed_norm
)


# ============================================================
# 5. TRI CHRONOLOGIQUE PAR CELLULE
# ============================================================

df = df.sort_values(
    ["latitude", "longitude", "datetime"]
).reset_index(drop=True)


# ============================================================
# 6. VARIABLES TEMPORELLES
# ============================================================

df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["dayofyear"] = df["datetime"].dt.dayofyear

# Encodage cyclique : beaucoup plus adapté que l'heure brute
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)


# ============================================================
# 7. LAGS PAR CELLULE
#
# 1  = -15 min
# 2  = -30 min
# 4  = -1 h
# 8  = -2 h
# 12 = -3 h
# 24 = -6 h
# 48 = -12 h
# 96 = -24 h
# ============================================================

GROUPE = ["latitude", "longitude"]

variables_lag = [
    "traffic_count",
    "adsb_observations",
    "density",
    "avg_speed",
    "speed_var",
    "congestion_index",
]

lags = [1, 2, 4, 8, 12, 24, 48, 96]

g = df.groupby(GROUPE, sort=False)

for col in variables_lag:
    for lag in lags:
        df[f"{col}_lag_{lag}"] = g[col].shift(lag)


# ============================================================
# 8. VARIATIONS TEMPORELLES
# ============================================================

df["traffic_change_15m"] = (
    df["traffic_count"]
    - df["traffic_count_lag_1"]
)

df["traffic_change_1h"] = (
    df["traffic_count"]
    - df["traffic_count_lag_4"]
)

df["traffic_change_2h"] = (
    df["traffic_count"]
    - df["traffic_count_lag_8"]
)

df["congestion_change_15m"] = (
    df["congestion_index"]
    - df["congestion_index_lag_1"]
)

df["congestion_change_1h"] = (
    df["congestion_index"]
    - df["congestion_index_lag_4"]
)

df["congestion_change_2h"] = (
    df["congestion_index"]
    - df["congestion_index_lag_8"]
)


# ============================================================
# 9. MOYENNES MOBILES PAR CELLULE
# Exclut la valeur actuelle avec shift(1) :
# aucune fuite d'information.
# ============================================================

for col in [
    "traffic_count",
    "density",
    "avg_speed",
    "speed_var",
    "congestion_index"
]:

    shifted = g[col].shift(1)

    for window in [4, 8, 12, 24, 48]:

        df[f"{col}_roll_mean_{window}"] = (
            shifted
            .groupby(
                [df["latitude"], df["longitude"]],
                sort=False
            )
            .transform(
                lambda x: x.rolling(window, min_periods=2).mean()
            )
        )

        df[f"{col}_roll_std_{window}"] = (
            shifted
            .groupby(
                [df["latitude"], df["longitude"]],
                sort=False
            )
            .transform(
                lambda x: x.rolling(window, min_periods=2).std()
            )
        )


# ============================================================
# 10. CIBLE +2 HEURES
#
# CORRECTION MAJEURE :
# L'ancien code utilisait shift(-8) sur tout le dataframe.
# Ici la cible est calculée INDÉPENDAMMENT POUR CHAQUE CELLULE.
# ============================================================

df["congestion_future_2h"] = (
    g["congestion_index"].shift(-HORIZON_2H)
)


# ============================================================
# 11. SUPPRESSION DES LIGNES INEXPLOITABLES
# ============================================================

df = df.replace([np.inf, -np.inf], np.nan)

# Pour les lags/rolling, on garde uniquement les lignes complètes.
# Cela évite que XGBoost apprenne sur des périodes sans historique.
df_model = df.dropna(
    subset=["congestion_future_2h"]
).copy()


# ============================================================
# 12. NIVEAU DE CONGESTION
# ============================================================

df_model["congestion_level"] = pd.cut(
    df_model["congestion_index"],
    bins=[-0.01, 0.30, 0.60, 0.80, 1.01],
    labels=["Faible", "Moyenne", "Forte", "Très forte"]
)


# ============================================================
# 13. SAUVEGARDE DU DATASET FINAL
# ============================================================

colonnes_export = [
    "datetime",
    "latitude",
    "longitude",
    "traffic_count",
    "adsb_observations",
    "density",
    "avg_speed",
    "speed_var",
    "congestion_index",
    "congestion_future_2h",
    "congestion_level",
]

# Ajouter toutes les nouvelles features
features_temporaires = [
    c for c in df_model.columns
    if c not in colonnes_export
]

df_export = df_model[
    colonnes_export + features_temporaires
].copy()

df_export.to_csv(
    CSV_SORTIE,
    index=False,
    encoding="utf-8"
)

print("\nDataset sauvegardé :", CSV_SORTIE)
print("Nombre total de lignes :", len(df_export))


# ============================================================
# 14. CONSTRUCTION X / y
# ============================================================

# On retire la cible et les colonnes non numériques.
# IMPORTANT : latitude/longitude restent comme variables spatiales.
TARGET = "congestion_future_2h"

EXCLUDE = [
    TARGET,
    "datetime",
    "congestion_level",
]

feature_cols = [
    c for c in df_model.columns
    if c not in EXCLUDE
]

X = df_model[feature_cols].copy()
y = df_model[TARGET].copy()

# Toutes les variables doivent être numériques
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# Remplacement sécurisé
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

y = pd.to_numeric(y, errors="coerce")

mask = y.notna()

X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)
df_model = df_model.loc[mask].reset_index(drop=True)


# ============================================================
# 15. SÉPARATION CHRONOLOGIQUE
#
# PAS de train_test_split aléatoire.
# Le futur ne doit jamais être utilisé pour entraîner le modèle.
# ============================================================

dates = df_model["datetime"]

date_limite = dates.quantile(TRAIN_RATIO)

train_mask = dates <= date_limite
test_mask = dates > date_limite

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print("\n" + "=" * 70)
print("SÉPARATION CHRONOLOGIQUE")
print("=" * 70)

print("Date limite :", date_limite)
print("Train :", len(X_train))
print("Test  :", len(X_test))

# ============================================================
# MODÈLE RandomForestRegressor   !!! a cmmenter
# ============================================================
model_rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2 = r2_score(y_test, y_pred_rf)

print("\n" + "=" * 70)
print("RÉSULTATS RandomForestRegressor +2H")
print("=" * 70)

print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")


# ============================================================
# 16. MODÈLE XGBOOST AMÉLIORÉ
# ============================================================

model = XGBRegressor(
    objective="reg:squarederror",

    n_estimators=1500,

    learning_rate=0.025,

    max_depth=7,

    min_child_weight=3,

    subsample=0.85,

    colsample_bytree=0.85,

    gamma=0.05,

    reg_alpha=0.02,

    reg_lambda=1.5,

    random_state=42,

    n_jobs=-1,

    tree_method="hist",

    eval_metric="rmse",

    early_stopping_rounds=80
)
# ============================================================
# 17. ENTRAÎNEMENT
# ============================================================

print("\n" + "=" * 70)
print("ENTRAÎNEMENT XGBOOST")
print("=" * 70)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)


# ============================================================
# 18. ÉVALUATION
# ============================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 70)
print("RÉSULTATS XGBOOST +2H")
print("=" * 70)

print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")


# ============================================================
# 19. IMPORTANCE DES FEATURES
# ============================================================

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print("\nTOP 30 FEATURES")
print(importance.head(30).to_string(index=False))


# ============================================================
# 20. SAUVEGARDE MODÈLE
# ============================================================

joblib.dump(model, MODELE_SORTIE)
joblib.dump(feature_cols, FEATURES_SORTIE)

print("\nModèle :", MODELE_SORTIE)
print("Features :", FEATURES_SORTIE)

print("\n" + "=" * 70)
print("TERMINÉ")
print("=" * 70)
