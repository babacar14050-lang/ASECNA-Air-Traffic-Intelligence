# ==========================================================
# COLLECTOR PRO OPENSKY (ANTI-BAN / STABLE / PRODUCTION)
# ==========================================================
# Features :
# ✅ Authentification OpenSky
# ✅ Respect rate limit
# ✅ Retry automatique
# ✅ Backoff exponentiel si erreur 429
# ✅ Timeout réseau
# ✅ Rotation fréquence requêtes
# ✅ Filtrage zone géographique
# ✅ Sauvegarde CSV incrémentale
# ✅ Logs propres
# ✅ Reprise automatique
# ==========================================================

import requests
import pandas as pd
import time
import random
import os
from datetime import datetime

# ==========================================================
# CONFIGURATION
# ==========================================================

USERNAME = "babacar14050@gmail.com"
PASSWORD = "babacar1122@@"

BASE_URL = "https://opensky-network.org/api/states/all"

# Zone ASECNA + région proche
PARAMS = {
    "lamin": -28,
    "lamax": 28,
    "lomin": -18,
    "lomax": 54
}




OUTPUT_FILE = "data/data.csv"

# fréquence normale (secondes)
MIN_SLEEP = 15
MAX_SLEEP = 30

# retry config
MAX_RETRIES = 5
TIMEOUT = 20

# ==========================================================
# COLONNES OPENSKY
# ==========================================================

COLUMNS = [
    "icao24", "callsign", "origin_country", "time_position",
    "last_contact", "longitude", "latitude", "baro_altitude",
    "on_ground", "velocity", "heading", "vertical_rate",
    "sensors", "geo_altitude", "squawk", "spi", "position_source"
]

# ==========================================================
# LOGGING SIMPLE
# ==========================================================

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ==========================================================
# SAUVEGARDE CSV
# ==========================================================

def save_csv(df):
    file_exists = os.path.isfile(OUTPUT_FILE)

    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )

# ==========================================================
# REQUÊTE SECURISEE
# ==========================================================

def fetch_opensky():

    retries = 0

    while retries < MAX_RETRIES:

        try:
            response = requests.get(
                BASE_URL,
                params=PARAMS,
                auth=(USERNAME, PASSWORD),
                timeout=TIMEOUT
            )

            # ========================
            # SUCCESS
            # ========================
            if response.status_code == 200:
                return response.json()

            # ========================
            # TOO MANY REQUESTS
            # ========================
            elif response.status_code == 429:
                wait_time = (2 ** retries) * 30
                log(f"429 Too Many Requests → pause {wait_time}s")
                time.sleep(wait_time)
                retries += 1

            # ========================
            # SERVER ERRORS
            # ========================
            elif response.status_code >= 500:
                wait_time = (2 ** retries) * 10
                log(f"Erreur serveur {response.status_code} → retry {wait_time}s")
                time.sleep(wait_time)
                retries += 1

            else:
                log(f"Erreur HTTP {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            log("Timeout réseau")
            retries += 1
            time.sleep(10)

        except requests.exceptions.ConnectionError:
            log("Connexion perdue")
            retries += 1
            time.sleep(15)

        except Exception as e:
            log(f"Erreur inconnue : {e}")
            retries += 1
            time.sleep(10)

    log("Max retries atteint")
    return None

# ==========================================================
# TRAITEMENT DATA
# ==========================================================

def process_data(data):

    if not data or not data.get("states"):
        return None

    df = pd.DataFrame(data["states"], columns=COLUMNS)

    # garder seulement lignes valides
    df = df.dropna(subset=["latitude", "longitude"])

    # filtrage qualité
    df = df[df["velocity"].fillna(0) > 30]

    # timestamp collecte
    df["collected_at"] = datetime.now()

    return df

# ==========================================================
# BOUCLE PRINCIPALE
# ==========================================================

def run_collector():

    log("Collector OpenSky PRO démarré")

    while True:

        data = fetch_opensky()

        if data:
            df = process_data(data)

            if df is not None and len(df) > 0:
                save_csv(df)
                log(f"{len(df)} lignes sauvegardées")
            else:
                log("Aucune donnée exploitable")

        # fréquence variable (anti-pattern bot)
        sleep_time = random.randint(MIN_SLEEP, MAX_SLEEP)
        log(f"Pause {sleep_time}s")
        time.sleep(sleep_time)

# ==========================================================
# START
# ==========================================================

if __name__ == "__main__":
    run_collector()