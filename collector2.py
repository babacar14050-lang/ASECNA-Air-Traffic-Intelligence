
# ============================================================
# COLLECTEUR FLIGHTRADAR24 API — PROJET ASECNA
# Collecte live, filtrage géographique et export CSV
# ============================================================

import os
import csv
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

API_TOKEN = os.getenv("FR24_API_TOKEN", "").strip()

BASE_URL = "https://fr24api.flightradar24.com/api"
ENDPOINT = "/live/flight-positions/light"

OUTPUT_FILE = Path("dataset_ASECNA.csv")
LOG_FILE = Path("collect_fr24.log")

# Rectangle de test : Mauritanie et partie de l'Afrique de l'Ouest.
# Ordre obligatoire : Nord, Sud, Ouest, Est.
NORTH = 30.0
SOUTH = -5.0
WEST = -25.0
EAST = 15.0

BOUNDS = f"{NORTH},{SOUTH},{WEST},{EAST}"

# Une requête toutes les 60 secondes par défaut.
# Vérifier les limites de crédits de ton abonnement.
POLL_INTERVAL = int(os.getenv("FR24_INTERVAL", "60"))

# Limite optionnelle du nombre de positions renvoyées.
# None = ne pas envoyer le paramètre limit.
# Un plafond peut réduire le nombre de résultats.
MAX_RESULTS = None

# Règles de conservation FR24 : maximum 30 jours.
RETENTION_DAYS = 30

REQUEST_TIMEOUT = 30
RETRY_DELAY = 30

OUTPUT_COLUMNS = [
    "collection_time_utc",
    "aircraft_time_utc",
    "fr24_id",
    "flight",
    "callsign",
    "hex",
    "registration",
    "aircraft_type",
    "latitude",
    "longitude",
    "altitude_ft",
    "ground_speed_kt",
    "vertical_speed",
    "track_deg",
    "squawk",
    "source",
    "origin_iata",
    "origin_icao",
    "destination_iata",
    "destination_icao",
]


# ============================================================
# 2. JOURNALISATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)

logger = logging.getLogger("fr24_collector")


# ============================================================
# 3. SESSION API
# ============================================================

session = requests.Session()

session.headers.update({
    "Authorization": f"Bearer {API_TOKEN}",
    "Accept": "application/json",
    "Accept-Version": "v1",
})


# ============================================================
# 4. APPEL DE L'API
# ============================================================

def fetch_flights():
    """Récupère les positions en direct dans le rectangle défini."""

    if not API_TOKEN:
        raise RuntimeError(
            "FR24_API_TOKEN absent. Configure ta clé API."
        )

    url = BASE_URL + ENDPOINT

    params = {
        "bounds": BOUNDS,
    }

    if MAX_RESULTS is not None:
        params["limit"] = MAX_RESULTS

    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 401:
        raise RuntimeError(
            "HTTP 401 : token invalide ou non autorisé."
        )

    if response.status_code == 403:
        raise RuntimeError(
            "HTTP 403 : accès refusé pour ce compte ou endpoint."
        )

    if response.status_code == 429:
        raise RuntimeError(
            "HTTP 429 : limite de requêtes atteinte."
        )

    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, dict):
        raise ValueError("Réponse JSON inattendue.")

    flights = payload.get("data", [])

    if not isinstance(flights, list):
        raise ValueError(
            "Le champ 'data' de la réponse API est invalide."
        )

    logger.info(
        "%d positions reçues depuis l'API",
        len(flights),
    )

    return flights


# ============================================================
# 5. NORMALISATION
# ============================================================

def normalize_flights(flights):
    """Transforme les positions API en lignes CSV."""

    collected_at = datetime.now(
        timezone.utc
    ).isoformat()

    rows = []

    for flight in flights:

        lat = flight.get("lat")
        lon = flight.get("lon")

        if lat is None or lon is None:
            continue

        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            continue

        # Contrôle de validité géographique.
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue

        # Filtrage de sécurité local.
        if not (
            SOUTH <= lat <= NORTH
            and WEST <= lon <= EAST
        ):
            continue

        rows.append({
            "collection_time_utc": collected_at,
            "aircraft_time_utc": flight.get("timestamp"),
            "fr24_id": flight.get("fr24_id"),
            "flight": flight.get("flight"),
            "callsign": flight.get("callsign"),
            "hex": flight.get("hex"),
            "registration": flight.get("reg"),
            "aircraft_type": flight.get("type"),
            "latitude": lat,
            "longitude": lon,
            "altitude_ft": flight.get("alt"),
            "ground_speed_kt": flight.get("gspeed"),
            "vertical_speed": flight.get("vspeed"),
            "track_deg": flight.get("track"),
            "squawk": flight.get("squawk"),
            "source": flight.get("source"),
            "origin_iata": flight.get("orig_iata"),
            "origin_icao": flight.get("orig_icao"),
            "destination_iata": flight.get("dest_iata"),
            "destination_icao": flight.get("dest_icao"),
        })

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Éliminer les doublons dans la même réponse.
    if not df.empty:
        df = df.drop_duplicates(
            subset=["fr24_id"],
            keep="last",
        )

    return df


# ============================================================
# 6. SAUVEGARDE
# ============================================================

def save_csv(df):
    """Ajoute les nouvelles positions au fichier historique."""

    if df.empty:
        logger.info("Aucune position exploitable.")
        return

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = (
        not OUTPUT_FILE.exists()
        or OUTPUT_FILE.stat().st_size == 0
    )

    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
    )

    logger.info(
        "%d lignes ajoutées dans %s",
        len(df),
        OUTPUT_FILE,
    )


# ============================================================
# 7. NETTOYAGE DES DONNÉES DE PLUS DE 30 JOURS
# ============================================================

def cleanup_old_data():
    """
    Supprime les données reçues depuis plus de 30 jours.
    À exécuter périodiquement, conformément aux règles FR24.
    """

    if not OUTPUT_FILE.exists():
        return

    try:
        df = pd.read_csv(OUTPUT_FILE)

        if df.empty:
            return

        timestamps = pd.to_datetime(
            df["collection_time_utc"],
            utc=True,
            errors="coerce",
        )

        cutoff = datetime.now(timezone.utc) - timedelta(
            days=RETENTION_DAYS
        )

        retained = df.loc[timestamps >= cutoff].copy()

        removed = len(df) - len(retained)

        # Réécriture du fichier avec les données conservées.
        retained.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8",
        )

        logger.info(
            "Nettoyage terminé : %d lignes supprimées",
            removed,
        )

    except (OSError, ValueError, KeyError) as exc:
        logger.exception("Erreur pendant le nettoyage : %s", exc)


# ============================================================
# 8. CYCLE DE COLLECTE
# ============================================================

def collect_once():
    flights = fetch_flights()
    df = normalize_flights(flights)
    save_csv(df)

    return len(df)


# ============================================================
# 9. BOUCLE PRINCIPALE
# ============================================================

def main():

    if not API_TOKEN:
        raise RuntimeError(
            "Configure FR24_API_TOKEN avant de démarrer."
        )

    if POLL_INTERVAL < 1:
        raise ValueError("FR24_INTERVAL doit être positif.")

    logger.info("Démarrage du collecteur Flightradar24")
    logger.info("Zone géographique : %s", BOUNDS)
    logger.info("Intervalle : %s secondes", POLL_INTERVAL)
    logger.info("CSV : %s", OUTPUT_FILE.resolve())

    next_cleanup = time.monotonic()

    while True:

        cycle_start = time.monotonic()

        try:
            collect_once()

            # Nettoyage périodique toutes les 24 heures.
            if time.monotonic() >= next_cleanup:
                cleanup_old_data()
                next_cleanup = time.monotonic() + 86400

        except KeyboardInterrupt:
            logger.info("Arrêt demandé.")
            break

        except Exception as exc:
            logger.error("Collecte impossible : %s", exc)

            # Ne pas multiplier les appels en cas d'erreur.
            time.sleep(RETRY_DELAY)
            continue

        elapsed = time.monotonic() - cycle_start

        time.sleep(
            max(0, POLL_INTERVAL - elapsed)
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Collecteur arrêté.")
