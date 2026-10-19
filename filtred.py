
import json
import gzip
import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"C:\Users\INDEX INFORMATIQUE\Desktop\Master\donnees_opensky3\states_2018-01-01-00.json.tar"

OUTPUT_FILE = r"C:\Users\INDEX INFORMATIQUE\Desktop\Master\data\afrique_states.csv"


# ============================================================
# ZONE AFRIQUE
# ============================================================

LAT_MIN = -35
LAT_MAX = 37

LON_MIN = -20
LON_MAX = 55


# ============================================================
# VERIFICATION
# ============================================================

if not os.path.isfile(INPUT_FILE):

    raise FileNotFoundError(
        f"Fichier introuvable :\n{INPUT_FILE}"
    )


print("=" * 60)
print("   OPENSKY : JSON/GZIP -> AFRIQUE -> CSV")
print("=" * 60)

print("\nFichier :")
print(INPUT_FILE)


# ============================================================
# DETECTION AUTOMATIQUE DU FORMAT
# ============================================================

with open(INPUT_FILE, "rb") as f:

    signature = f.read(2)


print("\nSignature du fichier :", signature)


# ============================================================
# LECTURE DU FICHIER
# ============================================================

if signature == b"\x1f\x8b":

    print("Format détecté : GZIP")

    with gzip.open(
        INPUT_FILE,
        "rt",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

else:

    print("Format détecté : JSON")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


print("✓ Fichier chargé correctement")


# ============================================================
# VERIFICATION STRUCTURE OPENSKY
# ============================================================

if not isinstance(data, dict):

    raise ValueError(
        "Le fichier JSON n'est pas un dictionnaire."
    )


print("\nClés disponibles :")
print(list(data.keys()))


if "states" not in data:

    raise ValueError(
        "\nLa clé 'states' n'existe pas.\n"
        "Le fichier n'est probablement pas un "
        "State Vector OpenSky classique."
    )


states = data["states"]


print(
    f"\nNombre total d'états : {len(states):,}"
)


# ============================================================
# STRUCTURE OPENSKY
# ============================================================

columns = [

    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source"

]


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    states,
    columns=columns
)


# ============================================================
# COORDONNEES
# ============================================================

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)


# ============================================================
# SUPPRESSION DES COORDONNEES VIDES
# ============================================================

df = df.dropna(
    subset=[
        "latitude",
        "longitude"
    ]
)


print(
    f"Positions valides : {len(df):,}"
)


# ============================================================
# FILTRAGE AFRIQUE
# ============================================================

print("\nFiltrage Afrique...")


mask = (

    (df["latitude"] >= LAT_MIN)
    &
    (df["latitude"] <= LAT_MAX)

    &

    (df["longitude"] >= LON_MIN)
    &
    (df["longitude"] <= LON_MAX)

)


df_afrique = df.loc[mask].copy()


print(
    f"Positions Afrique : {len(df_afrique):,}"
)


# ============================================================
# CONVERSION DES TEMPS
# ============================================================

for col in [
    "time_position",
    "last_contact"
]:

    if col in df_afrique.columns:

        df_afrique[col] = pd.to_datetime(
            df_afrique[col],
            unit="s",
            errors="coerce",
            utc=True
        )


# ============================================================
# TEMPS DU SNAPSHOT
# ============================================================

if "time" in data:

    try:

        snapshot_time = pd.to_datetime(
            data["time"],
            unit="s",
            utc=True
        )

        df_afrique["snapshot_time"] = snapshot_time

    except Exception:

        pass


# ============================================================
# TRI
# ============================================================

if "time_position" in df_afrique.columns:

    df_afrique = df_afrique.sort_values(
        "time_position"
    )


# ============================================================
# CREATION DOSSIER
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


# ============================================================
# EXPORT CSV
# ============================================================

df_afrique.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# RESULTATS
# ============================================================

print("\n" + "=" * 60)
print("                 TERMINE")
print("=" * 60)

print(
    f"États originaux : {len(states):,}"
)

print(
    f"Positions valides : {len(df):,}"
)

print(
    f"Positions Afrique : {len(df_afrique):,}"
)

if len(states) > 0:

    ratio = (
        len(df_afrique)
        / len(states)
    ) * 100

    print(
        f"Pourcentage Afrique : {ratio:.2f}%"
    )


print("\nFichier CSV :")
print(OUTPUT_FILE)

print("\n✓ Export terminé.")
