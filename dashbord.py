import os
import warnings
import joblib

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. CONFIGURATION
# ============================================================


st.set_page_config(
    page_title="ASECNA Air Traffic Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. STYLE DU DASHBOARD
# ============================================================

st.markdown(
    """
    <style>
    /* =========================================================
   SUPPRESSION DE LA BARRE SUPÉRIEURE STREAMLIT
   ========================================================= */

header[data-testid="stHeader"] {
    display: none !important;
}


/* =========================================================
   SUPPRESSION DU MENU / DEPLOY
   ========================================================= */

[data-testid="stToolbar"] {
    display: none !important;
}


/* =========================================================
   SUPPRESSION DE L'ESPACE BLANC SUPÉRIEUR
   ========================================================= */

.block-container {
    padding-top: 0rem !important;
    padding-bottom: 1rem !important;
}


    /* ------------------------------------------------------ */
    /* GLOBAL */
    /* ------------------------------------------------------ */

    .stApp {
        background:
            linear-gradient(
                180deg,
                #050b14 0%,
                #07111f 50%,
                #081421 100%
            );
        color: #e5edf5;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1700px;
    }


    /* ------------------------------------------------------ */
    /* TITRE */
    /* ------------------------------------------------------ */

    .main-title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #eaf7ff;
        margin-bottom: 2px;
    }

    .main-subtitle {
        font-size: 13px;
        color: #71869b;
        margin-bottom: 18px;
    }


    /* ------------------------------------------------------ */
    /* HEADER ATC */
    /* ------------------------------------------------------ */

    .atc-header {
        background:
            linear-gradient(
                90deg,
                rgba(7, 22, 37, 0.95),
                rgba(5, 16, 28, 0.85)
            );

        border: 1px solid #17354a;
        border-radius: 10px;

        padding: 12px 18px;
        margin-bottom: 12px;

        font-family: monospace;
    }

    .atc-header-title {
        color: #00d9ff;
        font-size: 16px;
        font-weight: bold;
    }

    .atc-header-info {
        color: #7f9bb0;
        font-size: 12px;
    }

 /* ====================================================== */
 /* KPI — LISIBILITÉ */
 /* ====================================================== */

div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #12283b,
        #0d2031
    );

    border: 1px solid #294960;
    border-radius: 10px;

    padding: 14px 16px;

    box-shadow:
        0 5px 20px rgba(0, 0, 0, 0.25);
}


/* ------------------------------------------------------ */
/* TITRE : AVIONS / CONGESTION / DENSITÉ / VITESSE */
/* ------------------------------------------------------ */

div[data-testid="stMetric"] label {
    color: #ffffff !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}


/* Texte contenu dans le label */
div[data-testid="stMetric"] label p {
    color: #ffffff !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}


/* ------------------------------------------------------ */
/* VALEUR : 125 / 0.45 / etc. */
/* ------------------------------------------------------ */

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}


/* ------------------------------------------------------ */
/* DELTA : +5.2% / -2.1% */
/* ------------------------------------------------------ */

div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #e4f1f8 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}


/* Tous les textes internes du KPI */
div[data-testid="stMetric"] span {
    opacity: 1 !important;
}
/* ======================================================
   SIDEBAR
   ====================================================== */


    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #07111d,
                #050c15
            );

        border-right: 1px solid #173047;
    }

/* Checkbox */
section[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    color: #f5f9fc !important;
}

/* Titres */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #ffffff !important;
    font-weight: 700;
}

/* Texte secondaire */
section[data-testid="stSidebar"] p {
    color: #d9e7f0 !important;
}

/* Champs de texte */
section[data-testid="stSidebar"] input {
    color: #ffffff !important;
}

# /* Placeholder */
# section[data-testid="stSidebar"] input::placeholder {
#     color: #b8cbd8 !important;
# }
   
    /* ------------------------------------------------------ */
    /* SECTION */
    /* ------------------------------------------------------ */

    .section-title {

        color: #dff7ff;

        font-size: 20px;

        font-weight: 750;

        margin-top: 18px;

        margin-bottom: 8px;
    }


    /* ------------------------------------------------------ */
    /* STATUS */
    /* ------------------------------------------------------ */

    .status-box {

        background: #071822;

        border: 1px solid #174357;

        border-radius: 8px;

        padding: 10px 14px;

        margin-bottom: 10px;

        font-family: monospace;

        font-size: 12px;
    }


    /* ------------------------------------------------------ */
    /* ATC FOOTER */
    /* ------------------------------------------------------ */

    .footer {

        color: #587086;

        font-size: 11px;

        text-align: center;

        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. CHEMINS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

AIRCRAFT_FILE = os.path.join(
    DATA_DIR,
    "data.csv"
)


# ============================================================
# 4. MODÈLES
# ============================================================

MODEL_CANDIDATES = [

    "xgboost_congestion_2h.pkl",
]


FEATURE_CANDIDATES = [

    "features_xgboost_2h.pkl",

 
]


def find_model():

    for filename in MODEL_CANDIDATES:

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        if os.path.exists(path):

            return path

    return None


def find_features():

    for filename in FEATURE_CANDIDATES:

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        if os.path.exists(path):

            return path

    return None


MODEL_PATH = find_model()

FEATURES_PATH = find_features()


# ============================================================
# 5. ZONES ASECNA
# ============================================================

ASECNA_ZONES = {

    "Bénin": {
        "lat_min": 6.0,
        "lat_max": 12.5,
        "lon_min": 0.5,
        "lon_max": 3.9
    },

    "Burkina Faso": {
        "lat_min": 9.3,
        "lat_max": 15.1,
        "lon_min": -5.5,
        "lon_max": 2.5
    },

    "Cameroun": {
        "lat_min": 1.5,
        "lat_max": 13.1,
        "lon_min": 8.4,
        "lon_max": 16.3
    },

    "Centrafrique": {
        "lat_min": 2.2,
        "lat_max": 11.0,
        "lon_min": 14.0,
        "lon_max": 27.5
    },

    "Comores": {
        "lat_min": -13.2,
        "lat_max": -11.0,
        "lon_min": 43.0,
        "lon_max": 44.7
    },

    "Congo": {
        "lat_min": -5.0,
        "lat_max": 3.8,
        "lon_min": 11.0,
        "lon_max": 18.7
    },

    "Côte d'Ivoire": {
        "lat_min": 4.2,
        "lat_max": 10.8,
        "lon_min": -8.7,
        "lon_max": -2.4
    },

    "France": {
        "lat_min": 41.0,
        "lat_max": 51.5,
        "lon_min": -5.5,
        "lon_max": 9.5
    },

    "Gabon": {
        "lat_min": -4.0,
        "lat_max": 2.5,
        "lon_min": 8.5,
        "lon_max": 14.5
    },

    "Guinée-Bissau": {
        "lat_min": 10.7,
        "lat_max": 12.7,
        "lon_min": -17.0,
        "lon_max": -13.5
    },

    "Guinée équatoriale": {
        "lat_min": 0.5,
        "lat_max": 3.9,
        "lon_min": 5.5,
        "lon_max": 11.5
    },

    "Madagascar": {
        "lat_min": -26.0,
        "lat_max": -11.5,
        "lon_min": 43.0,
        "lon_max": 51.0
    },

    "Mali": {
        "lat_min": 10.0,
        "lat_max": 25.0,
        "lon_min": -12.5,
        "lon_max": 4.5
    },

    "Mauritanie": {
        "lat_min": 14.5,
        "lat_max": 27.5,
        "lon_min": -17.5,
        "lon_max": -4.0
    },

    "Niger": {
        "lat_min": 11.5,
        "lat_max": 24.0,
        "lon_min": 0.0,
        "lon_max": 16.5
    },

    "Rwanda": {
        "lat_min": -2.9,
        "lat_max": -1.0,
        "lon_min": 28.8,
        "lon_max": 30.9
    },


    "Sénégal": {
        "lat_min": 12.2,
        "lat_max": 16.8,
        "lon_min": -17.7,
        "lon_max": -11.2
    },

    "Tchad": {
        "lat_min": 7.0,
        "lat_max": 23.5,
        "lon_min": 14.0,
        "lon_max": 24.0
    },

    "Togo": {
        "lat_min": 6.0,
        "lat_max": 11.2,
        "lon_min": -0.2,
        "lon_max": 1.8
    }
}


# ============================================================
# 6. DÉTECTION PAYS
# ============================================================

def detect_country(
    latitude,
    longitude
):

    for country, zone in ASECNA_ZONES.items():

        if (

            zone["lat_min"]
            <= latitude
            <= zone["lat_max"]

            and

            zone["lon_min"]
            <= longitude
            <= zone["lon_max"]

        ):

            return country

    return "Hors zone ASECNA"


# ============================================================
# 7. NIVEAU CONGESTION
# ============================================================

def congestion_level(
    value
):

    if value < 0.30:
        return "FAIBLE"

    if value < 0.60:
        return "MOYENNE"

    if value < 0.80:
        return "FORTE"

    return "TRÈS FORTE"


# ============================================================
# 8. CHARGEMENT ADS-B
# ============================================================

@st.cache_data(
    ttl=30,
    show_spinner=False
)
def load_adsb():

    if not os.path.exists(
        AIRCRAFT_FILE
    ):

        return (
            pd.DataFrame(),
            f"Fichier introuvable : {AIRCRAFT_FILE}"
        )


    try:

        df = pd.read_csv(
            AIRCRAFT_FILE
        )


        # ----------------------------------------------------
        # Nettoyage noms colonnes
        # ----------------------------------------------------

        df.columns = (
            df.columns
            .str.strip()
        )


        # ----------------------------------------------------
        # Colonnes indispensables
        # ----------------------------------------------------

        required = [

            "icao24",

            "latitude",

            "longitude"
        ]


        missing = [

            col

            for col in required

            if col not in df.columns
        ]


        if missing:

            return (
                pd.DataFrame(),
                "Colonnes manquantes : "
                + ", ".join(missing)
            )


        # ----------------------------------------------------
        # Colonnes numériques
        # ----------------------------------------------------

        numeric_cols = [

            "latitude",

            "longitude",

            "baro_altitude",

            "geo_altitude",

            "velocity",

            "heading",

            "vertical_rate"
        ]


        for col in numeric_cols:

            if col in df.columns:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )


        # ----------------------------------------------------
        # ICAO
        # ----------------------------------------------------

        df["icao24"] = (

            df["icao24"]

            .astype(str)

            .str.strip()

            .str.lower()
        )


        # ----------------------------------------------------
        # CALLSIGN
        # ----------------------------------------------------

        if "callsign" in df.columns:

            df["callsign"] = (

                df["callsign"]

                .fillna("")

                .astype(str)

                .str.strip()
            )

        else:

            df["callsign"] = ""


        # ----------------------------------------------------
        # ORIGIN COUNTRY
        # ----------------------------------------------------

        if "origin_country" not in df.columns:

            df["origin_country"] = ""


        # ----------------------------------------------------
        # ON GROUND
        # ----------------------------------------------------

        if "on_ground" not in df.columns:

            df["on_ground"] = False

        else:

            if df["on_ground"].dtype == object:

                df["on_ground"] = (

                    df["on_ground"]

                    .astype(str)

                    .str.lower()

                    .isin(
                        [
                            "true",
                            "1",
                            "yes"
                        ]
                    )
                )

            else:

                df["on_ground"] = (
                    df["on_ground"]
                    .fillna(False)
                    .astype(bool)
                )


        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        if "collected_at" in df.columns:

            df["collected_at"] = (

                pd.to_datetime(

                    df["collected_at"],

                    errors="coerce",

                    utc=True
                )
            )

        else:

            df["collected_at"] = pd.NaT


        # ----------------------------------------------------
        # POSITIONS VALIDES
        # ----------------------------------------------------

        df = df.dropna(

            subset=[
                "latitude",
                "longitude"
            ]
        ).copy()


        df = df[
            df["latitude"].between(
                -90,
                90
            )
            &
            df["longitude"].between(
                -180,
                180
            )
        ].copy()


        # ----------------------------------------------------
        # VITESSE
        # velocity ADS-B = m/s
        # ----------------------------------------------------

        if "velocity" in df.columns:

            df["speed_kmh"] = (

                df["velocity"]
                .fillna(0)
                * 3.6
            )

        else:

            df["speed_kmh"] = 0


        # ----------------------------------------------------
        # ALTITUDE
        # ----------------------------------------------------

        if "baro_altitude" in df.columns:

            df["altitude_ft"] = (

                df["baro_altitude"]
                .fillna(0)
                * 3.28084
            )

        elif "geo_altitude" in df.columns:

            df["altitude_ft"] = (

                df["geo_altitude"]
                .fillna(0)
                * 3.28084
            )

        else:

            df["altitude_ft"] = 0


        # ----------------------------------------------------
        # CAP
        # ----------------------------------------------------

        if "heading" in df.columns:

            df["heading"] = (

                df["heading"]
                .fillna(0)
                % 360
            )

        else:

            df["heading"] = 0


        # ----------------------------------------------------
        # PAYS
        # ----------------------------------------------------

        df["country"] = df.apply(

            lambda row:

            detect_country(

                row["latitude"],

                row["longitude"]
            ),

            axis=1
        )


        # ----------------------------------------------------
        # TRI CHRONOLOGIQUE
        # ----------------------------------------------------

        if "collected_at" in df.columns:

            df = df.sort_values(
                "collected_at"
            )


        return df, None


    except Exception as e:

        return (

            pd.DataFrame(),

            f"{type(e).__name__} : {e}"
        )


# ============================================================
# 9. CHARGEMENT
# ============================================================

adsb, load_error = load_adsb()


# ============================================================
# 10. DERNIÈRE POSITION DE CHAQUE AVION
# ============================================================

def get_latest_positions(
    df
):

    if df.empty:

        return df


    work = df.copy()


    if "collected_at" in work.columns:

        work = work.sort_values(
            "collected_at"
        )


    latest = (

        work

        .drop_duplicates(
            subset=["icao24"],
            keep="last"
        )

        .copy()
    )


    return latest


latest_adsb = get_latest_positions(
    adsb
)


# ============================================================
# 11. CALCUL CONGESTION
# ============================================================

def calculate_congestion(
    df,
    area
):

    if df.empty:

        return {

            "traffic_count": 0,

            "observations": 0,

            "density": 0.0,

            "avg_speed": 0.0,

            "speed_var": 0.0,

            "congestion": 0.0
        }


    traffic_count = int(

        df["icao24"]
        .nunique()
    )


    observations = int(
        len(df)
    )


    avg_speed = float(

        df["speed_kmh"]
        .mean()
    )


    speed_var = float(

        df["speed_kmh"]
        .std()
    )


    if np.isnan(speed_var):

        speed_var = 0.0


    density = (

        traffic_count
        /
        max(area, 1)
    )


    # --------------------------------------------------------
    # Normalisation
    # --------------------------------------------------------

    traffic_score = min(

        traffic_count / 100,

        1.0
    )


    density_score = min(

        density / 2,

        1.0
    )


    speed_score = min(

        speed_var / 150,

        1.0
    )


    # --------------------------------------------------------
    # INDICE
    # --------------------------------------------------------

    congestion = (

        0.50 * traffic_score

        +

        0.30 * density_score

        +

        0.20 * speed_score
    )


    congestion = float(

        np.clip(
            congestion,
            0,
            1
        )
    )


    return {

        "traffic_count":
            traffic_count,

        "observations":
            observations,

        "density":
            density,

        "avg_speed":
            avg_speed,

        "speed_var":
            speed_var,

        "congestion":
            congestion
    }


# ============================================================
# 12. SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## ✈️ ASECNA  ATC"
)

st.sidebar.caption(
    "AIR TRAFFIC SURVEILLANCE"
)


selected_country = st.sidebar.selectbox(

    "🌍 SECTEUR / PAYS",

    ["Tous les pays"]
    +
    list(
        ASECNA_ZONES.keys()
    )
)


st.sidebar.markdown("---")


show_aircraft = st.sidebar.checkbox(

    "✈️ Afficher les avions",

    value=True
)


show_labels = st.sidebar.checkbox(

    "🏷️ Afficher les labels",

    value=True
)


show_trails = st.sidebar.checkbox(

    "〰️ Afficher les trajectoires",

    value=True
)


show_congestion = st.sidebar.checkbox(

    "🔥 Afficher les zones congestionnées",

    value=True
)


show_ground = st.sidebar.checkbox(

    "🛬 Afficher avions au sol",

    value=False
)


st.sidebar.markdown("---")


if st.sidebar.button(
    "🔄 ACTUALISER LE TRAFIC",
    use_container_width=True
):

    st.cache_data.clear()

    st.rerun()


st.sidebar.markdown("---")

st.sidebar.caption(
    "Source : data.csv"
)

st.sidebar.caption(
    "ADS-B réel contenu dans le fichier"
)

st.sidebar.caption(
    "ML : XGBoost +2h"
)


# ============================================================
# 13. FILTRE PAYS
# ============================================================

if latest_adsb.empty:

    selected_data = pd.DataFrame()

else:

    if selected_country == "Tous les pays":

        selected_data = (
            latest_adsb.copy()
        )

    else:

        selected_data = (

            latest_adsb[

                latest_adsb["country"]
                ==
                selected_country

            ]

            .copy()
        )


    if not show_ground:

        selected_data = (

            selected_data[

                ~selected_data[
                    "on_ground"
                ]

            ]

            .copy()
        )


# ============================================================
# 14. ZONE CARTE
# ============================================================

if selected_country == "Tous les pays":

    map_lat = 7.0
    map_lon = 10.0
    map_zoom = 3.2

    map_area = (
        55
        *
        73
    )

else:

    zone = ASECNA_ZONES[
        selected_country
    ]

    map_lat = (

        zone["lat_min"]
        +
        zone["lat_max"]

    ) / 2


    map_lon = (

        zone["lon_min"]
        +
        zone["lon_max"]

    ) / 2


    width = (

        zone["lon_max"]
        -
        zone["lon_min"]
    )


    if width > 15:

        map_zoom = 3.8

    elif width > 8:

        map_zoom = 4.5

    elif width > 4:

        map_zoom = 5.5

    else:

        map_zoom = 6.2


    map_area = (

        zone["lat_max"]
        -
        zone["lat_min"]

    ) * (

        zone["lon_max"]
        -
        zone["lon_min"]
    )


# ============================================================
# 15. MÉTRIQUES
# ============================================================

metrics = calculate_congestion(

    selected_data,

    map_area
)


traffic_count = metrics[
    "traffic_count"
]


density = metrics[
    "density"
]

avg_speed = metrics[
    "avg_speed"
]

speed_var = metrics[
    "speed_var"
]

congestion = metrics[
    "congestion"
]

current_level = congestion_level(
    congestion
)


# ============================================================
# 16. TITRE
# ============================================================

st.markdown(
    """
    <div class="main-title">
        ✈️ ASECNA AIR TRAFFIC INTELLIGENCE
    </div>

    
    """,
    unsafe_allow_html=True
)


if latest_adsb.empty:

    st.error(
        "🔴 TRAFIC ADS-B INDISPONIBLE"
    )

    if load_error:

        st.warning(
            f"Diagnostic : {load_error}"
        )

    st.stop()


# ============================================================
# 19. KPI
# ============================================================

st.markdown(
    
    """
        <div class="section-title">
             📊 SITUATION DU TRAFIC
        </div>
        """,
        unsafe_allow_html=True
)


k1, k2, k3, k4= st.columns(4)


with k1:

    st.metric(

        "✈️ AVIONS",

        f"{traffic_count}"
    )


with k2:

    st.metric(

        "🚦 CONGESTION",

        f"{congestion:.2f}",

        help="Indice compris entre 0 et 1"
    )


with k3:

    st.metric(

        "📊 DENSITÉ",

        f"{density:.3f}"
    )


with k4:

    st.metric(

        "🚀 VITESSE MOY.",

        f"{avg_speed:.0f} km/h"
    )




# ============================================================
# 20. ÉTAT CONGESTION
# ============================================================

if congestion < 0.30:

    st.success(
        f"🟢 CONGESTION : {current_level}"
    )

elif congestion < 0.60:

    st.warning(
        f"🟡 CONGESTION : {current_level}"
    )

elif congestion < 0.80:

    st.warning(
        f"🟠 CONGESTION : {current_level}"
    )

else:

    st.error(
        f"🔴 CONGESTION : {current_level}"
    )

# ============================================================
# 36. PRÉVISION XGBOOST
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🔮 PRÉVISION DE CONGESTION À +2H
    </div>
    """,
    unsafe_allow_html=True
)


prediction = None


if MODEL_PATH is None:

    st.warning(
        "⚠️ Modèle XGBoost introuvable dans models/."
    )

elif FEATURES_PATH is None:

    st.warning(
        "⚠️ Fichier de features introuvable."
    )

else:

    try:

        model = joblib.load(
            MODEL_PATH
        )


        feature_cols = joblib.load(
            FEATURES_PATH
        )


        if isinstance(
            feature_cols,
            np.ndarray
        ):

            feature_cols = (
                feature_cols
                .tolist()
            )


        feature_cols = list(
            feature_cols
        )


        # ----------------------------------------------------
        # DATETIME
        # ----------------------------------------------------

        if (
            not selected_data.empty
            and
            "collected_at"
            in selected_data.columns
        ):

            current_datetime = (

                selected_data[
                    "collected_at"
                ].max()
            )

        else:

            current_datetime = (
                pd.Timestamp.now(
                    tz="UTC"
                )
            )


        if pd.isna(
            current_datetime
        ):

            current_datetime = (
                pd.Timestamp.now(
                    tz="UTC"
                )
            )


        hour = (
            current_datetime.hour
        )

        dayofweek = (
            current_datetime.dayofweek
        )

        month = (
            current_datetime.month
        )

        dayofyear = (
            current_datetime.dayofyear
        )


        # ----------------------------------------------------
        # FEATURES DE BASE
        # ----------------------------------------------------

        feature_values = {

            "traffic_count":
                traffic_count,

            "density":
                density,

            "avg_speed":
                avg_speed,

            "speed_var":
                speed_var,

            "congestion_index":
                congestion,

            "hour":
                hour,

            "dayofweek":
                dayofweek,

            "month":
                month,

            "dayofyear":
                dayofyear,

            "hour_sin":
                np.sin(
                    2 * np.pi
                    * hour
                    / 24
                ),

            "hour_cos":
                np.cos(
                    2 * np.pi
                    * hour
                    / 24
                ),

            "dow_sin":
                np.sin(
                    2 * np.pi
                    * dayofweek
                    / 7
                ),

            "dow_cos":
                np.cos(
                    2 * np.pi
                    * dayofweek
                    / 7
                )
        }


        # ----------------------------------------------------
        # FEATURES MANQUANTES
        #
        # Le modèle entraîné contient également
        # des variables temporelles/lags/rolling.
        # On les initialise à 0 si elles ne peuvent pas
        # être reconstruites à partir du fichier actuel.
        # ----------------------------------------------------

        X_pred = pd.DataFrame(
            [feature_values]
        )


        for feature in feature_cols:

            if feature not in X_pred.columns:

                X_pred[feature] = 0.0


        # ----------------------------------------------------
        # ORDRE EXACT
        # ----------------------------------------------------

        X_pred = X_pred[
            feature_cols
        ].copy()


        # ----------------------------------------------------
        # NETTOYAGE
        # ----------------------------------------------------

        X_pred = (

            X_pred

            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )

            .fillna(0)
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = float(

            model.predict(
                X_pred
            )[0]
        )


        prediction = float(

            np.clip(
                prediction,
                0,
                1
            )
        )


        prediction_level = (
            congestion_level(
                prediction
            )
        )


       

        # ----------------------------------------------------
        # JAUGE
        # ----------------------------------------------------

        fig_gauge = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=prediction,

                title={
                    "text":
                        (
                           # "Prévision congestion +2h — "
                           f"{prediction_level}"
                        )
                },

                gauge={

                    "axis": {

                        "range": [
                            0,
                            1
                        ]
                    },

                    "steps": [

                        {
                            "range": [
                                0,
                                0.30
                            ]
                        },

                        {
                            "range": [
                                0.30,
                                0.60
                            ]
                        },

                        {
                            "range": [
                                0.60,
                                0.80
                            ]
                        },

                        {
                            "range": [
                                0.80,
                                1.00
                            ]
                        }
                    ]
                }
            )
        )


        fig_gauge.update_layout(

            height=330,

            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20
            ),

            paper_bgcolor=
                "rgba(0,0,0,0)",

            font=dict(
                color="#dff7ff"
            )
        )


        st.plotly_chart(

            fig_gauge,

            use_container_width=True
        )


    except Exception as e:

        st.error(
            "Erreur XGBoost : "
            f"{type(e).__name__} — {e}"
        )


# ============================================================
# 20. CARTE PYDECK — RADAR ATC ASECNA
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🗺️ TRAFIC AÉRIEN RÉEL
    </div>
    """,
    unsafe_allow_html=True
)

if selected_data.empty:

    st.warning("⚠️ Aucun avion disponible dans la zone sélectionnée.")

else:

    # ========================================================
    # 21. PRÉPARATION DES DONNÉES
    # ========================================================

    aircraft = selected_data.copy()

    # --------------------------------------------------------
    # Colonnes obligatoires
    # --------------------------------------------------------

    required_cols = [
        "latitude",
        "longitude"
    ]

    aircraft = aircraft.dropna(
        subset=required_cols
    )

    # --------------------------------------------------------
    # Coordonnées
    # --------------------------------------------------------

    aircraft["latitude"] = pd.to_numeric(
        aircraft["latitude"],
        errors="coerce"
    )

    aircraft["longitude"] = pd.to_numeric(
        aircraft["longitude"],
        errors="coerce"
    )

    aircraft = aircraft.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    )

    # ========================================================
    # 22. ICAO24
    # ========================================================

    if "icao24" not in aircraft.columns:

        aircraft["icao24"] = "UNKNOWN"

    aircraft["icao24"] = (
        aircraft["icao24"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # ========================================================
    # 23. CALLSIGN
    # ========================================================

    if "callsign" not in aircraft.columns:

        aircraft["callsign"] = ""

    aircraft["callsign"] = (
        aircraft["callsign"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    aircraft["label"] = aircraft.apply(
        lambda x:
            x["callsign"]
            if x["callsign"] not in ["", "nan", "None"]
            else x["icao24"],
        axis=1
    )

    # ========================================================
    # 24. HEADING
    # ========================================================

    if "heading" not in aircraft.columns:

        aircraft["heading"] = 0.0

    aircraft["heading"] = pd.to_numeric(
        aircraft["heading"],
        errors="coerce"
    ).fillna(0)

    aircraft["heading"] = (
        aircraft["heading"] % 360
    )

    # ========================================================
    # 25. ALTITUDE
    # ========================================================

    if "baro_altitude" in aircraft.columns:

        aircraft["altitude_ft"] = (
            pd.to_numeric(
                aircraft["baro_altitude"],
                errors="coerce"
            )
            .fillna(0)
            * 3.28084
        )

    else:

        aircraft["altitude_ft"] = 0

    # ========================================================
    # 26. VITESSE
    # ========================================================

    if "velocity" in aircraft.columns:

        aircraft["speed_kmh"] = (
            pd.to_numeric(
                aircraft["velocity"],
                errors="coerce"
            )
            .fillna(0)
            * 3.6
        )

    else:

        aircraft["speed_kmh"] = 0

    # ========================================================
    # 27. PAYS
    # ========================================================

    if "country" not in aircraft.columns:

        if "origin_country" in aircraft.columns:

            aircraft["country"] = (
                aircraft["origin_country"]
                .fillna("Inconnu")
                .astype(str)
            )

        else:

            aircraft["country"] = "Inconnu"

    aircraft["country"] = (
        aircraft["country"]
        .fillna("Inconnu")
        .astype(str)
    )

    # ========================================================
    # 28. ORIGINE
    # ========================================================

    if "origin_country" not in aircraft.columns:

        aircraft["origin_country"] = "Inconnu"

    aircraft["origin_country"] = (
        aircraft["origin_country"]
        .fillna("Inconnu")
        .astype(str)
    )

    # ========================================================
    # 29. POSITION PYDECK
    # ========================================================

    aircraft["position"] = aircraft.apply(
        lambda row: [
            float(row["longitude"]),
            float(row["latitude"])
        ],
        axis=1
    )

    # ========================================================
    # 30. VRAIE ICÔNE AVION RADAR ATC
    # ========================================================
    #
    # Avion orienté vers le NORD.
    # PyDeck applique ensuite le heading.
    #
    # Blanc = corps de l'avion
    # Cyan  = contour
    # Cyan transparent = halo
    #
    # ========================================================

    import urllib.parse

    aircraft_svg = """
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width="128"
        height="128"
        viewBox="0 0 128 128">

        <!-- Halo -->
        <circle
            cx="64"
            cy="64"
            r="58"
            fill="none"
            stroke="#00D9FF"
            stroke-width="2"
            opacity="0.20"/>

        <!-- Corps principal -->
        <path
            d="
            M64 7

            C61 7 59 10 58 15

            L52 50

            L14 65

            C11 66 10 69 12 71

            L18 76

            L52 69

            L52 103

            L43 115

            L43 120

            L64 111

            L85 120

            L85 115

            L76 103

            L76 69

            L110 76

            L116 71

            C118 69 117 66 114 65

            L76 50

            L70 15

            C69 10 67 7 64 7

            Z
            "

            fill="#FFFFFF"

            stroke="#00D9FF"

            stroke-width="4"

            stroke-linejoin="round"/>

        <!-- Ligne centrale -->
        <path
            d="
            M64 18
            L64 103
            "

            stroke="#00BFFF"

            stroke-width="2"

            opacity="0.85"/>

        <!-- Nez lumineux -->
        <circle
            cx="64"
            cy="12"
            r="4"

            fill="#00FFFF"/>

    </svg>
    """

    # Encodage propre pour éviter les problèmes SVG
    aircraft_icon_url = (
        "data:image/svg+xml;charset=utf-8,"
        + urllib.parse.quote(
            aircraft_svg
        )
    )

    # ========================================================
    # 31. OBJET ICÔNE POUR CHAQUE AVION
    # ========================================================

    aircraft["icon"] = aircraft.apply(
        lambda row: {
            "url": aircraft_icon_url,

            "width": 128,

            "height": 128,

            "anchorY": 64,

            "anchorX": 64,

            "mask": False
        },
        axis=1
    )

    # ========================================================
    # 32. COUCHE HALO RADAR
    # ========================================================

    halo_layer = pdk.Layer(

        "ScatterplotLayer",

        data=aircraft,

        get_position="position",

        get_radius=3500,

        radius_min_pixels=4,

        radius_max_pixels=10,

        get_fill_color=[
            0,
            217,
            255,
            28
        ],

        get_line_color=[
            0,
            217,
            255,
            110
        ],

        line_width_min_pixels=1,

        stroked=True,

        filled=True,

        pickable=False
    )

    # ========================================================
    # 33. ICÔNES AVIONS
    # ========================================================

    aircraft_layer = pdk.Layer(

        "IconLayer",

        data=aircraft,

        get_icon="icon",

        get_position="position",

        # ----------------------------------------------------
        # IMPORTANT :
        # 0° = Nord
        # 90° = Est
        # 180° = Sud
        # 270° = Ouest
        # ----------------------------------------------------

        get_angle="heading",

        # Taille réelle
        get_size=38,

        size_scale=1,

        # Empêche les avions de devenir minuscules
        size_min_pixels=26,

        size_max_pixels=48,

        # L'avion reste orienté sur la carte
        billboard=False,

        # Important pour les événements de clic
        pickable=True,

        auto_highlight=True,

        highlight_color=[
            255,
            255,
            255,
            255
        ]
    )

    # ========================================================
    # 34. LAYERS
    # ========================================================

    layers = []

    # --------------------------------------------------------
    # Congestion
    # --------------------------------------------------------

    if show_congestion:

        congestion = (
            aircraft
            .assign(
                grid_lat=lambda x:
                    np.floor(
                        x["latitude"] / 0.5
                    ) * 0.5,

                grid_lon=lambda x:
                    np.floor(
                        x["longitude"] / 0.5
                    ) * 0.5
            )
            .groupby(
                [
                    "grid_lat",
                    "grid_lon"
                ],
                as_index=False
            )
            .agg(
                aircraft=(
                    "icao24",
                    "nunique"
                )
            )
        )

        if not congestion.empty:

            congestion["polygon"] = congestion.apply(

                lambda r: [

                    [
                        r["grid_lon"],
                        r["grid_lat"]
                    ],

                    [
                        r["grid_lon"] + 0.5,
                        r["grid_lat"]
                    ],

                    [
                        r["grid_lon"] + 0.5,
                        r["grid_lat"] + 0.5
                    ],

                    [
                        r["grid_lon"],
                        r["grid_lat"] + 0.5
                    ]

                ],

                axis=1
            )

            congestion_layer = pdk.Layer(

                "PolygonLayer",

                data=congestion,

                get_polygon="polygon",

                get_fill_color=[
                    255,
                    140,
                    0,
                    22
                ],

                get_line_color=[
                    255,
                    170,
                    0,
                    70
                ],

                get_line_width=1,

                stroked=True,

                filled=True,

                pickable=True
            )

            layers.append(
                congestion_layer
            )

    # --------------------------------------------------------
    # HALO
    # --------------------------------------------------------

    layers.append(
        halo_layer
    )

    # --------------------------------------------------------
    # AVIONS
    # --------------------------------------------------------

    layers.append(
        aircraft_layer
    )

    # ========================================================
    # 35. LABELS
    # ========================================================

    if show_labels:

        label_layer = pdk.Layer(

            "TextLayer",

            data=aircraft,

            get_position="position",

            get_text="label",

            get_size=12,

            size_units="pixels",

            get_color=[
                255,
                255,
                255,
                235
            ],

            get_pixel_offset=[
                0,
                -25
            ],

            get_text_anchor="middle",

            get_alignment_baseline="bottom",

            billboard=True,

            pickable=False
        )

        layers.append(
            label_layer
        )

    # ========================================================
    # 36. CENTRE AUTOMATIQUE
    # ========================================================

    center_lat = float(
        aircraft["latitude"].mean()
    )

    center_lon = float(
        aircraft["longitude"].mean()
    )

    # ========================================================
    # 37. ZOOM AUTOMATIQUE
    # ========================================================

    nb_aircraft = len(aircraft)

    if nb_aircraft <= 10:

        zoom = 6.0

    elif nb_aircraft <= 50:

        zoom = 5.0

    elif nb_aircraft <= 200:

        zoom = 4.3

    else:

        zoom = 3.8

    # ========================================================
    # 38. VUE
    # ========================================================

    view_state = pdk.ViewState(

        latitude=center_lat,

        longitude=center_lon,

        zoom=zoom,

        pitch=0,

        bearing=0
    )

    # ========================================================
    # 39. TOOLTIP ATC
    # ========================================================

    tooltip = {

        "html": """

        <div style="
            background:#07131F;
            border:1px solid #00D9FF;
            border-radius:6px;
            padding:10px 12px;
            color:white;
            font-family:Arial;
            min-width:240px;
            box-shadow:0 0 12px rgba(0,217,255,0.25);
        ">

            <div style="
                color:#00D9FF;
                font-size:17px;
                font-weight:bold;
                margin-bottom:8px;
            ">
                ✈ {label}
            </div>

            <div>
                <b>ICAO24 :</b> {icao24}
            </div>

            <div>
                <b>Pays :</b> {country}
            </div>

            <div>
                <b>Origine :</b> {origin_country}
            </div>

            <hr style="
                border:none;
                border-top:1px solid #244052;
                margin:7px 0;
            ">

            <div>
                <b>Altitude :</b>
                {altitude_ft} ft
            </div>

            <div>
                <b>Vitesse :</b>
                {speed_kmh} km/h
            </div>

            <div>
                <b>Cap :</b>
                {heading}°
            </div>

            <div>
                <b>Latitude :</b>
                {latitude}
            </div>

            <div>
                <b>Longitude :</b>
                {longitude}
            </div>

        </div>

        """,

        "style": {
            "backgroundColor": "transparent",
            "color": "white"
        }
    }

    # ========================================================
    # 40. DECK
    # ========================================================

    deck = pdk.Deck(

        layers=layers,

        initial_view_state=view_state,

        map_style=(
            "https://basemaps.cartocdn.com/gl/"
            "dark-matter-gl-style/style.json"
        ),

        tooltip=tooltip,

        map_provider="carto"
    )

    # ========================================================
    # 41. AFFICHAGE
    # ========================================================

    st.pydeck_chart(
        deck,
        use_container_width=True
    )



# ============================================================
# 37. TABLEAU TRAFIC PAR PAYS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🌍 TRAFIC ET CONGESTION PAR PAYS
    </div>
    """,
    unsafe_allow_html=True
)


country_results = []


for country, zone in ASECNA_ZONES.items():

    country_df = (

        latest_adsb[
            latest_adsb["country"]
            ==
            country
        ]

        .copy()
    )


    if not show_ground:

        country_df = (

            country_df[
                ~country_df["on_ground"]
            ]

            .copy()
        )


    area = (

        zone["lat_max"]
        -
        zone["lat_min"]

    ) * (

        zone["lon_max"]
        -
        zone["lon_min"]
    )


    m = calculate_congestion(

        country_df,

        area
    )


    country_results.append({

        "Pays":
            country,

        "✈️ Avions":
            m["traffic_count"],

        "📡 Observations":
            m["observations"],

        "🚀 Vitesse km/h":
            round(
                m["avg_speed"],
                1
            ),

        "📊 Densité":
            round(
                m["density"],
                4
            ),

        "🚦 Congestion":
            round(
                m["congestion"],
                3
            ),

        "Niveau":
            congestion_level(
                m["congestion"]
            )
    })


country_table = pd.DataFrame(
    country_results
)


country_table = (

    country_table

    .sort_values(
        "🚦 Congestion",
        ascending=False
    )
)


st.dataframe(

    country_table,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# 38. GRAPHIQUE TRAFIC
# ============================================================

if not country_table.empty:

    c1, c2 = st.columns(2)


    with c1:

        fig_traffic = px.bar(

            country_table,

            x="Pays",

            y="✈️ Avions",

            title=
                "✈️ TRAFIC ADS-B RÉEL PAR PAYS"
        )


        fig_traffic.update_layout(

            height=430,

            xaxis_tickangle=-45,

            paper_bgcolor=
                "rgba(0,0,0,0)",

            plot_bgcolor=
                "rgba(0,0,0,0)",

            font=dict(
                color="#dff7ff"
            )
        )


        st.plotly_chart(

            fig_traffic,

            use_container_width=True
        )


    with c2:

        fig_congestion = px.bar(

            country_table,

            x="Pays",

            y="🚦 Congestion",

            title=
                "🚦 CONGESTION PAR PAYS"
        )


        fig_congestion.update_layout(

            height=430,

            xaxis_tickangle=-45,

            yaxis=dict(
                range=[
                    0,
                    1
                ]
            ),

            paper_bgcolor=
                "rgba(0,0,0,0)",

            plot_bgcolor=
                "rgba(0,0,0,0)",

            font=dict(
                color="#dff7ff"
            )
        )


        st.plotly_chart(

            fig_congestion,

            use_container_width=True
        )


# ============================================================
# 39. LISTE DES AVIONS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        ✈️ FLIGHT LIST — TRAFIC ACTUEL
    </div>
    """,
    unsafe_allow_html=True
)


if not selected_data.empty:

    columns = [

        "icao24",

        "callsign",

        "origin_country",

        "country",

        "latitude",

        "longitude",

        "altitude_ft",

        "speed_kmh",

        "heading",

        "vertical_rate",

        "on_ground",

        "collected_at"
    ]


    available_columns = [

        c

        for c in columns

        if c in selected_data.columns
    ]


    flight_table = (

        selected_data[
            available_columns
        ]

        .copy()
    )


    flight_table = flight_table.rename(

        columns={

            "icao24":
                "ICAO24",

            "callsign":
                "CALLSIGN",

            "origin_country":
                "ORIGIN",

            "country":
                "ZONE",

            "latitude":
                "LAT",

            "longitude":
                "LON",

            "altitude_ft":
                "ALT FT",

            "speed_kmh":
                "SPEED KM/H",

            "heading":
                "HDG °",

            "vertical_rate":
                "V/S",

            "on_ground":
                "GROUND",

            "collected_at":
                "TIME UTC"
        }
    )


    st.dataframe(

        flight_table,

        use_container_width=True,

        hide_index=True
    )

else:

    st.info(
        "Aucun avion dans le secteur sélectionné."
    )



# ============================================================
# 41. FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

        ✈️ ASECNA AIR TRAFFIC INTELLIGENCE
    
        CONGESTION FORECAST +2H

    </div>$y
    """,
    unsafe_allow_html=True
)