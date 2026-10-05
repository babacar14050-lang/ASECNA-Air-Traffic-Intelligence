# ✈️ ASECNA Air Traffic Intelligence

### Plateforme intelligente de surveillance et de prédiction de la congestion du trafic aérien basée sur les données ADS-B et le Machine Learning

<p align="center">
  <strong>Surveillance temps réel • Analyse ADS-B • Prédiction de congestion • Visualisation interactive</strong>
</p>

---

## 📌 Présentation du projet

**ASECNA Air Traffic Intelligence** est une plateforme intelligente dédiée à la **surveillance, l'analyse et la prédiction de la congestion du trafic aérien** dans l'espace aérien couvert par l'ASECNA.

Le système exploite les données issues de l'**Automatic Dependent Surveillance–Broadcast (ADS-B)** afin de construire une vision dynamique du trafic aérien et d'identifier les situations susceptibles de générer une congestion.

La plateforme combine :

* la collecte et le traitement des données ADS-B ;
* l'analyse spatio-temporelle du trafic aérien ;
* le calcul d'indicateurs de congestion ;
* l'utilisation du **Machine Learning**, notamment **XGBoost** ;
* la prédiction de la congestion à court terme ;
* la visualisation interactive des aéronefs ;
* un dashboard développé avec **Streamlit**.

L'objectif est de fournir un outil d'aide à l'analyse du trafic permettant de mieux comprendre la dynamique du trafic aérien et d'anticiper les périodes de forte densité.

---

## 🎯 Objectifs

### Objectif général

Concevoir et développer un système intelligent capable de **surveiller et prédire la congestion du trafic aérien** à partir des données ADS-B.

### Objectifs spécifiques

* Collecter et exploiter des données ADS-B.
* Nettoyer et préparer les données aéronautiques.
* Identifier les positions et caractéristiques des aéronefs.
* Analyser la répartition géographique du trafic.
* Construire des indicateurs de congestion.
* Développer un modèle de prédiction basé sur le Machine Learning.
* Évaluer les performances du modèle.
* Visualiser le trafic aérien sur une carte interactive.
* Fournir une interface de surveillance accessible aux utilisateurs.
* Préparer la solution pour un déploiement cloud.

---

# 🏗️ Architecture du système

```text
                    ┌──────────────────────┐
                    │     Données ADS-B    │
                    │  Positions / vitesse │
                    │ altitude / heading   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Collecte & Nettoyage │
                    │      des données      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │                      │
                    │ • densité trafic     │
                    │ • vitesse            │
                    │ • altitude           │
                    │ • variation vitesse  │
                    │ • indicateurs spatio │
                    │   temporels           │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
      ┌────────────────────┐      ┌────────────────────┐
      │ Analyse du trafic  │      │ Machine Learning   │
      │                    │      │                    │
      │ • densité          │      │      XGBoost       │
      │ • distribution     │      │                    │
      │ • congestion       │      │ Prédiction +2h     │
      └──────────┬─────────┘      └──────────┬─────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Dashboard Streamlit  │
                    │                      │
                    │ • Carte ATC          │
                    │ • Aéronefs           │
                    │ • Statistiques       │
                    │ • Congestion         │
                    │ • Prédictions        │
                    └──────────────────────┘
```

---

# 📊 Données ADS-B

Le système exploite principalement les informations ADS-B associées aux aéronefs.

Les variables utilisées peuvent notamment inclure :

| Variable         | Description                     |
| ---------------- | ------------------------------- |
| `icao24`         | Identifiant unique de l'aéronef |
| `callsign`       | Indicatif de l'aéronef          |
| `latitude`       | Latitude                        |
| `longitude`      | Longitude                       |
| `baro_altitude`  | Altitude barométrique           |
| `geo_altitude`   | Altitude géométrique            |
| `velocity`       | Vitesse                         |
| `heading`        | Cap de l'aéronef                |
| `vertical_rate`  | Taux de montée/descente         |
| `on_ground`      | État au sol/en vol              |
| `origin_country` | Pays d'origine                  |
| `collected_at`   | Date et heure de collecte       |

Les données sont ensuite transformées afin de produire des variables adaptées à l'analyse et au Machine Learning.

---

# 🧠 Machine Learning

## Modèle utilisé

Le modèle principal utilisé dans le projet est :

**XGBoost — Extreme Gradient Boosting**

XGBoost est particulièrement adapté aux données tabulaires et permet de modéliser des relations non linéaires entre les caractéristiques du trafic et le niveau de congestion.

### Variables caractéristiques

Selon la configuration du modèle, les caractéristiques peuvent inclure :

* nombre d'aéronefs ;
* densité du trafic ;
* vitesse moyenne ;
* variation de vitesse ;
* altitude moyenne ;
* variation d'altitude ;
* indicateurs temporels ;
* indicateurs géographiques.

### Horizon de prédiction

Le système est conçu pour effectuer notamment une :

> **Prédiction de la congestion à +2 heures**

Cette approche permet d'aller au-delà de la simple surveillance et d'introduire une dimension prédictive dans l'analyse du trafic aérien.

---

# 🚦 Indicateur de congestion

La plateforme calcule un indice synthétique de congestion compris entre **0 et 1**.

L'indicateur combine notamment :

* le nombre d'aéronefs ;
* la densité du trafic ;
* la variabilité des vitesses.

Une pondération utilisée dans le système est :

```text
Congestion =
    0.50 × Traffic Score
  + 0.30 × Density Score
  + 0.20 × Speed Variation Score
```

### Classification

|       Score | Niveau        |
| ----------: | ------------- |
| 0.00 – 0.29 | 🟢 Faible     |
| 0.30 – 0.59 | 🟡 Moyenne    |
| 0.60 – 0.79 | 🟠 Forte      |
| 0.80 – 1.00 | 🔴 Très forte |

Cet indicateur constitue un **indicateur analytique du projet** et ne remplace pas les procédures opérationnelles officielles de gestion du trafic aérien.

---

# 🌍 Couverture géographique

La plateforme permet d'analyser différentes zones et pays associés à l'espace de travail du projet.

Les secteurs intégrés comprennent notamment :

* 🇧🇯 Bénin
* 🇧🇫 Burkina Faso
* 🇨🇲 Cameroun
* 🇨🇫 République centrafricaine
* 🇰🇲 Comores
* 🇨🇬 Congo
* 🇨🇮 Côte d'Ivoire
* 🇬🇦 Gabon
* 🇬🇼 Guinée-Bissau
* 🇬🇶 Guinée équatoriale
* 🇲🇬 Madagascar
* 🇲🇱 Mali
* 🇲🇷 Mauritanie
* 🇳🇪 Niger
* 🇷🇼 Rwanda
* 🇸🇳 Sénégal
* 🇹🇩 Tchad
* 🇹🇬 Togo

La plateforme permet notamment de filtrer les données par secteur/pays afin d'obtenir une analyse ciblée du trafic.

---

# 🖥️ Dashboard interactif

L'interface utilisateur est développée avec **Streamlit**.

Elle fournit notamment :

### ✈️ Surveillance aérienne

* Position des aéronefs.
* Identification `ICAO24`.
* Callsign.
* Altitude.
* Vitesse.
* Cap.
* État de l'aéronef.

### 🗺️ Cartographie

Une carte interactive permet de visualiser :

* la position des aéronefs ;
* leur direction ;
* leur répartition géographique ;
* les secteurs sélectionnés ;
* les zones de concentration du trafic.

### 📊 Analyse

Le dashboard permet également de suivre :

* nombre d'aéronefs ;
* densité du trafic ;
* vitesse moyenne ;
* variation de vitesse ;
* niveau de congestion ;
* congestion par pays ;
* prédiction de congestion.

---

# 🗂️ Structure du projet

```text
ASECNA-Air-Traffic-Intelligence/
│
├── 📄 dashbord.py
├── 📄 collector.py
├── 📄 filtre.py
├── 📄 LSTM.py
├── 📄 notebook.py
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 .gitignore
│
├── 📁 data/
│   └── data.csv
│
└── 📁 models/
    └── modèles Machine Learning
```

### Description des principaux fichiers

| Fichier            | Rôle                                         |
| ------------------ | -------------------------------------------- |
| `dashbord.py`      | Application Streamlit et dashboard principal |
| `collector.py`     | Collecte des données ADS-B                   |
| `filtre.py`        | Filtrage et préparation des données          |
| `LSTM.py`          | Expérimentation avec un modèle LSTM          |
| `notebook.py`      | Analyse / expérimentation                    |
| `data/data.csv`    | Données utilisées par l'application          |
| `models/`          | Modèles Machine Learning                     |
| `requirements.txt` | Dépendances Python                           |
| `.gitignore`       | Fichiers exclus du dépôt                     |

---

# ⚙️ Installation

## 1. Cloner le dépôt

```bash
git clone https://github.com/babacar14050-lang/ASECNA-Air-Traffic-Intelligence.git
```

Puis :

```bash
cd ASECNA-Air-Traffic-Intelligence
```

## 2. Créer un environnement virtuel

### Windows

```bash
python -m venv .venv
```

Activer l'environnement :

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# 📦 Installation des dépendances

Installer les bibliothèques nécessaires :

```bash
pip install -r requirements.txt
```

Les principales technologies utilisées sont :

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib
* Plotly
* PyDeck
* Streamlit

---

# ▶️ Lancer le dashboard

Depuis le répertoire du projet :

```bash
streamlit run dashbord.py
```

L'application sera ensuite accessible localement depuis l'adresse indiquée par Streamlit.

---

# ☁️ Déploiement

Le projet peut être déployé sur **Streamlit Community Cloud**.

Configuration recommandée :

```text
Repository : babacar14050-lang/ASECNA-Air-Traffic-Intelligence
Branch     : main
Main file  : dashbord.py
```

Le fichier suivant doit être présent à la racine du dépôt :

```text
requirements.txt
```

et les données nécessaires au fonctionnement de l'application doivent être disponibles dans :

```text
data/data.csv
```

---

# 🔒 Gestion des données

Les données ADS-B peuvent être volumineuses et certaines données peuvent être sensibles ou soumises à des restrictions de redistribution.

Le dépôt Git doit donc éviter de contenir :

* données brutes volumineuses inutiles ;
* fichiers temporaires ;
* secrets ;
* identifiants ;
* fichiers `.env` ;
* modèles expérimentaux non nécessaires.

Les règles correspondantes sont définies dans `.gitignore`.

---

# 📈 Évaluation du modèle

L'évaluation du modèle Machine Learning peut s'appuyer sur plusieurs métriques adaptées à la tâche étudiée.

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* R² — Coefficient de détermination

Les résultats expérimentaux peuvent être intégrés dans le dashboard ou présentés dans le mémoire.

---

# 🔬 Perspectives d'amélioration

Plusieurs extensions sont envisageables :

### Court terme

* amélioration du nettoyage ADS-B ;
* augmentation du volume de données historiques ;
* optimisation des variables explicatives ;
* amélioration du calcul de congestion.

### Moyen terme

* prédictions multi-horizons : **+30 min, +1 h, +2 h, +6 h** ;
* comparaison XGBoost / Random Forest / LightGBM ;
* intégration de modèles temporels ;
* amélioration de la détection des zones de forte densité.

### Long terme

* architecture de traitement en temps réel ;
* streaming ADS-B continu ;
* prédiction multi-secteurs ;
* détection automatique des anomalies ;
* intégration de données météorologiques ;
* intégration avec des données de plans de vol ;
* système d'aide à la décision pour l'analyse du trafic aérien.

---

# 🎓 Contexte académique

Ce projet s'inscrit dans le cadre d'un travail académique portant sur l'application du **Machine Learning à l'analyse du trafic aérien**.

Il combine plusieurs domaines :

```text
ADS-B
  ↓
Data Engineering
  ↓
Feature Engineering
  ↓
Machine Learning
  ↓
Prédiction
  ↓
Visualisation
  ↓
Aide à l'analyse du trafic aérien
```

---

# ⚠️ Limites du système

Cette plateforme constitue un **prototype de recherche et d'aide à l'analyse**.

Les prédictions et indicateurs produits ne doivent pas être considérés comme des instructions opérationnelles de contrôle de la circulation aérienne.

Les performances dépendent notamment :

* de la qualité des données ADS-B ;
* de la couverture des récepteurs ;
* du volume historique disponible ;
* de la qualité des variables utilisées ;
* de la représentativité des données d'entraînement ;
* des conditions opérationnelles réelles.

---


<p align="center">

### ✈️ ASECNA Air Traffic Intelligence

<strong>Surveiller • Analyser • Prédire</strong>

</p>
