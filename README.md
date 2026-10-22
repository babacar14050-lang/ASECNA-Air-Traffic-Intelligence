# ✈️ Conception et développement d’un système intelligent de prédiction de la congestion du trafic aérien

## 📌 1. Présentation du projet

Ce projet porte sur la **conception et le développement d'un système intelligent de prédiction de la congestion du trafic aérien**, fondé sur l'exploitation des données de surveillance aéronautique et l'utilisation de techniques d'intelligence artificielle.

L'objectif principal est de développer une plateforme capable de collecter, traiter, analyser et visualiser les données de trafic aérien afin d'estimer le niveau de congestion dans des zones géographiques déterminées.

Le système s'appuie notamment sur les données ADS-B (*Automatic Dependent Surveillance–Broadcast*) et sur des algorithmes de Machine Learning pour étudier la répartition spatiale des aéronefs, leur vitesse, leur altitude et la densité du trafic.

Une attention particulière est accordée à l'espace aérien mauritanien, notamment aux secteurs Est et Ouest, tout en permettant une analyse comparative avec les autres pays de la région.

## 🎯 2. Objectifs du projet

Les objectifs principaux sont les suivants :

- Collecter et exploiter les données de trafic aérien issues des sources ADS-B disponibles.
- Identifier les aéronefs à partir de leurs indicatifs de vol (*callsigns*), lorsque ceux-ci sont disponibles.
- Analyser la répartition géographique du trafic aérien.
- Calculer des indicateurs de densité, de vitesse et de concentration du trafic.
- Détecter les situations de forte densité et estimer le niveau de congestion.
- Développer un modèle prédictif utilisant des algorithmes de Machine Learning, notamment XGBoost.
- Concevoir un tableau de bord interactif avec Streamlit.
- Faciliter la surveillance et l'analyse du trafic dans l'espace aérien mauritanien.

## 🛰️ 3. Sources de données

Le système exploite des données de surveillance aéronautique disponibles auprès de sources compatibles avec le projet.

Les données ADS-B peuvent notamment contenir :

| Variable | Description |
|---|---|
| `timestamp` | Date et heure de l'observation |
| `icao24` | Adresse ICAO de l'aéronef |
| `callsign` | Indicatif de vol disponible |
| `latitude` | Position géographique en latitude |
| `longitude` | Position géographique en longitude |
| `altitude` | Altitude de l'aéronef |
| `velocity` | Vitesse de l'aéronef |
| `heading` | Cap de l'aéronef |
| `vertical_rate` | Taux de montée ou de descente |

La disponibilité et la qualité de ces variables dépendent de la source de données utilisée.

**Remarque :** un callsign doit correspondre à un indicatif réellement observé ou vérifié. Les identifiants ICAO, les indicatifs de vol et les immatriculations sont des informations distinctes.

## 🧠 4. Approche méthodologique

Le développement du système suit plusieurs étapes.

### 4.1. Collecte des données

Récupération périodique des observations aéronautiques à partir d'une source ADS-B de opensky et d'une API autorisée.

### 4.2. Prétraitement des données

- Nettoyage des données manquantes et incohérentes.
- Conversion des horodatages.
- Vérification des coordonnées géographiques.
- Suppression des observations dupliquées.
- Filtrage des observations selon les zones d'étude.

### 4.3. Analyse descriptive

Calcul d'indicateurs permettant de caractériser le trafic aérien :

- Nombre d'aéronefs observés.
- Densité du trafic par secteur.
- Répartition géographique des observations.
- Distribution des altitudes et des vitesses.
- Évolution temporelle du trafic.
- Comparaison entre les secteurs Est et Ouest de la Mauritanie.

### 4.4. Calcul du niveau de congestion

Le niveau de congestion est estimé à partir d'indicateurs de trafic, tels que la densité des observations, leur évolution temporelle et, lorsque les données le permettent, la proximité entre les aéronefs.

Le système peut ensuite classer les situations en plusieurs niveaux, par exemple :

- Faible congestion.
- Congestion modérée.
- Forte congestion.

Les seuils de classification doivent être définis et validés à partir de données pertinentes. Une forte densité d'observations ne constitue pas, à elle seule, une preuve de conflit ou de surcharge opérationnelle.

### 4.5. Modélisation prédictive

L'algorithme **XGBoost** est envisagé pour apprendre les relations entre les caractéristiques du trafic et le niveau de congestion observé.

Les variables explicatives peuvent inclure :

- La densité actuelle du trafic.
- L'évolution du nombre d'aéronefs.
- La vitesse moyenne.
- La variabilité des vitesses.
- L'heure et le secteur géographique.
- Les observations historiques.

Le modèle vise à estimer un niveau de congestion futur sur un horizon temporel défini.

Ses performances doivent être évaluées sur des données de test indépendantes, avec des métriques adaptées au type de prédiction réalisé.

## 📊 5. Tableau de bord interactif

L'application est conçue pour proposer une interface de visualisation et d'analyse du trafic aérien.

Les fonctionnalités prévues ou développées peuvent inclure :

- **Vue générale :** principaux indicateurs du trafic.
- **Carte aéronautique :** représentation des positions des aéronefs.
- **Analyse géographique :** comparaison des pays et des secteurs.
- **Analyse temporelle :** évolution du trafic au cours du temps.
- **Indicateurs de congestion :** suivi du niveau de densité par zone.
- **Prédiction intelligente :** visualisation des estimations du modèle XGBoost.
- **Filtres interactifs :** sélection des pays, secteurs et périodes d'analyse.
- **Actualisation des données :** mise à jour selon la fréquence autorisée par la source de données.

## 🛠️ 6. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python |