"""
Application Streamlit — Système de recommandation de culture agricole

Lancement :
    streamlit run streamlit_app.py

Charge les artefacts entraînés (modèle, scaler, encodeur de la cible, liste des
features) et propose un formulaire de saisie + une prédiction instantanée.
"""

import os
import joblib
import numpy as np
import streamlit as st
import streamlit_antd_components as sac

# ---------------------------------------------------------------------------
# Configuration (CHEMIN DYNAMIQUE VERS model_artifacts)
# ---------------------------------------------------------------------------

# Le dossier "model_artifacts" doit être DANS le même dossier que ce script.
# Peu importe si vous êtes sur Windows, Linux ou Streamlit Cloud.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOSSIER_ARTEFACTS = os.path.join(BASE_DIR, "model_artifacts")

# Plages plausibles observées dans le dataset d'entraînement (Crop_recommendation.csv).
PLAGES_ENTRAINEMENT = {
    "N": (0, 140),
    "P": (5, 145),
    "K": (5, 205),
    "temperature": (8.8, 43.7),
    "humidity": (14.3, 100.0),
    "ph": (3.5, 9.9),
    "rainfall": (20.2, 298.6),
}

LABELS_AFFICHAGE = {
    "N": "Azote (N, kg/ha)",
    "P": "Phosphore (P, kg/ha)",
    "K": "Potassium (K, kg/ha)",
    "temperature": "Température (°C)",
    "humidity": "Humidité relative (%)",
    "ph": "pH du sol",
    "rainfall": "Pluviométrie (mm)",
}

# Valeurs par défaut du curseur/champ (prises au milieu des plages d'entraînement)
VALEURS_DEFAUT = {f: round((lo + hi) / 2, 1) for f, (lo, hi) in PLAGES_ENTRAINEMENT.items()}


# ---------------------------------------------------------------------------
# Chargement des artefacts (mis en cache pour ne pas recharger à chaque interaction)
# ---------------------------------------------------------------------------

@st.cache_resource
def charger_artefacts():
    # Construction des chemins vers chaque fichier dans model_artifacts
    chemin = lambda nom: os.path.join(DOSSIER_ARTEFACTS, nom)

    # Vérification que le dossier existe
    if not os.path.exists(DOSSIER_ARTEFACTS):
        raise FileNotFoundError(
            f"Le dossier '{DOSSIER_ARTEFACTS}' est introuvable. "
            f"Assurez-vous que le dossier 'model_artifacts' est bien à côté de votre script."
        )

    modele = joblib.load(chemin("modele_final.joblib"))
    encodeur_cible = joblib.load(chemin("encodeur_cible.joblib"))
    noms_features = list(joblib.load(chemin("noms_features.joblib")))

    scaler = None
    chemin_scaler = chemin("scaler.joblib")
    if os.path.exists(chemin_scaler):
        scaler = joblib.load(chemin_scaler)

    return modele, scaler, encodeur_cible, noms_features


def predire(modele, scaler, encodeur_cible, features, valeurs: dict):
    vecteur = [float(valeurs[f]) for f in features]
    X = np.array(vecteur).reshape(1, -1)
    if scaler is not None:
        X = scaler.transform(X)

    prediction_encodee = modele.predict(X)[0]
    culture = encodeur_cible.inverse_transform([prediction_encodee])[0]

    avertissements = []
    for f, val in zip(features, vecteur):
        if f in PLAGES_ENTRAINEMENT:
            lo, hi = PLAGES_ENTRAINEMENT[f]
            if val < lo or val > hi:
                avertissements.append(
                    f"{LABELS_AFFICHAGE.get(f, f)} = {val} est en dehors de la plage "
                    f"observée à l'entraînement ({lo} - {hi}). La prédiction est une "
                    f"extrapolation et doit être interprétée avec prudence."
                )

    top3 = None
    if hasattr(modele, "predict_proba"):
        probas = modele.predict_proba(X)[0]
        indices_tries = np.argsort(probas)[::-1][:3]
        classes = encodeur_cible.inverse_transform(indices_tries)
        top3 = list(zip(classes, [probas[i] for i in indices_tries]))

    return culture, avertissements, top3


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Recommandation de culture",
    page_icon="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f33e.svg",
    layout="centered",
)

# Titre avec icône Material
sac.alert(
    label="Recommandation de culture agricole",
    description="Saisis les caractéristiques du sol et du climat de la parcelle.",
    color="green",
    icon="forest-green"
)

try:
    MODELE, SCALER, ENCODEUR_CIBLE, FEATURES = charger_artefacts()
except Exception as e:
    st.error(
        f"Impossible de charger les artefacts depuis `{os.path.abspath(DOSSIER_ARTEFACTS)}`.\n\n"
        f"Erreur : {e}\n\n"
        f"Vérifiez que les fichiers `modele_final.joblib`, `encodeur_cible.joblib` et "
        f"`noms_features.joblib` existent bien dans le dossier `model_artifacts` à côté de ce script."
    )
    st.stop()

with st.sidebar:
    sac.alert(label="Informations sur le modèle", color="gray", icon="info")
    st.write(f"**{len(ENCODEUR_CIBLE.classes_)} cultures possibles**")
    st.write(", ".join(sorted(ENCODEUR_CIBLE.classes_)))
    st.write(f"**Scaler appliqué :** {'Oui' if SCALER is not None else 'Non'}")

sac.alert(
    label="Caractéristiques de la parcelle",
    color="blue",
    icon="science"
)

col1, col2 = st.columns(2)
valeurs = {}
for i, f in enumerate(FEATURES):
    lo, hi = PLAGES_ENTRAINEMENT.get(f, (0.0, 500.0))
    colonne = col1 if i % 2 == 0 else col2
    valeurs[f] = colonne.number_input(
        LABELS_AFFICHAGE.get(f, f),
        value=float(VALEURS_DEFAUT.get(f, 0.0)),
        help=f"Plage observée à l'entraînement : {lo} - {hi}",
        format="%.2f",
    )

if st.button("Obtenir la recommandation", icon=":material/agriculture:", type="primary"):
    try:
        culture, avertissements, top3 = predire(MODELE, SCALER, ENCODEUR_CIBLE, FEATURES, valeurs)

        st.success(f"Culture recommandée : **{culture}**")

        for a in avertissements:
            st.warning(a)

        if top3:
            st.write("**Top 3 des cultures les plus probables :**")
            for nom_culture, proba in top3:
                st.write(f"- {nom_culture} : {proba * 100:.1f}%")
                st.progress(float(proba))

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

st.divider()
st.caption(
    "Cet outil constitue une aide à la décision et ne remplace pas l'expertise "
    "d'un agronome. Les recommandations reposent sur un modèle statistique "
    "entraîné sur un jeu de données historique."
)
