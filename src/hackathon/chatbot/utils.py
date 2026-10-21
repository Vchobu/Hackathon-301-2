import math
from pathlib import Path

import joblib

# utils.py est dans src/hackathon/chatbot/ → racine du projet = 3 niveaux au-dessus
RACINE = Path(__file__).resolve().parents[3]

# Le joblib contient un dict : {'modele', 'seuil', 'features', 'categories'}
ARTEFACT = joblib.load(RACINE / "output" / "modele_credit.joblib")
MODEL = ARTEFACT["modele"]            # le LGBMClassifier
SEUIL = float(ARTEFACT["seuil"])
CATEGORIES = ARTEFACT["categories"]


def construire_schema() -> dict:
    """{feature: (type, valeurs)} lu dans le modèle. Numérique → [min, max], catégorielle → libellés."""
    info = MODEL.booster_.dump_model()
    libelles_iter = iter(info["pandas_categorical"])
    schema = {}
    for nom, fi in info["feature_infos"].items():
        if len(fi["values"]) > 0:
            libelles = next(libelles_iter)
            schema[nom] = ("catégorielle", [libelles[int(c)] for c in sorted(fi["values"])])
        else:
            schema[nom] = ("numérique", [float(fi["min_value"]), float(fi["max_value"])])
    return schema


SCHEMA = construire_schema()   # calculé une seule fois, au premier import


def verifier(nom: str, valeur):
    """Renvoie (valeur_nettoyée, erreur, avertissement). Une seule des deux premières est non nulle."""
    type_, valeurs = SCHEMA[nom]

    if type_ == "numérique":
        if isinstance(valeur, bool):                      # True/False n'est pas un nombre ici
            return None, f"{nom} : nombre attendu, reçu {valeur!r}", None
        try:
            x = float(valeur)                             # accepte 700, 700.0 ou "700"
        except (TypeError, ValueError):
            return None, f"{nom} : nombre attendu, reçu {valeur!r}", None
        if math.isnan(x):
            return None, f"{nom} : valeur vide", None
        mini, maxi = valeurs
        avert = None
        if not mini <= x <= maxi:
            avert = f"{nom} = {x} est hors de la plage connue du modèle [{mini}, {maxi}]"
        return x, None, avert

    # catégorielle : comparaison insensible à la casse ("rent" → "RENT")
    correspondance = {v.lower(): v for v in valeurs}
    cle = str(valeur).strip().lower()
    if cle not in correspondance:
        return None, f"{nom} : valeur {valeur!r} inconnue, attendu une parmi {valeurs}", None
    return correspondance[cle], None, None