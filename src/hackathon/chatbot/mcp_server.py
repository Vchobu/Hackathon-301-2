"""Serveur MCP exposant le modèle de crédit (LightGBM).

Lancement : uv run python -m hackathon.chatbot.mcp_server
Test      : uv run mcp dev src/hackathon/chatbot/mcp_server.py

"""

import pandas as pd
from mcp.server.fastmcp import FastMCP

from hackathon.chatbot.utils import CATEGORIES, MODEL, SCHEMA, SEUIL, verifier

mcp = FastMCP("credit")


@mcp.tool()
def get_model_features() -> dict:
    """Renvoie les features attendues par le modèle de crédit, avec leur type et leurs valeurs possibles.

    Pour une feature numérique : [min, max] observés à l'entraînement.
    Pour une feature catégorielle : la liste des valeurs acceptées.
    """
    return SCHEMA


@mcp.tool()
def evaluer_credit(profil: dict) -> dict:
    """Évalue si une demande de crédit peut être accordée à partir du profil de l'emprunteur.

    Appeler d'abord get_model_features pour connaître les features attendues.
    Les features inconnues ou de mauvais type sont ignorées et signalées dans la réponse.
    Les features absentes sont traitées comme manquantes par le modèle.

    Args:
        profil: dictionnaire {nom_de_feature: valeur} décrivant l'emprunteur.
    """
    erreurs, avertissements = [], []
    ligne = {nom: float("nan") for nom in MODEL.feature_name_}   # tout « manquant » au départ

    for nom, valeur in profil.items():
        if nom not in SCHEMA:
            erreurs.append(f"{nom} : feature inconnue du modèle, ignorée")
            continue
        propre, err, avert = verifier(nom, valeur)
        if err:
            erreurs.append(err + ", ignorée")
            continue
        if avert:
            avertissements.append(avert)
        ligne[nom] = propre

    manquantes = [n for n in MODEL.feature_name_ if pd.isna(ligne[n])]
    if len(manquantes) == len(MODEL.feature_name_):
        return {"erreur": "Aucune feature valide fournie, prédiction impossible.", "erreurs": erreurs}

    try:
        client = pd.DataFrame([ligne])[MODEL.feature_name_]
        for col, cats in CATEGORIES.items():
            client[col] = pd.Categorical(client[col], categories=cats)
        proba = float(MODEL.predict_proba(client)[0, 1])
    except Exception as e:                                     # filet de sécurité final
        return {"erreur": f"Échec de la prédiction : {e}", "erreurs": erreurs}

    return {
        "accorde": bool(proba < SEUIL),    # bool() : un np.bool_ ne passe pas en JSON
                                           # à inverser si la classe 1 = « remboursé »
        "proba_defaut": round(proba, 3),
        "seuil": round(SEUIL, 3),
        "features_manquantes": manquantes,
        "erreurs": erreurs,
        "avertissements": avertissements,
    }


if __name__ == "__main__":
    mcp.run()          # transport stdio par défaut