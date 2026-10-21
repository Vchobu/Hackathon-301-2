"""Serveur MCP exposant le modèle de crédit (LightGBM).

Lancement : uv run python -m hackathon.chatbot.mcp_server
Test      : uv run mcp dev src/hackathon/chatbot/mcp_server.py

"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]   # src/hackathon/chatbot → racine du projet
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from mcp.server.fastmcp import FastMCP
from credit_tool import outil_credit
from hackathon.chatbot.parameters import LLM_MODEL
import pandas as pd
from mcp.server.fastmcp import FastMCP

from credit_tool import outil_credit

mcp = FastMCP("credit")


@mcp.tool()
def evaluer_credit(profil: dict) -> dict:
    """Évalue une demande de crédit à partir du profil du client et renvoie la décision du modèle.

    Quand l'appeler : une seule fois que toutes les informations ont été recueillies auprès du client
    (ou que le client a indiqué ne pas connaître certaines informations facultatives).

    Entrée :
        arguments : dictionnaire {nom_technique: valeur} avec les informations du client.
            - Utiliser exactement les noms techniques donnés dans tes consignes.
            - Le montant demandé et le revenu sont obligatoires.
            - N'inclure une information facultative que si le client l'a donnée : ne jamais inventer de valeur.
            - Les valeurs numériques sont des nombres (pas de texte comme "3000 $").

    Sortie :
        - decision : "accord" ou "refus".
        - points_forts : éléments du profil qui jouent en faveur du client, déjà formulés pour lui.
        - points_faibles : éléments du profil qui jouent en sa défaveur, déjà formulés pour lui.
        - montant_max : uniquement en cas de refus, plus grand montant qui serait accepté
          avec le même profil. À proposer au client comme alternative.
        - erreur : présent uniquement si l'évaluation est impossible (ex. information obligatoire
          manquante). Dans ce cas, demander au client les informations indiquées puis rappeler l'outil.
    """

    return outil_credit(profil)


if __name__ == "__main__":
    mcp.run()          # transport stdio par défaut