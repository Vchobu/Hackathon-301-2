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
from typing import Literal
from pydantic import BaseModel, Field, model_validator

mcp = FastMCP("credit")


class Profil(BaseModel):
    loan_amnt: float = Field(description="Montant du prêt demandé, en dollars. Obligatoire.")
    annual_inc: float = Field(description="Revenu annuel brut, en dollars. Obligatoire.")
    fico: float | None = Field(None, description="Score de crédit FICO, entre 300 et 850.")
    mensualites_dettes: float | None = Field(None, description="Total des mensualités de dettes actuelles, en dollars PAR MOIS.")
    inq_last_6mths: int | None = Field(None, description="Nombre de demandes de crédit ces 6 derniers mois.")
    revol_util: float | None = Field(None, description="Utilisation des cartes de crédit, en % (20 pour 20 %).")
    mort_acc: int | None = Field(None, description="Nombre de prêts immobiliers.")
    purpose: Literal["car", "credit_card", "debt_consolidation", "educational", "home_improvement","house", "major_purchase", "medical", "moving", "other", "renewable_energy","small_business", "vacation", "wedding"] | None = Field(None, description="Objet du prêt.")
    home_ownership: Literal["RENT", "OWN", "MORTGAGE"] | None = Field(None, description="RENT = locataire, OWN = propriétaire sans crédit, MORTGAGE = propriétaire avec crédit immobilier.")


    # Par défaut, Pydantic IGNORE les champs inconnus (ex. "fico_score") : la valeur serait perdue
    # en silence. Ici on les REFUSE, avec un message qui donne au LLM les noms valides,
    # pour qu'il corrige et rappelle l'outil.
    @model_validator(mode="before")
    @classmethod
    def refuser_champs_inconnus(cls, data):
        if isinstance(data, dict):
            inconnus = [k for k in data if k not in cls.model_fields]
            if inconnus:
                raise ValueError(
                    f"Champs inconnus : {inconnus}. "
                    f"Utilise uniquement ces noms : {list(cls.model_fields)}. "
                    "Corrige les noms et rappelle l'outil."
                )
        return data
 

@mcp.tool()
def evaluer_credit(profil: Profil) -> dict:
    """Évalue une demande de crédit à partir du profil du client et renvoie la décision du modèle.

    Quand l'appeler : une seule fois, quand toutes les informations ont été recueillies auprès du client
    (ou que le client a indiqué ne pas connaître certaines informations facultatives).

    Règles :
        - Respecter la classe Profil
        
    En cas d'erreur de validation (valeur au mauvais format ou hors liste), corrige toi-même
    le champ indiqué dans le message d'erreur et rappelle l'outil, sans redemander au client.
        
    Sortie :
        - decision : "accord" ou "refus".
        - points_forts / points_faibles : éléments du profil en faveur / en défaveur du client,
          déjà formulés pour lui.
        - montant_max : uniquement en cas de refus. Plus grand montant accepté avec le même profil,
          à proposer au client. Vaut null si aucun montant ne passe : orienter vers un conseiller.
        - erreur : présent si l'évaluation est impossible. Demander au client les informations
          indiquées, puis rappeler l'outil.
    """

    arguments = profil.model_dump(exclude_none=True)   # objet Pydantic → dict, sans les champs vides
    print("[MCP] evaluer_credit reçoit :", arguments, file=sys.stderr)
    resultat = outil_credit(arguments)
    print("[MCP] evaluer_credit renvoie :", resultat, file=sys.stderr)


    return outil_credit(profil.model_dump(exclude_none=True))


if __name__ == "__main__":
    mcp.run()          # transport stdio par défaut