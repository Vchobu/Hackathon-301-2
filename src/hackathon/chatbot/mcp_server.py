"""Serveur MCP exposant le modèle de crédit (LightGBM).

Lancement : uv run python -m hackathon.chatbot.mcp_server
Test      : uv run mcp dev src/hackathon/chatbot/mcp_server.py

"""
import anyio
import sys
from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

RACINE = Path(__file__).resolve().parents[3]   # src/hackathon/chatbot → racine du projet
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from mcp.server.fastmcp import FastMCP
from credit_tool import outil_credit
from hackathon.chatbot.parameters import LLM_MODEL
import pandas as pd
from mcp.server.fastmcp import FastMCP
import credit_tool
from credit_tool import outil_credit, charger_modele, _ART
from typing import Literal
from pydantic import BaseModel, Field, model_validator
import json
import faulthandler

_TRACE = open(RACINE / "mcp_traceback.log", "w", encoding="utf-8")


import logging

logging.basicConfig(
    filename=RACINE / "mcp_server.log",          # à la racine du projet
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("mcp")
log.info("serveur démarré")

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

credit_tool._ART = credit_tool.charger_modele()

@mcp.tool()
async def evaluer_credit(profil: Profil) -> dict:
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
    log.info("evaluer_credit reçoit : %s", arguments)
    faulthandler.dump_traceback_later(200, file=_TRACE)   # si ça dure plus de 200 s : écrit où ça bloque
    try:
        resultat = outil_credit(arguments)
    except Exception:
        log.exception("evaluer_credit a planté")   # écrit l'erreur complète (traceback) dans le log
        raise                                      # l'erreur est quand même renvoyée au LLM
    finally:
        faulthandler.cancel_dump_traceback_later()        # fini à temps : on annule
    log.info("evaluer_credit renvoie : %s", resultat)
    return resultat


if __name__ == "__main__":
    mcp.run()          # transport stdio par défaut