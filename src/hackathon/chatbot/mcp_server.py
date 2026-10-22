import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faulthandler
import logging
import sys
from pathlib import Path
from typing import Literal

RACINE = Path(__file__).resolve().parents[3]   # src/hackathon/chatbot → racine du projet
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

# 1. Logging EN PREMIER : tout ce qui suit peut utiliser log, même en cas d'erreur
logging.basicConfig(
    filename=RACINE / "mcp_server.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("mcp")
log.info("serveur démarré")

# 2. Le .env : le serveur ne reçoit pas les variables de l'agent
from dotenv import load_dotenv
load_dotenv(RACINE / ".env")

import anyio
import requests
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, model_validator

import credit_tool
from credit_tool import outil_credit

_TRACE = open(RACINE / "mcp_traceback.log", "a", encoding="utf-8")

# 3. Tavily (après le .env, puisqu'il lit TAVILY_API_KEY)
TAVILY_URL = "https://api.tavily.com/search"
TAVILY_KEY = os.environ.get("TAVILY_API_KEY")
TIMEOUT_TAVILY = 10
TIMEOUT_SUGGESTIONS = 15

if TAVILY_KEY:
    try:
        requests.head("https://api.tavily.com", timeout=5)   # échauffement réseau avant mcp.run()
        log.info("réseau prêt, Tavily actif")
    except Exception as e:
        log.warning("échauffement réseau échoué (%s)", e)
else:
    log.warning("TAVILY_API_KEY absente du .env : suggestions désactivées")

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
    """Évalue une demande de crédit à partir du profil du client (après demande de toutes les questions utiles) et renvoie la décision du modèle.

    Règles :
        - Respecter la classe Profil

    En cas d'erreur de validation (valeur au mauvais format ou hors liste), corrige toi-même
    le champ indiqué dans le message d'erreur et rappelle l'outil, sans redemander et le relver au client.
        
    Sortie :
        - decision : "accord" ou "refus".
        - points_forts / points_faibles : éléments du profil en faveur / en défaveur du client,
          déjà formulés pour lui.
        - montant_max : uniquement en cas de refus. Plus grand montant accepté avec le même profil,
          à proposer au client. Vaut null si aucun montant ne passe : orienter vers un conseiller.
        - suggestions_texte : uniquement en cas de refus avec montant_max. Liste de liens au format
          Markdown. Si elle n'est pas vide, la recopier TELLE QUELLE sous le titre « Quelques idées »,
          après avoir proposé montant_max, sans la résumer, sans ajouter de prix ni de commentaire.
        - erreur : présent si l'évaluation est impossible. Demander au client les informations
          indiquées, puis rappeler l'outil.
    """

    arguments = profil.model_dump(exclude_none=True)   # objet Pydantic → dict, sans les champs vides
    log.info("evaluer_credit reçoit : %s", arguments)
    faulthandler.dump_traceback_later(200, file=_TRACE)   # si ça dure plus de 200 s : écrit où ça bloque
    try:
        resultat = outil_credit(arguments)

        enrichi = {**resultat, "suggestions": []}
        with anyio.move_on_after(TIMEOUT_SUGGESTIONS):
            enrichi = await anyio.to_thread.run_sync(
                suggestions, resultat, arguments,
                abandon_on_cancel=True,   # on n'attend pas la fin du thread abandonné
            )
        log.info("suggestions : %s", enrichi["suggestions"])
        resultat = enrichi


    except Exception:
        log.exception("evaluer_credit a planté")   # écrit l'erreur complète (traceback) dans le log
        raise                                      # l'erreur est quand même renvoyée au LLM
    finally:
        faulthandler.cancel_dump_traceback_later()        # fini à temps : on annule
    log.info("evaluer_credit renvoie : %s", resultat)



    return resultat

def suggestions(resultat: dict, arguments: dict) -> dict:
    """En cas de refus, enrichit le résultat avec 2-3 liens montrant ce que le client
    pourrait financer avec montant_max.

    resultat  : sortie de outil_credit (decision, montant_max…)
    arguments : profil envoyé à l'outil (on y prend l'objet du prêt, absent de resultat)
    Renvoie une COPIE de resultat avec "suggestions" et "suggestions_texte" (vides si rien à proposer).
    Ne lève jamais d'exception.
    """
    enrichi = {**resultat, "suggestions": [], "suggestions_texte": ""}

    # Uniquement en cas de refus avec un montant alternatif et un objet connu
    if resultat.get("decision") != "refus":
        return enrichi
    montant = resultat.get("montant_max")
    purpose = arguments.get("purpose")
    if not TAVILY_KEY or not purpose or not montant:       # montant_max = None → rien de finançable
        return enrichi

    objet = purpose.replace("_", " ")                      # "home_improvement" → "home improvement"
    budget = f"{int(montant):,}"                           # 72000.0 → "72,000"
    requete = f"buy a {objet} in Switzerland for under {budget} dollars"  
    try:
        reponse = requests.post(
            TAVILY_URL,
            json={"query": requete, "max_results": 5, "search_depth": "basic"},   # 5, pour en garder 3 après filtrage
            headers={"Authorization": f"Bearer {TAVILY_KEY}"},
            timeout=TIMEOUT_TAVILY,
        )
        reponse.raise_for_status()
        resultats = reponse.json().get("results", [])

        SCORE_MIN = 0.5   # pertinence minimale selon Tavily (0 à 1) : écarte les résultats hors sujet
        enrichi["suggestions"] = [
            {"titre": r.get("title") or r["url"], "url": r["url"]}
            for r in resultats
            if r.get("url") and r.get("score", 0) >= SCORE_MIN
        ][:3]
        # Texte prêt à afficher : le LLM n'a qu'à le recopier
        enrichi["suggestions_texte"] = "\n".join(
            f"- [{s['titre']}]({s['url']})" for s in enrichi["suggestions"])
    except Exception:
        log.exception("recherche Tavily échouée (requête : %s)", requete)
    return enrichi


if __name__ == "__main__":
    mcp.run()          # transport stdio par défaut