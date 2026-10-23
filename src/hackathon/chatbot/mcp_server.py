"""Serveur MCP du conseiller crédit.

Outils :
  - noter_reponse  : enregistre UNE réponse du client (validée avec les règles de Profil).
  - evaluer_credit : évalue le profil complet avec le modèle LightGBM (+ suggestions web en cas de refus).

Le serveur est SANS ÉTAT : il ne garde pas le profil en mémoire. C'est l'agent qui reconstruit le
profil à partir des résultats de noter_reponse (voir collecte.py). Ainsi, plusieurs conversations
(plusieurs onglets de la GUI) peuvent partager le même serveur sans se mélanger.

Lancement : uv run python -m hackathon.chatbot.mcp_server
Test MCP  : uv run mcp dev src/hackathon/chatbot/mcp_server.py

⚠️ Transport stdio : stdout sert au protocole MCP → messages via logging (fichier mcp_server.log).
⚠️ Windows : tout ce qui charge des DLL (modèle, scipy, SSL/réseau) doit être fait AVANT mcp.run().
"""

import os

# Un seul thread pour les bibliothèques de calcul (évite les blocages OpenMP sous Windows).
# Doit être défini AVANT l'import de numpy / scipy / lightgbm.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import faulthandler
import logging
import re
import time
import sys
from pathlib import Path
from typing import Annotated, Literal, Union, get_args

RACINE = Path(__file__).resolve().parents[3]   # src/hackathon/chatbot → racine du projet
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))            # credit_tool.py est à la racine : importable APRÈS cette ligne

# 1. Logging en premier : tout ce qui suit peut écrire dans le log, même en cas d'erreur
logging.basicConfig(
    filename=RACINE / "mcp_server.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("mcp")
# Les mêmes messages vont aussi dans flux.log (partagé avec l'agent), préfixés [serveur]
_flux = logging.FileHandler(RACINE / "flux.log", encoding="utf-8")
_flux.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d [serveur] %(message)s", "%H:%M:%S"))
_flux.addFilter(lambda r: r.name == "mcp")      # seulement nos messages (pas « Processing request… »)
logging.getLogger().addHandler(_flux)
log.info("serveur démarré")

# 2. Le .env : le serveur ne reçoit pas les variables d'environnement de l'agent
from dotenv import load_dotenv
load_dotenv(RACINE / ".env")

import anyio
import requests
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

import credit_tool
from credit_tool import OBLIGATOIRES, outil_credit

_TRACE = open(RACINE / "mcp_traceback.log", "a", encoding="utf-8")

# 3. Tavily (après le .env, puisqu'il lit TAVILY_API_KEY)
TAVILY_URL = "https://api.tavily.com/search"
TAVILY_KEY = os.environ.get("TAVILY_API_KEY")
TIMEOUT_TAVILY = 10       # secondes max pour la requête HTTP
TIMEOUT_SUGGESTIONS = 15  # secondes max avant de renoncer aux suggestions

if TAVILY_KEY:
    try:
        requests.head("https://api.tavily.com", timeout=5)   # échauffement réseau avant mcp.run()
        log.info("réseau prêt, Tavily actif")
    except Exception as e:
        log.warning("échauffement réseau échoué (%s)", e)
else:
    log.warning("TAVILY_API_KEY absente du .env : suggestions désactivées")

mcp = FastMCP("credit")


# ***********************************************************************************
# Profil du client (schéma envoyé au LLM + règles de validation)
# ***********************************************************************************

Facultatif = Literal["fico", "mensualites_dettes", "inq_last_6mths", "revol_util",
                     "mort_acc", "purpose", "home_ownership"]
Champ = Literal["loan_amnt", "annual_inc", "fico", "mensualites_dettes", "inq_last_6mths",
                "revol_util", "mort_acc", "purpose", "home_ownership"]


class Profil(BaseModel):
    loan_amnt: float = Field(gt=0, description="Montant du prêt demandé, en dollars. Obligatoire.")
    annual_inc: float = Field(gt=0, description="Revenu annuel brut, en dollars. Obligatoire.")
    fico: float | None = Field(None, ge=300, le=850, description="Score de crédit FICO, entre 300 et 850.")
    mensualites_dettes: float | None = Field(
        None, ge=0, description="Total des mensualités de dettes actuelles, en dollars PAR MOIS "
                                "(hors loyer et hors crédit immobilier). 0 si aucune dette.")
    inq_last_6mths: int | None = Field(
        None, ge=0, description="Nombre de demandes de crédit faites ces 6 derniers mois.")
    revol_util: float | None = Field(
        None, ge=0, description="Utilisation des cartes de crédit, en % du plafond (20 pour 20 %). "
                                "Peut dépasser 100.")
    mort_acc: int | None = Field(None, ge=0, description="Nombre de prêts immobiliers (en cours ou remboursés).")
    purpose: Literal["car", "credit_card", "debt_consolidation", "educational", "home_improvement",
                     "house", "major_purchase", "medical", "moving", "other", "renewable_energy",
                     "small_business", "vacation", "wedding"] | None = Field(
        None, description="Objet du prêt (ex. voiture → car, regrouper des dettes → debt_consolidation).")
    home_ownership: Literal["RENT", "OWN", "MORTGAGE"] | None = Field(
        None, description="Logement : RENT = locataire, OWN = propriétaire sans crédit, "
                          "MORTGAGE = propriétaire avec crédit immobilier.")
    inconnus: list[Facultatif] = Field(
        default_factory=list,
        description="Informations que le client a dit ne pas connaître ou a refusé de donner. "
                    "Chaque information facultative doit être soit renseignée, soit listée ici.")

    # ⚠️ Les validateurs "before" s'exécutent dans l'ordre INVERSE de leur écriture.
    # Ordre réel : noms inventés → obligatoires → (types) → questions oubliées.

    @model_validator(mode="before")
    @classmethod
    def verifier_obligatoires(cls, data):
        """Champs obligatoires absents (ou null) : message clair avec leur description."""
        if isinstance(data, dict):
            manquants = [c for c in OBLIGATOIRES if data.get(c) is None]
            if not manquants:
                log.info("   Profil (validation 2/3) : champs obligatoires présents → OK")
            if manquants:
                log.info("   Profil (validation 2/3) : obligatoires manquants %s → refus", manquants)
                champs = cls.model_fields
                liste = "\n".join(f"- {c} : {champs[c].description}" for c in manquants)
                raise ValueError(
                    f"Informations OBLIGATOIRES manquantes :\n{liste}\n"
                    "Demande-les au client (sans elles, aucune évaluation n'est possible), "
                    "puis rappelle l'outil.")
        return data

    @model_validator(mode="before")
    @classmethod
    def refuser_champs_inconnus(cls, data):
        """Noms de champs inventés (ex. "fico_score") : refusés, avec la liste des noms valides."""
        if isinstance(data, dict):
            inconnus = [k for k in data if k not in cls.model_fields]
            log.info("   Profil (validation 1/3) : noms de champs %s", f"inconnus {inconnus} → refus" if inconnus else "OK")
            if inconnus:
                raise ValueError(
                    f"Champs inconnus : {inconnus}. "
                    f"Utilise uniquement ces noms : {list(cls.model_fields)}. "
                    "Corrige les noms et rappelle l'outil.")
        return data

    @model_validator(mode="after")
    def verifier_questions_posees(self):
        """Chaque information facultative : renseignée OU déclarée dans "inconnus"."""
        champs = type(self).model_fields
        oublies = [c for c in get_args(Facultatif)
                   if getattr(self, c) is None and c not in self.inconnus]
        log.info("   Profil (validation 3/3, après les types) : %s",
                 f"questions oubliées {oublies} → refus" if oublies else "toutes les questions ont une réponse → OK")
        if oublies:
            liste = "\n".join(f"- {c} : {champs[c].description}" for c in oublies)
            raise ValueError(
                f"Questions pas encore posées :\n{liste}\n"
                "Pose-les au client, une par une, puis rappelle l'outil. "
                "Si le client ne sait pas ou refuse de répondre, mets le nom du champ dans 'inconnus'.")
        return self


def _validateur(champ: str) -> TypeAdapter:
    """Validateur d'UN champ de Profil (même type et mêmes bornes, sans le « | None »)."""
    f = Profil.model_fields[champ]
    types = [t for t in get_args(f.annotation) if t is not type(None)] or [f.annotation]
    t = types[0] if len(types) == 1 else Union[tuple(types)]
    return TypeAdapter(Annotated[(t, *f.metadata)] if f.metadata else t)


def _nettoyer_nombre(v):
    """'10 000 $' → '10000', '3,5' → '3.5', '73,459' → '73459'. Laisse le reste tel quel."""
    if not isinstance(v, str):
        return v
    s = re.sub(r"[\s$€%_  ]", "", v)
    if re.fullmatch(r"-?\d+,\d{1,2}", s):     # virgule décimale
        return s.replace(",", ".")
    return s.replace(",", "")                 # séparateur de milliers


# ***********************************************************************************
# Chargements AVANT mcp.run() (modèle + échauffement : charge toutes les DLL)
# ***********************************************************************************

credit_tool._ART = credit_tool.charger_modele()
credit_tool.score_loan({"loan_amnt": 10000, "annual_inc": 50000}, credit_tool._ART)
log.info("modèle chargé")


# ***********************************************************************************
# Outil 1 : enregistrer une réponse du client
# ***********************************************************************************

@mcp.tool()
def noter_reponse(champ: Champ, valeur: float | str | None = None, inconnu: bool = False,
                  par_mois: bool = False) -> dict:
    """Enregistre la réponse du client à UNE information. À appeler dès que le client répond.

    - champ : nom technique de l'information (ex. "annual_inc").
    - valeur : recopie EXACTEMENT le nombre ou le mot donné par le client, sans le convertir
      ni le recalculer (ex. client « 100000 » → 100000 ; « aucune » → 0).
    - par_mois : true seulement si le client a donné un REVENU MENSUEL (l'outil le convertit en annuel).
    - inconnu : true si le client ne connaît pas l'information ou refuse de la donner (valeur vide).
    Si la valeur est invalide, l'erreur explique pourquoi : corrige-la ou redemande-la au client.
    """
    log.info("noter_reponse(champ=%s, valeur=%r, inconnu=%s, par_mois=%s)", champ, valeur, inconnu, par_mois)
    if inconnu:
        if champ in OBLIGATOIRES:
            log.info("   %s est obligatoire : « inconnu » refusé", champ)
            raise ValueError(f"{champ} est obligatoire : explique au client qu'elle est indispensable "
                             "pour évaluer sa demande, puis redemande-la.")
        log.info("   → renvoie {champ: %s, inconnu: true}", champ)
        return {"champ": champ, "inconnu": True}

    if valeur is None or valeur == "":
        log.info("   aucune valeur → refus")
        raise ValueError(f"Aucune valeur pour {champ}. Si le client ne sait pas, "
                         "rappelle noter_reponse avec inconnu=true.")

    description = Profil.model_fields[champ].description
    if champ not in ("purpose", "home_ownership"):    # nettoyage des nombres seulement : sinon
        valeur = _nettoyer_nombre(valeur)            # "home_improvement" perdrait son « _ »
        log.info("   nettoyage du nombre → %r", valeur)
    if par_mois and champ == "annual_inc":           # conversion faite par le code, pas par le LLM
        try:
            valeur = float(valeur) * 12
            log.info("   revenu mensuel × 12 → %r", valeur)
        except (TypeError, ValueError):
            pass                                     # la validation ci-dessous renverra l'erreur
    try:
        propre = _validateur(champ).validate_python(valeur)
    except ValidationError as e:
        log.info("   validation refusée : %s", e.errors()[0]["msg"])
        raise ValueError(f"Valeur invalide pour {champ} ({description}) : {e.errors()[0]['msg']}. "
                         "Corrige-la, ou redemande-la au client.") from None

    log.info("   validé → renvoie {champ: %s, valeur: %r}", champ, propre)
    return {"champ": champ, "valeur": propre}


# ***********************************************************************************
# Règles métier : profils hors de ce que le modèle connaît
# ***********************************************************************************

# Plages vues à l'entraînement. Au-delà, un arbre de décision répond comme à la valeur limite :
# un prêt de 10 fois le revenu serait traité comme un prêt de 71 % du revenu → « accord » absurde.
RATIO_MAX = 0.7125     # montant / revenu annuel maximal vu à l'entraînement
FICO_MIN = 662         # score FICO minimal vu à l'entraînement


def evaluer(arguments: dict) -> dict:
    """outil_credit + règles pour les profils hors de ce que le modèle a appris."""
    debut = time.perf_counter()
    resultat = outil_credit(arguments)
    log.info("   modèle (outil_credit) en %.2f s → décision %s, montant_max %s",
             time.perf_counter() - debut, resultat.get("decision"), resultat.get("montant_max", "-"))
    if "decision" not in resultat:                    # erreur d'évaluation : on ne touche à rien
        return resultat

    alertes = []
    revenu = arguments["annual_inc"]
    ratio = arguments["loan_amnt"] / revenu
    log.info("   règle montant/revenu : %.3f %s %.4f", ratio, ">" if ratio > RATIO_MAX else "≤", RATIO_MAX)
    if ratio > RATIO_MAX:
        # Refus imposé : le modèle n'a jamais vu de prêt aussi élevé par rapport au revenu.
        # Le montant maximal est cherché à partir du plafond (et plus du montant demandé : plus rapide).
        plafond = int(RATIO_MAX * revenu) // 500 * 500
        libelle = credit_tool.LIBELLES.get("loan_to_income", "le montant demandé par rapport à votre revenu")
        resultat["decision"] = "refus"
        resultat["points_forts"] = [p for p in resultat.get("points_forts", []) if p != libelle]
        resultat["points_faibles"] = [libelle] + [p for p in resultat.get("points_faibles", []) if p != libelle]
        resultat |= credit_tool.montant_max({**arguments, "loan_amnt": plafond + 500}, credit_tool._ART)
        log.info("   → refus imposé, montant_max cherché sous le plafond %s → %s", plafond, resultat.get("montant_max"))
        alertes.append("montant supérieur à 71 % du revenu annuel : au-delà des dossiers connus du modèle")
    log.info("   règle FICO : %s", arguments.get("fico", "inconnu"))
    if arguments.get("fico") is not None and arguments["fico"] < FICO_MIN:
        alertes.append("score de crédit inférieur aux profils connus du modèle : évaluation moins fiable")
    if alertes:
        resultat["alertes"] = alertes
        log.info("alertes : %s", alertes)
    return resultat


# ***********************************************************************************
# Outil 2 : évaluer le profil complet
# ***********************************************************************************

@mcp.tool()
async def evaluer_credit(profil: Profil) -> dict:
    """Évalue une demande de crédit à partir du profil COMPLET du client et renvoie la décision du modèle.

    Règles :
        - Utiliser exactement les noms de champs du schéma, aucun autre.
        - Chaque information facultative doit avoir été demandée au client. S'il ne la connaît pas
          ou refuse de la donner, mettre son nom dans "inconnus" au lieu d'inventer une valeur.

    En cas d'erreur de validation, corrige toi-même le champ indiqué dans le message d'erreur
    et rappelle l'outil, sans redemander au client et sans lui signaler l'erreur.

    Sortie :
        - decision : "accord" ou "refus".
        - points_forts / points_faibles : éléments du profil en faveur / en défaveur du client,
          déjà formulés pour lui.
        - montant_max : uniquement en cas de refus. Plus grand montant accepté avec le même profil,
          à proposer au client. Vaut null si aucun montant ne passe : orienter vers un conseiller.
        - suggestions_texte : liste de liens au format Markdown. Si elle n'est pas vide, la recopier
          TELLE QUELLE sous le titre « Quelques idées », après avoir proposé montant_max.
        - alertes : présent si le profil sort de ce que le modèle connaît (montant très élevé par
          rapport au revenu, score de crédit bas). Le dire simplement au client : la pré-évaluation
          est alors moins fiable.
        - erreur : présent si l'évaluation est impossible. Demander au client les informations
          indiquées, puis rappeler l'outil.
    """
    arguments = profil.model_dump(exclude={"inconnus"}, exclude_none=True)
    log.info("evaluer_credit : profil validé (3 validations OK) → %s", arguments)
    faulthandler.dump_traceback_later(200, file=_TRACE)   # si ça dure plus de 200 s : écrit où ça bloque
    try:
        # Calcul dans un thread : le serveur reste disponible pendant ce temps
        resultat = await anyio.to_thread.run_sync(evaluer, arguments)

        # Suggestions web (uniquement en cas de refus) ; au-delà de 15 s, on répond sans liens
        enrichi = {**resultat, "suggestions": [], "suggestions_texte": ""}
        with anyio.move_on_after(TIMEOUT_SUGGESTIONS):
            enrichi = await anyio.to_thread.run_sync(
                suggestions, resultat, arguments, abandon_on_cancel=True)
        resultat = enrichi
    except Exception:
        log.exception("evaluer_credit a planté")   # traceback complet dans le log
        raise                                      # l'erreur est quand même renvoyée au LLM
    finally:
        faulthandler.cancel_dump_traceback_later()
    log.info("evaluer_credit renvoie : %s", resultat)
    return resultat


def suggestions(resultat: dict, arguments: dict) -> dict:
    """En cas de refus, enrichit le résultat avec 2-3 liens montrant ce que le client
    pourrait financer avec montant_max. Renvoie une COPIE de resultat. Ne lève jamais d'exception.
    """
    enrichi = {**resultat, "suggestions": [], "suggestions_texte": ""}

    if resultat.get("decision") != "refus":
        log.info("   suggestions : pas de refus → pas de recherche")
        return enrichi
    montant = resultat.get("montant_max")
    purpose = arguments.get("purpose")
    if not TAVILY_KEY or not purpose or not montant:       # montant_max = None → rien de finançable
        log.info("   suggestions : pas de recherche (clé Tavily %s, objet %s, montant_max %s)",
                 "présente" if TAVILY_KEY else "absente", purpose, montant)
        return enrichi

    objet = purpose.replace("_", " ")                      # "home_improvement" → "home improvement"
    budget = f"{int(montant):,}"                           # 72000.0 → "72,000"
    requete = f"buy a {objet} in Switzerland for under {budget} dollars"
    log.info("   suggestions : recherche Tavily « %s »", requete)
    try:
        reponse = requests.post(
            TAVILY_URL,
            json={"query": requete, "max_results": 5, "search_depth": "basic"},
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
        log.info("   suggestions : %d résultat(s) reçu(s), %d gardé(s) (score ≥ %.1f)",
                 len(resultats), len(enrichi["suggestions"]), SCORE_MIN)
        enrichi["suggestions_texte"] = "\n".join(
            f"- [{s['titre']}]({s['url']})" for s in enrichi["suggestions"])
    except Exception:
        log.exception("recherche Tavily échouée (requête : %s)", requete)
    return enrichi


if __name__ == "__main__":
    mcp.run()          # transport stdio par défaut
