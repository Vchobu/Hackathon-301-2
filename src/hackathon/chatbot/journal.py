"""Journal des flux : écrit dans flux.log (racine du projet) tout ce qui se passe pendant un tour.

L'agent (ce fichier + collecte.py) et le serveur MCP (mcp_server.py) écrivent dans le MÊME fichier,
préfixés [agent] ou [serveur] : on lit ainsi le parcours complet d'un message dans l'ordre.

Pour suivre en direct, dans un 2e terminal à la racine du projet :
    Get-Content flux.log -Wait -Encoding UTF8

Niveaux :
    INFO  : tours, appels du LLM, étape choisie, outils, résultats
    DEBUG : en plus, le détail des fonctions internes de collecte.py (_apres_evaluation, profil_courant…)

Middlewares fournis (à mettre dans create_agent) :
    debut_tour     (before_agent)   : une fois par message du client
    journal_modele (after_model)    : ce que le LLM a répondu (texte ou appel d'outil)
    journal_outils (wrap_tool_call) : chaque appel d'outil MCP, sa durée et son résultat
    fin_tour       (after_agent)    : une fois par message, avec la réponse finale
"""

import json
import logging
import time
from pathlib import Path

from langchain.agents.middleware import after_agent, after_model, before_agent, wrap_tool_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

RACINE = Path(__file__).resolve().parents[3]          # src/hackathon/chatbot → racine du projet
FICHIER = RACINE / "flux.log"

log = logging.getLogger("flux")
compteur = {"tour": 0, "appel": 0}                    # n° du tour (message client) et de l'appel LLM


def configurer_journal(niveau: int = logging.DEBUG) -> None:
    """Branche le logger « flux » sur flux.log (une seule fois)."""
    if log.handlers:
        return
    h = logging.FileHandler(FICHIER, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d [agent]   %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.setLevel(niveau)
    log.propagate = False          # n'écrit pas dans le terminal / les logs de NiceGUI
    log.info("=" * 90)
    log.info("Démarrage de l'agent")


def court(texte, n: int = 160) -> str:
    """Texte sur une ligne, tronqué à n caractères."""
    t = " ".join(str(texte).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def texte(m) -> str:
    """Contenu texte d'un message (les résultats MCP arrivent sous forme de liste de blocs)."""
    if isinstance(m.content, str):
        return m.content
    return "".join(b.get("text", "") for b in m.content if isinstance(b, dict))


def resume(m) -> str:
    """Résumé d'un message en quelques mots (utilisé partout dans les logs)."""
    if isinstance(m, HumanMessage):
        return f"Human « {court(m.content, 60)} »"
    if isinstance(m, AIMessage):
        if m.tool_calls:
            return "AI → " + ", ".join(f"{c['name']}({court(json.dumps(c['args'], ensure_ascii=False), 70)})"
                                       for c in m.tool_calls)
        return f"AI « {court(m.content, 60)} »"
    if isinstance(m, ToolMessage):
        return f"Tool {m.name} [{m.status}] {court(texte(m), 70)}"
    return type(m).__name__


# ***********************************************************************************
# Middlewares de journal (ils ne modifient rien : ils écrivent seulement)
# ***********************************************************************************

@before_agent
def debut_tour(state, runtime) -> None:
    compteur["tour"] += 1
    compteur["appel"] = 0
    log.info("")
    log.info(f"═══ TOUR {compteur['tour']} ═══ {resume(state['messages'][-1])}   "
             f"(historique : {len(state['messages'])} messages)")
    return None


@after_model
def journal_modele(state, runtime) -> None:
    m = state["messages"][-1]
    if isinstance(m, AIMessage) and m.tool_calls:
        for c in m.tool_calls:
            log.info(f"   LLM a répondu → appel d'outil {c['name']}"
                     f"({court(json.dumps(c['args'], ensure_ascii=False), 300)})")
    else:
        log.info(f"   LLM a répondu → texte « {court(m.content, 200)} »  (pas d'outil : fin du tour)")
    return None


@wrap_tool_call
async def journal_outils(request, handler):
    appel = request.tool_call
    log.info(f"   ToolNode → MCP : {appel['name']}({court(json.dumps(appel['args'], ensure_ascii=False), 300)})")
    debut = time.perf_counter()
    resultat = await handler(request)
    duree = time.perf_counter() - debut
    if isinstance(resultat, ToolMessage):
        log.info(f"   ToolNode ← MCP : [{resultat.status}] en {duree:.2f} s : {court(texte(resultat), 300)}")
    else:
        log.info(f"   ToolNode ← MCP : {type(resultat).__name__} en {duree:.2f} s")
    return resultat


@after_agent
def fin_tour(state, runtime) -> None:
    m = state["messages"][-1]
    log.info(f"═══ FIN DU TOUR {compteur['tour']} ═══ {compteur['appel']} appel(s) du LLM · "
             f"historique : {len(state['messages'])} messages · réponse : « {court(m.content, 120)} »")
    return None
