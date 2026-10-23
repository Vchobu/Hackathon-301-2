"""Agent conseiller crédit : LLM (Ollama) + outils MCP (noter_reponse, evaluer_credit).

Mode terminal : uv run src\\hackathon\\chatbot\\agent.py
Mode web      : uv run GUI.py   (GUI.py importe creer_agent et ACCUEIL)

Le prompt n'est plus fixe : collecte.py le reconstruit avant chaque appel du LLM
(une question à la fois, puis l'appel à evaluer_credit, puis l'annonce de la décision).
"""

from pathlib import Path

import asyncio
from contextlib import AsyncExitStack

from dotenv import load_dotenv
load_dotenv()   # LangSmith

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver

from hackathon.chatbot.collecte import configurer, historique_minimal, prompt_selon_etape
from hackathon.chatbot.journal import (configurer_journal, debut_tour, fin_tour, journal_modele,
                                       journal_outils, log)
from hackathon.chatbot.parameters import LLM_MODEL


SERVEUR = Path(__file__).resolve().parent / "mcp_server.py"

ACCUEIL = "Bonjour ! Je suis votre conseiller crédit. Quel montant souhaitez-vous emprunter ?"

_PILE = AsyncExitStack()   # garde la session MCP ouverte tant que le programme tourne


async def creer_agent():
    """Crée l'agent (LLM + outils MCP). Utilisé par le terminal et par GUI.py."""
    configurer_journal()             # flux.log à la racine (niveau DEBUG : tout le détail)
    log.info(f"Lancement du serveur MCP : uv run python {SERVEUR.name}")
    client = MultiServerMCPClient({
        "credit": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "python", str(SERVEUR)],
        }
    })
    session = await _PILE.enter_async_context(client.session("credit"))   # serveur lancé une seule fois
    tools = await load_mcp_tools(session)
    log.info(f"Session MCP ouverte · outils disponibles : {[t.name for t in tools]}")
    for t in tools:
        t.handle_tool_error = True   # une erreur de l'outil est renvoyée au LLM comme message, sans planter

    configurer(tools)                # lit noms, descriptions et obligatoires dans le schéma de evaluer_credit

    middleware = [
        debut_tour,             # journal : début du tour (message du client)
        prompt_selon_etape,     # choisit le prompt système de l'étape   ┐ l'ordre compte :
        historique_minimal,     # choisit les messages et les outils      ┘ prompt d'abord
        journal_modele,         # journal : ce que le LLM a répondu
        journal_outils,         # journal : chaque appel d'outil MCP et son résultat
        fin_tour,               # journal : fin du tour (réponse finale)
    ]
    log.info(f"Middlewares : {[m.name if hasattr(m, 'name') else type(m).__name__ for m in middleware]}")
    

    llm = init_chat_model(LLM_MODEL, temperature=0.0, reasoning=False)
    log.info(f"LLM : {LLM_MODEL} · agent prêt")
    return create_agent(llm, tools, checkpointer=InMemorySaver(), middleware=middleware)


async def main():
    """Mode terminal."""
    agent = await creer_agent()
    config = {"configurable": {"thread_id": "1"}}
    print("Agent :", ACCUEIL)
    while True:
        question = input("Vous : ")
        if question.lower() in {"quit", "exit"}:
            break
        reponse = await agent.ainvoke({"messages": [{"role": "user", "content": question}]}, config)
        print("Agent :", reponse["messages"][-1].content)


if __name__ == "__main__":      # la boucle ne démarre pas quand GUI.py importe ce fichier
    asyncio.run(main())
