"""Agent conseiller crédit : LLM (Ollama) + outils MCP (noter_reponse, evaluer_credit).

Mode terminal : uv run src\\hackathon\\chatbot\\agent.py
Mode web      : uv run GUI.py   (GUI.py importe creer_agent et ACCUEIL)

Le prompt n'est plus fixe : collecte.py le reconstruit avant chaque appel du LLM
(une question à la fois, puis l'appel à evaluer_credit, puis l'annonce de la décision).
"""

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

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
from hackathon.chatbot.parameters import LLM_MODEL

VERIFICATEUR = False   # True : un second LLM contrôle chaque réponse (checker.py) — plus lent

SERVEUR = Path(__file__).resolve().parent / "mcp_server.py"

ACCUEIL = "Bonjour ! Je suis votre conseiller crédit. Quel montant souhaitez-vous emprunter ?"

_PILE = AsyncExitStack()   # garde la session MCP ouverte tant que le programme tourne


async def creer_agent():
    """Crée l'agent (LLM + outils MCP). Utilisé par le terminal et par GUI.py."""
    client = MultiServerMCPClient({
        "credit": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "python", str(SERVEUR)],
        }
    })
    session = await _PILE.enter_async_context(client.session("credit"))   # serveur lancé une seule fois
    tools = await load_mcp_tools(session)
    for t in tools:
        t.handle_tool_error = True   # une erreur de l'outil est renvoyée au LLM comme message, sans planter

    configurer(tools)                # lit noms, descriptions et obligatoires dans le schéma de evaluer_credit

    middleware = [prompt_selon_etape, historique_minimal]   # l'ordre compte : prompt d'abord
    if VERIFICATEUR:
        from hackathon.chatbot.checker import verifier_reponse
        middleware.append(verifier_reponse)

    llm = init_chat_model(LLM_MODEL, temperature=0.0, reasoning=False)
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
