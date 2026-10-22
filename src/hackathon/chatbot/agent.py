# agent.py

import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[3]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))



import asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from credit_tool import DESCRIPTIONS, OBLIGATOIRES

from hackathon.chatbot.parameters import LLM_MODEL

SERVEUR = Path(__file__).resolve().parent / "mcp_server.py"   # ← adapte

ACCUEIL = (
    "Bonjour ! Je suis votre conseiller crédit."
)

texte_descriptions = "\n".join(f"- {nom} : {desc}" for nom, desc in DESCRIPTIONS.items())
texte_obligatoires = ", ".join(OBLIGATOIRES)

# MODIF (section F du prompt) : ajout de la règle « n'envoie QUE ce que le client a dit ».
# Pourquoi : lors du test, l'agent a envoyé mort_acc = 0 sans jamais poser la question.
CONSIGNES_LLM = f"""Tu es un conseiller crédit bienveillant et professionnel.
Tu échanges en français avec un particulier qui envisage un emprunt.

A. Contexte
- Le service porte uniquement sur des prêts personnels d'une durée de 36 mois.
- Les autres durées et les autres types de crédit (immobilier, auto en leasing…).
- Les montants sont en dollars ($).

B. Déroulé
1. Demande au client s'il souhaite faire une demande de prêt. S'il n'en veut pas, réponds simplement à ses questions générales sur le crédit.
2. S'il en veut un, recueille les informations listées en {texte_descriptions}. Ces informations sont obligatoires : {texte_obligatoires}.
3. Pose les questions UNE PAR UNE, en langage courant. Si utile, donne un exemple de réponse, mais n'énumère jamais toutes les options possibles.
4. Quand le profil est complet (il a répondu à un maximum aux questions), appelle l'outil d'évaluation.
6. résume la décision de l'outil d'évaluation.
"""

print(CONSIGNES_LLM)  # pour debug, à supprimer dans la version finale

from contextlib import AsyncExitStack
from langchain_mcp_adapters.tools import load_mcp_tools

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
    session = await _PILE.enter_async_context(client.session("credit"))
    tools = await load_mcp_tools(session)
    for t in tools:
        t.handle_tool_error = True   # une erreur de l'outil est renvoyée au LLM comme message, sans planter
    llm = init_chat_model(LLM_MODEL, temperature=0.0, reasoning=False)
    return create_agent(llm, tools, checkpointer=InMemorySaver(), system_prompt=CONSIGNES_LLM)


async def main():
    """Mode terminal : uv run src\\hackathon\\chatbot\\agent.py"""
    agent = await creer_agent()
    config = {"configurable": {"thread_id": "1"}}
    print("Agent :", ACCUEIL)
    while True:
        question = input("Vous : ")
        if question.lower() in {"quit", "exit"}:
            break
        reponse = await agent.ainvoke({"messages": [{"role": "user", "content": question}]}, config)
        print("Agent :", reponse["messages"][-1].content)


if __name__ == "__main__":      # ← la boucle ne démarre plus quand GUI.py importe ce fichier
    asyncio.run(main())