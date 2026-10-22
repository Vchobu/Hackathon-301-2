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
CONSIGNES_LLM = f"""
Tu es un conseiller crédit et uniquement ça.
Tu échanges en français avec un particulier qui envisage un emprunt et tu es neutre, concis et tu ne te répètes pas.

A. Contexte
- Inutile de faire des salutations
- Le service porte uniquement sur des prêts d'une durée de 36 mois.
- Les autres durées sont hors sujets.
- Les montants sont en dollars ($).

B. Déroulé
1. Demande au client s'il souhaite faire une demande de prêt.
2. S'il en veut un, recueille toutes les informations listées en {texte_descriptions}. 
3. Pose les questions UNE PAR UNE, en langage courant et de manière concis. 
4. Si utile pour une question, donne un seul exemple de réponse, mais n'énumère jamais toutes les options possibles.
5. Ces informations sont obligatoires : {texte_obligatoires}, les autres sont facultatives.
6. Quand le profil est complet (il a répondu à toutes les questions), appelle l'outil d'évaluation et pas avant.
7. résume la décision de l'outil d'évaluation.
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