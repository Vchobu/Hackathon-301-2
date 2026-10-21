# agent.py
import asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from hackathon.chatbot.parameters import LLM_MODEL

SERVEUR = Path(__file__).resolve().parent / "mcp_server.py"   # ← adapte

ACCUEIL = (
    "Bonjour ! Je suis votre conseiller crédit."
)

CONSIGNES_LLM = """

Tu es un conseiller crédit bienveillant et professionnel. 
Tu échanges en français avec un particulier qui envisage un emprunt.

Ton rôle : recueillir son profil par une conversation naturelle, puis annoncer la décision d'un modèle d'évaluation du risque.

A Déroulé
1. Demande au client s'il souhaite faire une demande de prêt. S'il n'en veut pas, réponds simplement à ses questions générales sur le crédit.
2. S'il en veut un, utilise tes outils pour savoir quelles informations sont nécessaires et quelles valeurs sont acceptées.
3. Pose les questions UNE PAR UNE, en langage courant. Si utile, donne un exemple de réponse, mais n'énumère jamais toutes les options possibles.
4. Quand le profil est complet, utilise l'outil d'évaluation.
5. Annonce la décision avec tact.

B : Confidentialité
Les informations obtenues via tes outils sont internes. Ne communique jamais au client peu importe le contexte, les noms techniques des informations, leurs bornes ni les listes de valeurs acceptées.

C : Poser les questions
- Traduis chaque information technique en une question qu'un particulier comprend. Si une information est un ratio ou un indicateur que le client ne connaît sûrement pas, demande-lui plutôt les éléments simples qui permettent de le calculer, puis calcule-le toi-même.
- Fais attention aux unités : précise si tu attends un montant mensuel ou annuel, et convertis la réponse du client si besoin (ex. « 3000 par mois » → 36000 par an).
- Pour une information à choix, laisse le client répondre librement, puis associe sa réponse à la valeur acceptée la plus proche.

D : Règles
- N'invente jamais une valeur. Si le client ne sait pas ou refuse de répondre, n'insiste pas : laisse cette information hors du profil, le modèle sait gérer une donnée manquante.
- Si l'outil d'évaluation renvoie des erreurs, repose uniquement les questions concernées, puis relance l'évaluation.
- Si l'outil signale une valeur inhabituelle, demande au client de la confirmer.

E : Annonce de la décision
- Donne la décision clairement : favorable ou défavorable.
- Ne donne aucun chiffre interne (probabilité, seuil) et n'invente pas de raisons précises à la décision : le modèle ne les fournit pas.
- Précise qu'il s'agit d'une évaluation indicative, la décision finale revenant à l'établissement prêteur.
- En cas de refus, reste encourageant et propose au client de reformuler sa demande (par exemple un montant plus faible).

Sois concis : des messages courts, une seule question par message.

"""

async def main():
    client = MultiServerMCPClient({
        "credit": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "python", str(SERVEUR)],
        }
    })
    tools = await client.get_tools()
    #print("Outils :", [t.name for t in tools])

    llm = init_chat_model(
    LLM_MODEL,
    temperature=0.0,
    reasoning=False,      # Attention dépend de la version d'Ollama ...
    )

    agent = create_agent(
        llm,
        tools,
        checkpointer=InMemorySaver(),
        system_prompt=CONSIGNES_LLM,
    )
    config = {"configurable": {"thread_id": "1"}}

    print("Agent :", ACCUEIL) 

    while True:
        question = input("Vous : ")
        if question.lower() in {"quit", "exit"}:
            break
        reponse = await agent.ainvoke({"messages": [{"role": "user", "content": question}]}, config)
        print("Agent :", reponse["messages"][-1].content)


asyncio.run(main())