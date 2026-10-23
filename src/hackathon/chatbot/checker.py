"""Vérificateur de réponses de l'agent crédit (middleware LangChain).

Après chaque réponse finale de l'agent, un second LLM vérifie qu'elle :
  1. reste dans le cadre du crédit / prêt / emprunt,
  2. répond au dernier message du client,
  3. annonce la même décision que l'outil evaluer_credit,
  4. ne donne ni chiffre interne ni chiffre inventé.
Si ce n'est pas le cas, le problème est renvoyé à l'agent, qui réécrit sa réponse
(au plus MAX_ESSAIS fois, puis réponse de secours).

Utilisation dans agent.py :
    from hackathon.checker import verifier_reponse
    create_agent(..., middleware=[..., verifier_reponse])
"""

from typing import Literal, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import after_model
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from hackathon.chatbot.parameters import LLM_MODEL

MAX_ESSAIS = 2   # nombre de réécritures autorisées avant la réponse de secours


class Verdict(BaseModel):
    """Jugement du vérificateur sur la réponse de l'agent."""
    conforme: bool = Field(description="True si la réponse respecte TOUS les critères")
    probleme: Literal["aucun", "hors_sujet", "ne_repond_pas", "decision_differente",
                      "chiffre_interne_ou_invente"] = Field(description="Le problème principal")
    explication: str = Field(description="Une phrase expliquant le jugement")


CONSIGNES_VERIFICATEUR = """Tu contrôles la réponse d'un conseiller crédit avant qu'elle soit envoyée au client.

Critères (la réponse est conforme seulement si TOUS sont respectés) :
1. Elle reste dans le cadre du crédit, du prêt ou de l'emprunt.
2. Elle répond au dernier message du client.
3. Si un résultat d'évaluation est fourni : elle annonce la MÊME décision (accord ou refus) et le même
   montant alternatif éventuel sur la base des mêmes critères.
4. Elle ne donne ni probabilité, ni score, ni seuil du modèle, et n'invente aucun chiffre absent
   du résultat ou de la conversation."""

REPONSE_DE_SECOURS = ("Je suis votre conseiller crédit : je peux vous aider à évaluer une demande de prêt "
                      "ou répondre à vos questions sur l'emprunt. Comment puis-je vous aider ?")

# Le LLM vérificateur (même modèle que l'agent), avec une sortie structurée (Verdict)
verificateur = init_chat_model(LLM_MODEL, temperature=0.0, reasoning=False).with_structured_output(Verdict)


class EtatVerif(AgentState):
    """État de l'agent + compteur de réécritures demandées par le vérificateur."""
    essais_verif: NotRequired[int]


@after_model(state_schema=EtatVerif, can_jump_to=["model"])
async def verifier_reponse(state: EtatVerif, runtime: Runtime) -> dict | None:
    """Contrôle chaque réponse finale. Si elle n'est pas conforme, renvoie le problème à l'agent pour qu'il réécrive."""
    messages = state["messages"]
    reponse = messages[-1]

    # On ne contrôle que les réponses finales (pas les appels d'outils)
    if not isinstance(reponse, AIMessage) or reponse.tool_calls or not reponse.content:
        return None

    question = next((m.content for m in reversed(messages)
                     if isinstance(m, HumanMessage) and m.name != "verificateur"), "")
    evaluation = next((m.content for m in reversed(messages)
                       if isinstance(m, ToolMessage) and m.name == "evaluer_credit"), "aucune")

    verdict = await verificateur.ainvoke([
        {"role": "system", "content": CONSIGNES_VERIFICATEUR},
        {"role": "user", "content": f"Dernier message du client :\n{question}\n\n"
                                    f"Résultat de l'évaluation :\n{evaluation}\n\n"
                                    f"Réponse du conseiller à contrôler :\n{reponse.content}"},
    ])
    essais = state.get("essais_verif", 0)
    print(f"[VÉRIF essai {essais}] {verdict.probleme} : {verdict.explication}")

    # Messages de contrôle à effacer une fois le tour terminé (invisibles pour le client)
    nettoyage = [RemoveMessage(id=m.id) for m in messages
                 if m.name in ("verificateur", "brouillon_rejete")]

    # 1. Conforme : on nettoie et on laisse partir la réponse
    if verdict.conforme:
        return {"messages": nettoyage, "essais_verif": 0}

    # 2. Trop d'essais : on abandonne et on remplace par la réponse de secours
    if essais >= MAX_ESSAIS:
        return {"messages": [*nettoyage, AIMessage(content=REPONSE_DE_SECOURS, id=reponse.id)],
                "essais_verif": 0}

    # 3. Non conforme : on garde la réponse comme brouillon, on explique le problème, retour au LLM
    brouillon = AIMessage(content=reponse.content, id=reponse.id, name="brouillon_rejete")  # même id → remplace
    retour = HumanMessage(
        name="verificateur",
        content=(f"[Contrôle qualité, invisible pour le client] Ta réponse précédente n'est pas conforme "
                 f"({verdict.probleme}) : {verdict.explication} "
                 "Réécris ta réponse au client en corrigeant ce point, sans mentionner ce contrôle."),
    )
    return {"messages": [brouillon, retour], "essais_verif": essais + 1, "jump_to": "model"}