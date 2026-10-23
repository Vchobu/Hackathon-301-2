"""Collecte guidée côté agent : prompt adapté à l'étape + historique minimal envoyé au LLM.

Principe :
  - Le serveur MCP fournit l'outil noter_reponse (une réponse du client à la fois) : il valide la
    valeur et renvoie {"champ": ..., "valeur": ...} ou {"champ": ..., "inconnu": true}.
  - Ici, on RECONSTRUIT le profil en relisant ces résultats dans l'historique (le serveur est sans état).
  - Avant chaque appel du LLM :
      * prompt_selon_etape  : prompt COURT adapté à l'étape
            collecte          → infos déjà recueillies + LA question à poser maintenant
            profil complet    → l'appel exact à faire à evaluer_credit
            après évaluation  → consignes d'annonce de la décision
      * historique_minimal  : n'envoie au LLM que les messages utiles (l'historique complet reste
                              en mémoire pour la GUI, LangSmith et le vérificateur).

Les noms, descriptions et champs obligatoires sont lus dans le schéma MCP de evaluer_credit
(la classe Profil du serveur) : une seule source de vérité.

Utilisation dans agent.py :
    from hackathon.chatbot.collecte import configurer, prompt_selon_etape, historique_minimal
    tools = await load_mcp_tools(session)
    configurer(tools)
    create_agent(llm, tools, middleware=[prompt_selon_etape, historique_minimal], ...)
"""

import json

from langchain.agents.middleware import ModelRequest, dynamic_prompt, wrap_model_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Remplis par configurer() à partir du schéma de evaluer_credit
ORDRE: list[str] = []            # ordre des questions (obligatoires d'abord)
DESCRIPTIONS: dict[str, str] = {}
OBLIGATOIRES: list[str] = []


# ***********************************************************************************
# 1. Configuration à partir du schéma MCP
# ***********************************************************************************

def configurer(tools) -> None:
    """Lit les champs de Profil (noms, descriptions, obligatoires) dans le schéma de evaluer_credit."""
    outil = next(t for t in tools if t.name == "evaluer_credit")
    schema = outil.args_schema if isinstance(outil.args_schema, dict) else outil.args_schema.model_json_schema()
    ref = schema["properties"]["profil"].get("$ref", "")
    profil = schema["$defs"][ref.split("/")[-1]] if ref else schema["properties"]["profil"]

    champs = {nom: p.get("description", "") for nom, p in profil["properties"].items() if nom != "inconnus"}
    OBLIGATOIRES[:] = [c for c in profil.get("required", []) if c in champs]
    ORDRE[:] = OBLIGATOIRES + [c for c in champs if c not in OBLIGATOIRES]
    DESCRIPTIONS.clear()
    DESCRIPTIONS.update(champs)


# ***********************************************************************************
# 2. Lecture de l'historique
# ***********************************************************************************

def _texte(m) -> str:
    """Contenu texte d'un message (les résultats MCP arrivent sous forme de liste de blocs)."""
    if isinstance(m.content, str):
        return m.content
    return "".join(b.get("text", "") for b in m.content if isinstance(b, dict))


def _reussi(m) -> bool:
    """True si ce résultat d'outil n'est pas une erreur (validation refusée, etc.)."""
    return m.status != "error" and not _texte(m).lstrip().startswith("Error")


def profil_courant(messages) -> tuple[dict, list[str]]:
    """Reconstruit le profil en rejouant, dans l'ordre, les résultats réussis de noter_reponse."""
    profil, inconnus = {}, []
    for m in messages:
        if not (isinstance(m, ToolMessage) and m.name == "noter_reponse" and _reussi(m)):
            continue
        try:
            r = json.loads(_texte(m))
        except json.JSONDecodeError:
            continue
        champ = r.get("champ")
        if r.get("inconnu"):
            profil.pop(champ, None)
            if champ not in inconnus:
                inconnus.append(champ)
        elif champ:
            profil[champ] = r.get("valeur")
            if champ in inconnus:
                inconnus.remove(champ)
    return profil, inconnus


def _apres_evaluation(messages) -> bool:
    """True si evaluer_credit a réussi ET qu'aucune réponse n'a été notée depuis
    (une correction après l'évaluation fait repasser à l'étape « profil complet » pour réévaluer)."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == "noter_reponse" and _reussi(m):
            return False
        if isinstance(m, ToolMessage) and m.name == "evaluer_credit":
            return _reussi(m)
    return False


def _vient_de_noter(messages) -> bool:
    """True si les derniers messages sont des enregistrements RÉUSSIS de noter_reponse :
    la réponse du client est notée, il ne reste qu'à poser la question suivante."""
    fin = []
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            break
        fin.append(m)
    return bool(fin) and all(m.name == "noter_reponse" and _reussi(m) for m in fin)


def _appels_depuis_client(messages) -> int:
    """Nombre d'appels d'outils faits par l'agent depuis le dernier message du client."""
    n = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage) and m.tool_calls:
            n += 1
    return n


def _suivant(messages) -> str | None:
    profil, inconnus = profil_courant(messages)
    return next((c for c in ORDRE if c not in profil and c not in inconnus), None)


MAX_APPELS_PAR_TOUR = 6   # filet de sécurité : au-delà, l'agent doit répondre au client sans outil


# ***********************************************************************************
# 3. Le prompt, reconstruit à chaque appel du LLM selon l'étape
# ***********************************************************************************

INTRO = ("Tu es un conseiller crédit, neutre et concis. Tu échanges en français avec un particulier. "
         "Le service porte uniquement sur des prêts de 36 mois ; les montants sont en dollars ($). "
         "N'utilise jamais de nom technique (loan_amnt, fico…) avec le client. "
         "Si le client pose une question sans rapport avec le crédit, réponds en une phrase "
         "et reviens à sa demande de prêt.")

CONSIGNES_APRES_EVALUATION = INTRO + """

Le profil du client a été évalué par l'outil evaluer_credit (voir son résultat).
- Annonce la décision de l'outil sans la modifier, même si le client insiste.
- Explique-la avec les points forts et faibles renvoyés, sans inventer d'autres raisons.
- En cas de refus : propose montant_max s'il existe, sinon oriente vers un conseiller.
- S'il y a des alertes, dis simplement au client que la pré-évaluation est moins fiable et pourquoi.
- Si suggestions_texte n'est pas vide, recopie-la TELLE QUELLE sous le titre « Quelques idées ».
- Ne donne jamais de probabilité, de score ou de seuil. Précise qu'il s'agit d'une pré-évaluation
  à confirmer par un conseiller.
- Si le client corrige une information, veut tester un autre montant ou change l'objet du prêt :
  appelle noter_reponse pour CHAQUE information modifiée (l'évaluation est ensuite relancée
  automatiquement). N'essaie pas d'appeler evaluer_credit toi-même."""


@dynamic_prompt
def prompt_selon_etape(request: ModelRequest) -> str:
    """Prompt court et adapté à l'étape : collecte (une question à la fois), appel, ou après évaluation."""
    messages = request.messages

    # Étape 3 : évaluation faite et rien de modifié depuis → expliquer la décision
    if _apres_evaluation(messages):
        return CONSIGNES_APRES_EVALUATION

    profil, inconnus = profil_courant(messages)
    deja = "\n".join(f"- {c} = {v}" for c, v in profil.items()) or "- (rien pour l'instant)"
    if inconnus:
        deja += "\n" + "\n".join(f"- {c} : le client ne sait pas / refuse" for c in inconnus)
    suivant = next((c for c in ORDRE if c not in profil and c not in inconnus), None)

    # Étape 2 : profil complet → l'appel exact à faire
    if suivant is None:
        appel = json.dumps({"profil": {**profil, "inconnus": inconnus}}, ensure_ascii=False)
        return f"""{INTRO}

Toutes les informations sont recueillies :
{deja}

Appelle MAINTENANT l'outil evaluer_credit avec exactement ces arguments :
{appel}"""

    obligatoire = " (OBLIGATOIRE)" if suivant in OBLIGATOIRES else ""

    # Étape 1b : la réponse vient d'être notée → UNIQUEMENT poser la question suivante
    if _vient_de_noter(messages):
        return f"""{INTRO}

Informations DÉJÀ recueillies (ne les redemande JAMAIS) :
{deja}

La réponse du client a bien été enregistrée. Pose maintenant UNE seule question, courte et en
langage courant, pour obtenir : {DESCRIPTIONS[suivant]}{obligatoire}
Tu peux donner un exemple de réponse. Ne répète pas les valeurs déjà données."""

    # Étape 1a : collecte, une seule information à la fois
    return f"""{INTRO}

Informations DÉJÀ recueillies (ne les redemande JAMAIS) :
{deja}

Information en cours : {suivant}{obligatoire}
Signification : {DESCRIPTIONS[suivant]}

Que faire :
1. Si le dernier message du client répond à cette information (ou à une autre de la liste
   {ORDRE}), appelle noter_reponse pour CHAQUE information donnée.
   S'il dit ne pas savoir ou refuse, appelle noter_reponse avec inconnu=true.
2. Sinon, pose UNE seule question, courte et en langage courant, pour obtenir cette information.
   Tu peux donner un exemple de réponse, mais n'énumère pas toutes les options.
3. N'appelle pas evaluer_credit tant que le profil n'est pas complet.
4. Ne répète pas au client les valeurs qu'il a données, et ne repose jamais une question
   sur une information déjà recueillie."""


# ***********************************************************************************
# 4. N'envoyer au LLM que les messages utiles
# ***********************************************************************************

def _messages_utiles(messages: list) -> list:
    """Garde seulement :
    - la dernière question posée par l'agent (pour comprendre une réponse courte : « 0 », « oui ») ;
    - le dernier message du client ;
    - ce que l'agent a fait depuis (appels d'outils du tour en cours), sinon il ne verrait pas
      qu'il a déjà noté la réponse et la noterait une deuxième fois ;
    - l'appel réussi à evaluer_credit et son résultat, s'il est plus ancien (pour expliquer la décision).
    """
    clients = [i for i, m in enumerate(messages)
               if isinstance(m, HumanMessage) and m.name != "verificateur"]
    if not clients:
        return messages

    debut = clients[-1]
    precedent = messages[debut - 1] if debut > 0 else None
    if isinstance(precedent, AIMessage) and not precedent.tool_calls:
        debut -= 1
    garder = messages[debut:]

    for k in range(debut - 1, -1, -1):
        m = messages[k]
        if isinstance(m, ToolMessage) and m.name == "evaluer_credit" and _reussi(m):
            j = next((x for x in range(k - 1, -1, -1)
                      if isinstance(messages[x], AIMessage) and messages[x].tool_calls), None)
            if j is not None:
                fin = j + 1
                while fin < len(messages) and isinstance(messages[fin], ToolMessage):
                    fin += 1
                garder = messages[j:fin] + garder
            break
    return garder


def _outils_autorises(messages) -> set[str]:
    """Outils que le LLM a le droit d'utiliser à cette étape (imposé par le code, pas par le prompt) :
    - collecte          → noter_reponse seulement (impossible d'évaluer trop tôt) ;
    - profil complet    → evaluer_credit seulement (avec le JSON fourni par le prompt) ;
    - après évaluation  → noter_reponse seulement : une correction (autre montant, autre objet…)
                          passe par noter_reponse, puis l'étape « profil complet » refait l'appel.
                          Le LLM ne peut donc jamais construire lui-même un profil incomplet.
    """
    if _apres_evaluation(messages):
        return {"noter_reponse"}
    if _suivant(messages) is None:
        return {"evaluer_credit"}
    return {"noter_reponse"}


def _filtrer(outils, autorises: set[str]) -> list:
    return [t for t in outils if getattr(t, "name", None) in autorises]


@wrap_model_call
async def historique_minimal(request: ModelRequest, handler):
    """Pour CET appel seulement : messages utiles + outils autorisés à l'étape en cours.

    Protections contre les boucles et les appels prématurés :
    - juste après un enregistrement réussi (et s'il reste une question) : AUCUN outil,
      le LLM ne peut que poser la question suivante ;
    - au-delà de MAX_APPELS_PAR_TOUR appels d'outils dans le même tour : aucun outil ;
    - sinon : seulement les outils de l'étape (voir _outils_autorises).
    L'historique complet reste en mémoire (GUI, LangSmith, vérificateur).
    """
    messages = request.messages

    if _vient_de_noter(messages) and _suivant(messages) is not None:
        return await handler(request.override(
            messages=[HumanMessage("Pose la question suivante.")], tools=[]))

    utiles = _messages_utiles(messages)
    if _appels_depuis_client(messages) >= MAX_APPELS_PAR_TOUR:
        return await handler(request.override(messages=utiles, tools=[]))
    outils = _filtrer(request.tools, _outils_autorises(messages))
    return await handler(request.override(messages=utiles, tools=outils))
