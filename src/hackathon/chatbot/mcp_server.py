"""MCP server exposing the credit model (LightGBM).

Run       : uv run python -m hackathon.chatbot.mcp_server
Test      : uv run mcp dev src/hackathon/chatbot/mcp_server.py

"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

RACINE = Path(__file__).resolve().parents[3]   # src/hackathon/chatbot → racine du projet
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from mcp.server.fastmcp import FastMCP

import credit_tool
from credit_tool import outil_credit

logging.basicConfig(
    filename=RACINE / "mcp_server.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("mcp")

mcp = FastMCP("credit")

credit_tool._ART = credit_tool.charger_modele()
BANK_KNOWLEDGE_PATH = RACINE / "bank_knowledge.md"

class Profil(BaseModel):
    loan_amnt: float = Field(description="Requested loan amount in dollars. Required.")
    annual_inc: float = Field(description="Gross annual income in dollars. Required.")
    fico: float | None = Field(None, description="FICO credit score.")
    mensualites_dettes: float | None = Field(
        None, description="Current monthly debt payments in dollars."
    )
    inq_last_6mths: int | None = Field(
        None, description="Number of credit inquiries in the last six months."
    )
    revol_util: float | None = Field(
        None, description="Credit card utilization percentage."
    )
    mort_acc: int | None = Field(None, description="Number of mortgage accounts.")
    purpose: Literal[
        "car",
        "credit_card",
        "debt_consolidation",
        "educational",
        "home_improvement",
        "house",
        "major_purchase",
        "medical",
        "moving",
        "other",
        "renewable_energy",
        "small_business",
        "vacation",
        "wedding",
    ] | None = Field(None, description="Loan purpose.")
    home_ownership: Literal["RENT", "OWN", "MORTGAGE"] | None = Field(
        None, description="Housing status."
    )

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_fields(cls, data):
        if isinstance(data, dict):
            unknown = [key for key in data if key not in cls.model_fields]
            if unknown:
                raise ValueError(
                    f"Unknown fields: {unknown}. Use only these names: "
                    f"{list(cls.model_fields)}. Correct the names and call the tool again."
                )
        return data


def load_bank_sections() -> list[tuple[str, str]]:
    text = BANK_KNOWLEDGE_PATH.read_text(encoding="utf-8")
    sections = re.split(r"(?m)(?=^## )", text)
    return [
        (section.splitlines()[0].removeprefix("## ").strip(), section.strip())
        for section in sections
        if section.strip()
    ]


@mcp.tool()
def rechercher_information_banque(question: str) -> str:
    """Find relevant general bank information in the local knowledge document.

    Use this for questions about the bank, personal-loan conditions, fees,
    repayment, eligibility, or support. Do not use it for risk assessment.
    Do not infer or invent facts that are not present in the returned sections.
    """
    if not question.strip():
        return "The bank information question cannot be empty."

    query_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    scored_sections = []
    for title, content in load_bank_sections():
        section_terms = set(re.findall(r"[a-z0-9]+", f"{title} {content}".lower()))
        score = len(query_terms & section_terms)
        if score:
            scored_sections.append((score, title, content))

    if not scored_sections:
        return (
            "No relevant information was found in the bank knowledge document. "
            "Tell the client that a bank advisor must confirm the answer."
        )

    scored_sections.sort(key=lambda item: item[0], reverse=True)
    return "\n\n".join(section[2] for section in scored_sections[:2])


@mcp.tool()
async def evaluer_credit(profil: Profil) -> str:
    """Evaluate a credit application and return the model's decision.

    Call this once all client information has been collected (or the client has said
    they do not know optional information).

    Input:
        profil: structured profile containing the client's information.
            - Use exactly the technical names from your instructions.
            - The requested amount and income are required.
            - Include optional information only if the client provided it; never invent a value.
            - Numeric values must be numbers, not text such as "3000 $".

    Output (JSON string):
        - decision: internal value "accord" (approved) or "refus" (declined).
        - points_forts: profile factors in the client's favor, already formatted for the client.
        - points_faibles: profile factors against the client, already formatted for the client.
        - montant_max: only for a declined application, the largest amount accepted with the same profile.
        - erreur: only if assessment is impossible (for example, required information is missing).
    """

    arguments = profil.model_dump(exclude_none=True)
    log.info("evaluer_credit received: %s", arguments)
    try:
        resultat = outil_credit(arguments)
    except Exception:
        log.exception("evaluer_credit failed")
        raise
    log.info("evaluer_credit returned: %s", resultat)
    return json.dumps(resultat, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()          # transport stdio par défaut