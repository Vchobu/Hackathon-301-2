"""MCP server exposing the credit model (LightGBM).

Run       : uv run python -m hackathon.chatbot.mcp_server
Test      : uv run mcp dev src/hackathon/chatbot/mcp_server.py

"""

import sys
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

RACINE = Path(__file__).resolve().parents[3]   # src/hackathon/chatbot → racine du projet
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from mcp.server.fastmcp import FastMCP
import credit_tool
from credit_tool import outil_credit
mcp = FastMCP("credit")

credit_tool._ART = credit_tool.charger_modele()

@mcp.tool()
async def evaluer_credit(profil: dict) -> str:
    """Evaluate a credit application and return the model's decision.

    Call this once all client information has been collected (or the client has said
    they do not know optional information).

    Input:
        profil: dictionary {technical_name: value} containing client information.
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

    resultat = outil_credit(profil)
    return json.dumps(resultat, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()          # transport stdio par défaut