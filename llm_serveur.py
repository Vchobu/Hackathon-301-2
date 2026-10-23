"""Choix du serveur LLM : Carnaval (gpu6) en principal, Ollama local en secours.

Carnaval est joint par un tunnel SSH ouvert à l'avance :
    ssh -N -L 11435:localhost:11434 gpu6
"""
import json
import urllib.request

# Serveurs essayés dans l'ordre : (nom, adresse Ollama, modèle)
SERVEURS = [
    ("Carnaval (gpu6)", "http://localhost:11435", "qwen3.5:9b"),
    ("Ollama local", "http://localhost:11434", "qwen3.5:4b"),
]


def choisir_serveur(delai: float = 3) -> tuple[str, str, str]:
    """Renvoie (nom, adresse, modèle) du premier serveur qui répond et qui a le modèle téléchargé."""
    for nom, adresse, modele in SERVEURS:
        try:
            with urllib.request.urlopen(f"{adresse}/api/tags", timeout=delai) as reponse:
                disponibles = [m["name"] for m in json.load(reponse)["models"]]
        except OSError:
            print(f"{nom} ne répond pas")
            continue
        if modele in disponibles:
            print(f"Serveur utilisé : {nom}, modèle : {modele}")
            return nom, adresse, modele
        print(f"{nom} répond, mais le modèle {modele} n'y est pas téléchargé")
    raise RuntimeError("Aucun serveur LLM disponible")


def client_openai():
    """Client OpenAI (interface compatible d'Ollama) et nom du modèle."""
    from openai import OpenAI
    _, adresse, modele = choisir_serveur()
    return OpenAI(base_url=f"{adresse}/v1", api_key="ollama", timeout=120, max_retries=0), modele


def modele_langchain():
    """Modèle LangChain, à passer à create_agent (agent.py)."""
    from langchain.chat_models import init_chat_model
    _, adresse, modele = choisir_serveur()
    return init_chat_model(f"ollama:{modele}", base_url=adresse, temperature=0.0, reasoning=False)


if __name__ == "__main__":
    choisir_serveur()
