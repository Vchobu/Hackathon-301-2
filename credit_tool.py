"""Modèle de crédit Lending Club : fonctions utilisées par le notebook, l'API et le chatbot."""
import gzip
import io

import joblib
import pandas as pd
import polars as pl

DATA_PATH = "data/accepted_2007_to_2018Q4.csv.gz"
MODELE_PATH = "output/modele_credit.joblib"   # écrit par credit_tool_analysis.py (section 6)
DICO_PATH = "LCDataDictionary.xlsx"   # dictionnaire annoté (filtres 1 à 3, instructions chatbot)


def dictionnaire(path: str = DICO_PATH) -> pl.DataFrame:
    """chargement du CSV"""
    return (
        pl.read_excel(path, sheet_name="LoanStats")
        .rename({"LoanStatNew": "colonne"})
        .filter(pl.col("colonne").is_not_null())
        .with_columns(pl.col("colonne").str.strip_chars())
    )


def _ouvrir(path: str):
    """Ouvre le CSV en texte"""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def colonnes_disponibles(path: str = DATA_PATH) -> list[str]:
    """Titres de toutes les colonnes du CSV"""
    with _ouvrir(path) as f:
        return pl.read_csv(io.StringIO(f.readline()), n_rows=0).columns

# ************************************************************************
# fonctions pour le chatbot
# definitions:
# demande (dict) : les réponses du client
#  e.g. {"loan_amnt": 25000, "annual_inc": 45000, "mensualites_dettes": 900, "fico": 675,
#    "mort_acc": 0, "inq_last_6mths": 2, "revol_util": 80,
#    "purpose": "debt_consolidation", "home_ownership": "RENT"}

# loan_amnt et annual_inc sont obligatoires

def charger_modele(path: str = MODELE_PATH) -> dict:
    """contient Modèle LightGBM, seuil, liste des features et catégories"""
    return joblib.load(path)


def vers_features(demande: dict, art: dict) -> pd.DataFrame:
    """Input : réponses du client (dict) et artefact du modèle.
    Output : DataFrame d'une ligne avec les 9 features du modèle (valeur absente = NaN).
    Utilisation : appelée par score_loan ; à appeler directement seulement pour vérifier les features."""
    revenu = max(float(demande["annual_inc"]), 1)
    mensualites = demande.get("mensualites_dettes")

    # Les réponses reprises telles quelles (fico, mort_acc, purpose...), puis les features calculées
    ligne = {f: demande.get(f) for f in art["features"]}
    ligne["annual_inc"] = revenu
    ligne["loan_to_income"] = demande["loan_amnt"] / revenu
    ligne["dti"] = None if mensualites is None else 100 * mensualites / (revenu / 12)

    X = pd.DataFrame([ligne])
    for c, cats in art["categories"].items():          # mêmes catégories qu'à l'entraînement
        X[c] = pd.Categorical(X[c], categories=cats)
    num = [f for f in art["features"] if f not in art["categories"]]
    X[num] = X[num].astype(float)
    return X


def score_loan(demande: dict, art: dict, k: int = 3) -> dict:
    """Décision (accord / refus), points forts et points faibles du client (SHAP, k maximum chacun).
    Ni probabilité, ni seuil : le client ne doit pas pouvoir viser le seuil.
    Une valeur manquante n'est jamais citée. Dépendance : vers_features."""
    X = vers_features(demande, art)
    p = art["modele"].predict_proba(X)[0, 1]

    # Contribution SHAP de chaque feature (sans la dernière colonne = biais), valeurs connues seulement
    contrib = pd.Series(art["modele"].predict(X, pred_contrib=True)[0][:-1], index=art["features"])
    contrib = contrib[X.iloc[0].notna()]

    return {"decision": "refus" if p >= art["seuil"] else "accord",
            "points_forts": contrib[contrib < 0].sort_values().index[:k].tolist(),                  # réduisent le risque
            "points_faibles": contrib[contrib > 0].sort_values(ascending=False).index[:k].tolist()}  # l'augmentent


def montant_max(demande: dict, art: dict, pas: int = 500, minimum: int = 1000) -> dict:
    """En cas de refus : plus grand montant (par pas de 500 $) que le modèle accepterait, ou None."""
    for montant in range(int(demande["loan_amnt"]) - pas, minimum - 1, -pas):
        if score_loan({**demande, "loan_amnt": montant}, art)["decision"] == "accord":
            return {"montant_max": montant}
    return {"montant_max": None}
