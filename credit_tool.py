

import gzip  # lit le CSV compressé
import io  # aide à lire l’en-tête

import joblib  # charge le modèle
import pandas as pd  # LightGBM veut du pandas
import polars as pl  # lire et traiter les données

# définition à connaître.
# demande (dict) : les réponses du client, envoyées par le chatbot
#   e.g. {"loan_amnt": 25000, ...}
#   Les champs facultatifs absents sont traités comme inconnus.

# art (dict) : l'artefact sauvegardé par le notebook, chargé par charger_modele()
#   il contient le modèle, le seuil, la liste des features et les catégories apprises :
#   e.g. {"modele": LGBMClassifier,   "seuil": 0.5, "features": ["fico", "annual_inc", ...], "categories": {...}}


DATA_PATH = "data/accepted_2007_to_2018Q4.csv.gz"
MODELE_PATH = "output/modele_credit.joblib"   # écrit par credit_tool_analysis.py (section 6)
DICO_PATH = "LCDataDictionary.xlsx"   # dictionnaire annoté (filtres 1 à 3, instructions chatbot)


# ***********************************************************************************
# 1. CSV tools: lecture data
# ***********************************************************************************

def dictionnaire(path: str = DICO_PATH) -> pl.DataFrame:
    """Dictionnaire annoté (onglet LoanStats de l'Excel) : une ligne par colonne du CSV."""
    return (
        pl.read_excel(path, sheet_name="LoanStats")
        .rename({"LoanStatNew": "colonne"})                  # nom plus clair
        .filter(pl.col("colonne").is_not_null())             # retire les lignes vides en bas de la feuille
        .with_columns(pl.col("colonne").str.strip_chars())   # retire les espaces autour des noms
    )

def colonnes_disponibles(path: str = DATA_PATH) -> list[str]:
    """Noms de toutes les colonnes du CSV"""
    with gzip.open(path, "rt") as f:
        return f.readline().strip().split(",")

# ***********************************************************************************
# 2. fonctions pour le chatbot
# ***********************************************************************************

# workflow du chatbot :
# charger_modele     ->     vers_features      ->     score_loan           ->    montant_max
# (lire le fichier)        (réponses -> features)     (décision + SHAP)          (contre-proposition)
# P.S.: loan_amnt et annual_inc sont obligatoires

def charger_modele(path: str = MODELE_PATH) -> dict:
    """contient Modèle LightGBM, seuil, liste des features et catégories"""
    return joblib.load(path)

def vers_features(demande: dict, art: dict) -> pd.DataFrame:
    """ Input : réponses du client (dict) et artefact du modèle.
       Output : DataFrame d'une ligne avec les 9 features du modèle (valeur absente = NaN)."""

    # step 1: Préparer le revenu : max(..., 1) évite une division par zéro si le client tape 0.
    revenu = max(float(demande["annual_inc"]), 1)
    mensualites = demande.get("mensualites_dettes")

    # step 2: Les 9 features : 7 reprises telles quelles, 2 calculées
    ligne = {
        "fico": demande.get("fico"),
        "annual_inc": revenu,
        "dti": None if mensualites is None else 100 * mensualites / (revenu / 12),  # calculée
        "loan_to_income": demande["loan_amnt"] / revenu,                               # calculée
        "mort_acc": demande.get("mort_acc"),
        "inq_last_6mths": demande.get("inq_last_6mths"),
        "revol_util": demande.get("revol_util"),
        "purpose": demande.get("purpose"),
        "home_ownership": demande.get("home_ownership"),
    }

    # step 3: Mettre au format du modèle (type et encodage)
    X = pd.DataFrame([ligne])
    # a. text : LightGBM ne lit pas le texte, il lit un numéro par catégorie: MORTGAGE 0 	NONE	1 OTHER	2 OWN	3 RENT 4
    for c, cats in art["categories"].items():
        X[c] = pd.Categorical(X[c], categories=cats)
    # b. numerical: convertit toutes en nombres décimaux
    num = [f for f in art["features"] if f not in art["categories"]]
    X[num] = X[num].astype(float)
    return X


def score_loan(demande: dict, art: dict, k: int = 3) -> dict:
    """Décision (accord / refus),
    points forts et points faibles du client (SHAP, k maximum chacun). Une valeur manquante n'est jamais citée.
    on ne donne pas la probabilité et le seuil (info confidentiels)
    Dépendance : vers_features."""

    # 1. Réponses du client -> ligne de features
    X = vers_features(demande, art)

    # 2. Probabilité de défaut, puis décision
    proba = art["modele"].predict_proba(X)[0, 1]  # [0, 1] = 1er client, colonne « défaut »
    decision = "refus" if proba >= art["seuil"] else "accord"

    # 3. Contribution SHAP de chaque feature : > 0 augmente le risque, < 0 le réduit
    shap = art["modele"].predict(X, pred_contrib=True)[0][:-1]  # dernière valeur = biais, retirée
    contrib = pd.Series(shap, index=art["features"])
    contrib = contrib[X.iloc[0].notna()]  # on ne cite que les valeurs connues

    # 4. Les k contributions les plus fortes de chaque côté
    points_forts = contrib[contrib < 0].sort_values().index[:k].tolist()
    points_faibles = contrib[contrib > 0].sort_values(ascending=False).index[:k].tolist()

    return {"decision": decision, "points_forts": points_forts, "points_faibles": points_faibles}


def montant_max(demande: dict, art: dict, pas: int = 500, minimum: int = 1000) -> dict:
    """En cas de refus : on test des montants d'emprunt plus faible
     (par pas de 500 $) que le modèle accepterait
     ou None si aucun montant ne passe avec les valeurs du client"""
    montant = demande["loan_amnt"] - pas                 # premier essai : 500 $ de moins

    while montant >= minimum:
        demande_reduite = {**demande, "loan_amnt": montant}   # reprend la demande avec un montant plus petit
        if score_loan(demande_reduite, art)["decision"] == "accord":
            return {"montant_max": montant}              # si ça passe on retourne ce montant
        montant -= pas                                   # sinon, on continue la boucle (-500 $)

    return {"montant_max": None}                         # aucun montant accepté

