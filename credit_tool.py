

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
    art = joblib.load(path)
    modele = art["modele"]
    if hasattr(modele, "set_params"):
        modele.set_params(n_jobs=1)
    return art

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
    premier_montant = demande["loan_amnt"] - pas
    if premier_montant < minimum:
        return {"montant_max": None}

    nombre_candidats = (premier_montant - minimum) // pas + 1
    gauche, droite = 0, nombre_candidats - 1
    meilleur = None

    # The risk increases with the loan amount through loan_to_income, so the
    # accepted amounts form a lower range of the candidate amounts.
    while gauche <= droite:
        milieu = (gauche + droite) // 2
        montant = premier_montant - milieu * pas
        X = vers_features({**demande, "loan_amnt": montant}, art)
        probabilite = art["modele"].predict_proba(X)[0, 1]

        if probabilite < art["seuil"]:
            meilleur = montant
            droite = milieu - 1
        else:
            gauche = milieu + 1

    return {"montant_max": meilleur}                     # aucun montant accepté si None





# ***********************************************************************************
# mode d’emploi de l'outil
# ***********************************************************************************

# noms des features -> mots compréhensibles par le client
FEATURES = {
    "fico": {
        "libelle": "votre historique de crédit",
        "description": "score de crédit FICO (échelle 300 à 850, plus c'est haut, meilleur est l'historique). "
                       "Le modèle ne connaît que les scores à partir de ~660. "
                       "Si le client ne le connaît pas, ne pas l'estimer.",
    },
    "annual_inc": {
        "libelle": "votre revenu",
        "description": "revenu annuel brut, en dollars. Si le client donne un revenu mensuel, multiplier par 12.",
    },
    # MODIF : la clé reste le nom du MODÈLE ("loan_to_income"), car score_loan renvoie ce nom dans
    # points_forts / points_faibles et LIBELLES doit le connaître (sinon KeyError).
    # "champ" = nom que le LLM envoie ; c'est vers_features qui calcule le ratio, pas le LLM
    # (un LLM se trompe facilement dans une division, Python jamais).
    # Avant : clé "loan_to_amnt" (faute de frappe) + description de ratio -> le LLM envoyait 0.5.
    "loan_to_income": {
        "champ": "loan_amnt",
        "libelle": "le montant demandé par rapport à votre revenu",
        "description": "montant du prêt demandé, en dollars (ex. 15000). Envoyer le montant tel quel, "
                       "ne calculer aucun ratio.",
    },
    # MODIF : même logique pour "dti" (nom du modèle) / "mensualites_dettes" (nom envoyé par le LLM).
    # Avant : la description parlait de "taux d'endettement en %" -> le LLM calculait lui-même
    # un pourcentage (2.5) au lieu d'envoyer le montant donné par le client (1000).
    # On précise aussi "PAR MOIS" : une dette totale n'est pas une mensualité.
    "dti": {
        "champ": "mensualites_dettes",
        "libelle": "votre niveau d'endettement actuel",
        "description": "total des mensualités de dettes actuelles, en dollars PAR MOIS (hors loyer, "
                       "hors crédit immobilier et hors prêt demandé). 0 si aucune dette. "
                       "Si le client donne un montant total de dette, lui demander combien il rembourse "
                       "par mois. Envoyer le montant tel quel, ne calculer aucun ratio.",
    },
    "inq_last_6mths": {
        "libelle": "vos demandes de crédit récentes",
        # MODIF : "(pas un montant)" ajouté, l'agent demandait "le montant total des demandes".
        "description": "nombre (pas un montant) de demandes de crédit faites ces 6 derniers mois "
                       "(hors auto et immobilier), entier.",
    },
    "revol_util": {
        "libelle": "l'utilisation de vos cartes de crédit",
        "description": "taux d'utilisation des cartes de crédit en % : solde dû / plafond total × 100 "
                       "(ex. 1 500 $ sur 5 000 $ → 30). Peut dépasser 100.",
    },
    "mort_acc": {
        "libelle": "vos prêts immobiliers",
        "description": "nombre de prêts immobiliers (en cours ou remboursés), entier.",
    },
    "purpose": {
        "libelle": "l'objet du prêt",
        "description": "objet du prêt : choisir la valeur acceptée la plus proche de ce que décrit le client "
                       "(ex. voiture → car, regrouper des dettes → debt_consolidation).",
    },
    "home_ownership": {
        "libelle": "votre situation de logement",
        "description": "locataire → RENT, propriétaire sans crédit en cours → OWN, "
                       "propriétaire avec crédit immobilier en cours → MORTGAGE.",
    },
}

# Dérivés automatiquement : une seule source à maintenir
LIBELLES = {nom: f["libelle"] for nom, f in FEATURES.items()}          # pour le client (outil_credit)
# MODIF : clé = "champ" s'il existe (nom envoyé par le LLM), sinon le nom du modèle.
# Ainsi le prompt annonce loan_amnt / mensualites_dettes, exactement ce que lit vers_features,
# tandis que LIBELLES garde dti / loan_to_income, exactement ce que renvoie score_loan.
DESCRIPTIONS = {f.get("champ", nom): f["description"] for nom, f in FEATURES.items()}  # pour le LLM (prompt)


# champs obligatoires
OBLIGATOIRES = ["loan_amnt", "annual_inc"]


# ******************************************************************
# commandes à exécuter: point d'entrée du chatbot
# modèle chargé une seule fois, au premier appel
_ART = None

def outil_credit(arguments: dict) -> dict:
    """Seule fonction appelée par le chatbot (Python direct ou mcp_credit.py).
    Input : arguments envoyés par le LLM (dict, clés de PARAMETRES).
    Output : decision, points_forts et points_faibles en mots pour le client, montant_max en cas de refus."""
    # 1. Charger l'artfact du modèle une seule fois
    global _ART
    if _ART is None:
        _ART = charger_modele()

    # MODIF : nettoyage des arguments du LLM avant tout calcul.
    # - on retire les champs inconnus (ex. "loan_to_amnt" envoyé par erreur) ;
    # - on retire les null : un champ facultatif absent est traité comme inconnu par vers_features.
    champs_valides = set(DESCRIPTIONS)
    arguments = {k: v for k, v in arguments.items() if k in champs_valides and v is not None}

    # 2. Sans montant ou sans revenu, on ne peut pas calculer : le LLM doit les demander
    manquants = [c for c in OBLIGATOIRES if arguments.get(c) is None]
    if manquants:
        return {"erreur": f"Champs obligatoires manquants : {manquants}. Demande-les au client."}

    # 3. Décision du modèle, avec les points forts et faibles (noms techniques)
    resultat = score_loan(arguments, _ART)

    # 4. Noms techniques -> mots pour le client
    # MODIF : .get(f, f) au lieu de [f] -> si une feature n'a pas de libellé, on renvoie son nom
    # au lieu de planter avec une KeyError.
    resultat["points_forts"] = [LIBELLES.get(f, f) for f in resultat["points_forts"]]
    resultat["points_faibles"] = [LIBELLES.get(f, f) for f in resultat["points_faibles"]]

    # 5. En cas de refus : ajouter le plus grand montant acceptable
    if resultat["decision"] == "refus":
        resultat |= montant_max(arguments, _ART)
    return resultat
