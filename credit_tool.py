"""Modèle de crédit Lending Club : fonctions utilisées par le notebook, l'API et le chatbot."""
import gzip
import io

import polars as pl

DATA_PATH = "data/accepted_2007_to_2018Q4.csv.gz"
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
