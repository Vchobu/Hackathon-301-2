import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    import credit_tool as ct

    return ct, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Credit tool : analyse

    ## 1. Features disponibles et sélection

    ### Colonnes disponibles
    """)
    return


@app.cell
def _(ct):
    disponibles = ct.colonnes_disponibles()
    print(f"{len(disponibles)} colonnes dans le CSV")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Filtrage avec le dictionnaire de données

    utilisation du fichier `LCDataDictionary.xlsx` qui permet à la sélection des features.

    On va ci-dessous appliquer filtre par filtre
    """)
    return


@app.cell
def _(ct):
    dico = ct.dictionnaire().drop("Description")
    print(f"{dico.height} colonnes dans le dictionnaire")
    dico
    return (dico,)


@app.cell
def _(dico, pl):
    # Filtre 1 : connue au moment de la demande (sinon fuite : le modèle verrait le futur)
    etape1 = dico.filter(pl.col("Connue au moment de la demande") == "Oui")
    print(f"Filtre 1 : {dico.height} -> {etape1.height} colonnes")
    dico.filter(pl.col("Connue au moment de la demande") == "Non").select("colonne")
    return (etape1,)


@app.cell
def _(etape1, pl):
    # Filtre 2 : pas une décision de la banque (grade, int_rate... servent seulement de comparaison)
    etape2 = etape1.filter(pl.col("Pas une décision de la banque") == "Oui")
    print(f"Filtre 2 : {etape1.height} -> {etape2.height} colonnes")
    etape1.filter(pl.col("Pas une décision de la banque") == "Non").select("colonne")
    return (etape2,)


@app.cell
def _(etape2, pl):
    # Filtre 3 : le client connaît l'information (sinon le chatbot ne peut pas la demander)
    etape3 = etape2.filter(pl.col("Le client a l'information") == "Oui")
    print(f"Filtre 3 : {etape2.height} -> {etape3.height} colonnes")
    etape2.filter(pl.col("Le client a l'information") == "Non").select("colonne")
    return (etape3,)


@app.cell
def _(etape3, pl):
    # Filtre 4 : exploitable par le modèle et le chatbot
    NON_EXPLOITABLES = {
        "desc": "Not relevant =Texte libre",
        "title": "Not relevant =Texte libre",
        "emp_title": "Not relevant =Texte libre",
        "zip_code": "Not relevant =US",
        "addr_state": "Not relevant =US",
        "annual_inc_joint": "Not relevant =demande individuelle",
        "dti_joint": "Not relevant =demande individuelle",
        "application_type": "Not relevant =demande individuelle",
        "term": "Not relevant =Périmètre fixé : prêts de 36 mois",
    }
    etape4 = etape3.filter(~pl.col("colonne").is_in(list(NON_EXPLOITABLES)))
    print(f"Filtre 4 : {etape3.height} -> {etape4.height} colonnes")
    pl.DataFrame({"colonne": list(NON_EXPLOITABLES), "raison": list(NON_EXPLOITABLES.values())})
    return (etape4,)


@app.cell
def _(etape4):
    # Candidats
    candidats = etape4["colonne"].to_list()
    print(f"{len(candidats)} candidats :", candidats)
    etape4.select("colonne")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. periodes disponibles et sélection
    """)
    return


if __name__ == "__main__":
    app.run()
