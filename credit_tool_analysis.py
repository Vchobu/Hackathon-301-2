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
    return (candidats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. periodes disponibles, sélection et split (train, validation, test)
    """)
    return


@app.cell
def _(candidats, ct, pl):
    # 1. Charger les prêts : date, statut, durée et les 15 candidats
    prets = pl.read_csv(ct.DATA_PATH, columns=["issue_d", "loan_status", "term"] + candidats,
                        infer_schema_length=200_000, ignore_errors=True)

    # 2. Garder les prêts de 36 mois et ajouter l'année d'émission
    # 36 et pas 60 car il ne resterait que 2012 et 2013 (pas 2014 et 2015) = pas assez de données
    prets_36 = prets.filter(pl.col("term").str.contains("36")).with_columns(
        pl.col("issue_d").str.strptime(pl.Date, "%b-%Y", strict=False).dt.year().alias("annee")
    )
    return (prets_36,)


@app.cell
def _(candidats, pl, prets_36):
    # 3. Résumer par année
    termine = pl.col("loan_status").is_in(["Fully Paid", "Charged Off", "Default"])
    pire_manquant = pl.max_horizontal([pl.col(c).is_null().mean() for c in candidats])

    par_annee = prets_36.group_by("annee").agg(
        pl.len().alias("prets"),
        (termine.mean() * 100).round(1).alias("termines_%"),       # part des prêts terminés
        (pire_manquant * 100).round(1).alias("manquants_%"),       # pire candidat de l'année
    ).sort("annee")
    par_annee
    return (par_annee,)


@app.cell
def _(par_annee, pl):
    # Filtre A : prêts terminés fin 2018 (sinon on ne connaît pas encore le défaut = target)
    periode_a = par_annee.filter(pl.col("termines_%") >= 99)
    print(f"Filtre A : {par_annee.height} -> {periode_a.height} années")
    par_annee.filter(pl.col("termines_%") < 99).select("annee", "termines_%")
    return (periode_a,)


@app.cell
def _(periode_a, pl):
    # Filtre B : variables manquantes (avant 2012, certaines colonnes n'existent pas)
    periode_b = periode_a.filter(pl.col("manquants_%") <= 20)
    print(f"Filtre B : {periode_a.height} -> {periode_b.height} années")
    periode_a.filter(pl.col("manquants_%") > 20).select("annee", "manquants_%")
    return (periode_b,)


@app.cell
def _(periode_b):
    # Split temporel : dernière année = test, avant-dernière = validation, le reste = train
    periode = periode_b["annee"].to_list()
    ANNEES = {"train": periode[:-2], "validation": [periode[-2]], "test": [periode[-1]]}
    ANNEES
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Analyse des features sélectionnées pour la période de train
    """)
    return


if __name__ == "__main__":
    app.run()
