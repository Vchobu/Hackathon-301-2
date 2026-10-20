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
    return (ANNEES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Analyse des features sélectionnées pour la période de train
    """)
    return


@app.cell
def _(ANNEES, pl, prets_36):
    # Train : prêts terminés des années de train, cible defaut = 1
    train = prets_36.filter(
        pl.col("annee").is_in(ANNEES["train"]),
        pl.col("loan_status").is_in(["Fully Paid", "Charged Off", "Default"]),
    ).with_columns(
        (pl.col("loan_status") != "Fully Paid").cast(pl.Int8).alias("defaut"),
        # ratio montant / revenu : le montant seul ne dit presque rien, son poids sur le revenu oui
        (pl.col("loan_amnt") / pl.col("annual_inc").clip(lower_bound=1)).alias("loan_to_income"),
    )
    print(f"Train : {train.height} prêts, {train['defaut'].mean() * 100:.1f} % de défauts")
    return (train,)


@app.cell
def _(candidats, pl, train):
    # Séparation valeurs numériques vs texte

    # a. Valeur numérique
    # on ne prend que fico_range_low car fico_range_high = low + 4 (tranche de 5 points)
    NUM = [c for c in candidats if train[c].dtype.is_numeric() and c != "fico_range_high"] + ["loan_to_income"]

    # b. Valeurs texte
    CAT = [c for c in candidats if train[c].dtype == pl.String]
    print("NUM :", NUM)
    print("CAT :", CAT)
    return (NUM,)


@app.cell
def _(NUM, pl, train):
    # Matrice de corrélation (Spearman = sur les rangs, robuste aux revenus extrêmes)
    import plotly.express as px

    corr = train.select(NUM + ["defaut"]).drop_nulls().select(pl.all().rank()).corr()
    px.imshow(corr.to_numpy(), x=corr.columns, y=corr.columns, text_auto=".2f",
              color_continuous_scale="RdBu_r", zmin=-1, zmax=1, width=700, height=700)
    return (corr,)


@app.cell
def _(corr, pl):
    # a. Valeur numérique: filtre corrélation feature avec variable default


    # Filtre si |corrélation avec défaut| >= 0.05
    lien = (corr.select(pl.Series("variable", corr.columns), pl.col("defaut").round(3).alias("corr"))
            .filter(pl.col("variable") != "defaut").sort(pl.col("corr").abs(), descending=True)
            .with_columns((pl.col("corr").abs() >= 0.05).alias("gardee")))
    print(f"Filtre 5 : {lien.height} -> {lien['gardee'].sum()} variables numériques")
    lien
    return (lien,)


@app.cell
def _(corr, lien, pl):
    # a. Valeur numérique: filtre corrélation features entre elles : > 0.6 -> filtrer

    gardees = lien.filter("gardee")["variable"].to_list()
    paires = (corr.with_columns(pl.Series("a", corr.columns))
              .unpivot(index="a", variable_name="b", value_name="corr")
              .filter(pl.col("a").is_in(gardees), pl.col("b").is_in(gardees),
                      pl.col("a") < pl.col("b"), pl.col("corr").abs() > 0.6))
    print(f"{paires.height} paire(s) redondante(s)")
    paires
    return


@app.cell
def _(pl, train):
    # b. Valeurs texte : on regarde si le taux de défaut change selon la réponse

    # Nombre de prêts et % de défauts pour chaque réponse possible
    # ignore les réponses vides et les catégories de moins de 1 000 prêts
    def taux_defaut(colonne):
        """Nombre de prêts et % de défauts pour chaque réponse possible"""
        return (train.filter(pl.col(colonne).is_not_null())
                .group_by(colonne)
                .agg(pl.len().alias("prets"), (pl.col("defaut").mean() * 100).round(1).alias("defaut_%"))
                .filter(pl.col("prets") >= 1000)
                .sort("defaut_%"))

    return (taux_defaut,)


@app.cell
def _(taux_defaut):
    t_purpose = taux_defaut("purpose")
    ecart_purpose = t_purpose["defaut_%"].max() - t_purpose["defaut_%"].min()
    print(f"purpose : écart {ecart_purpose:.1f} points -> {'gardée' if ecart_purpose >= 3 else 'écartée'}")
    t_purpose
    return (ecart_purpose,)


@app.cell
def _(taux_defaut):
    t_home = taux_defaut("home_ownership")
    ecart_home = t_home["defaut_%"].max() - t_home["defaut_%"].min()
    print(f"home_ownership : écart {ecart_home:.1f} points -> {'gardée' if ecart_home >= 3 else 'écartée'}")
    t_home
    return (ecart_home,)


@app.cell
def _(taux_defaut):
    t_emp = taux_defaut("emp_length")
    ecart_emp = t_emp["defaut_%"].max() - t_emp["defaut_%"].min()
    print(f"emp_length : écart {ecart_emp:.1f} points -> {'gardée' if ecart_emp >= 3 else 'écartée'}")
    t_emp
    return (ecart_emp,)


@app.cell
def _(ecart_emp, ecart_home, ecart_purpose, pl):
    # b. Valeurs texte : résumé

    ecarts = pl.DataFrame({
        "variable": ["purpose", "home_ownership", "emp_length"],
        "ecart_pts": [round(ecart_purpose, 1), round(ecart_home, 1), round(ecart_emp, 1)],
    }).with_columns((pl.col("ecart_pts") >= 3).alias("gardee"))
    print(f"Filtre : {ecarts.height} -> {ecarts['gardee'].sum()} variables texte")
    ecarts
    return


@app.cell
def _(dico, pl):
    # Sélection finale (résultat des filtres 5 et 6)

    # Features données au modèle
    FEATURES = ["fico", "annual_inc", "dti", "loan_to_income", "mort_acc",
                "inq_last_6mths", "revol_util", "purpose", "home_ownership"]

    # Colonnes que le chatbot demande au client
    # (fico = moyenne de fico_range_low et fico_range_high ; 
    # (loan_to_income = loan_amnt / annual_inc)
    A_DEMANDER = ["loan_amnt", "annual_inc", "dti", "fico_range_low",
                  "mort_acc", "inq_last_6mths", "revol_util", "purpose", "home_ownership"]

    dico.filter(pl.col("colonne").is_in(A_DEMANDER)).select("colonne", "Justification")
    return (FEATURES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Model training
    ### 4.a dataset spliting
    """)
    return


@app.cell
def _(pl, prets_36):
    # Prêts terminés, avec la cible et les features calculées
    donnees = (prets_36
        .filter(pl.col("loan_status").is_in(["Fully Paid", "Charged Off", "Default"]))
        .with_columns(
            (pl.col("loan_status") != "Fully Paid").cast(pl.Int8).alias("defaut"),            # 1 = défaut
            ((pl.col("fico_range_low") + pl.col("fico_range_high")) / 2).alias("fico"),        # score FICO
            (pl.col("loan_amnt") / pl.col("annual_inc").clip(lower_bound=1)).alias("loan_to_income"),
        )
        .to_pandas())

    # Texte -> catégorie (LightGBM les gère directement)
    donnees["purpose"] = donnees["purpose"].astype("category")
    donnees["home_ownership"] = donnees["home_ownership"].astype("category")
    return (donnees,)


@app.cell
def _(ANNEES, FEATURES, donnees, pl):
    # Découpage par année : train, validation, test
    train_df = donnees[donnees["annee"].isin(ANNEES["train"])]
    val_df = donnees[donnees["annee"].isin(ANNEES["validation"])]
    test_df = donnees[donnees["annee"].isin(ANNEES["test"])]

    X_train, y_train = train_df[FEATURES], train_df["defaut"].to_numpy()
    X_val, y_val = val_df[FEATURES], val_df["defaut"].to_numpy()
    X_test, y_test = test_df[FEATURES], test_df["defaut"].to_numpy()

    pl.DataFrame({
        "jeu": ["train", "validation", "test"],
        "prets": [len(y_train), len(y_val), len(y_test)],
        "defaut_%": [round(y.mean() * 100, 1) for y in (y_train, y_val, y_test)],
    })
    return X_train, X_val, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.b LightGBM

    Entraîné sur le train uniquement. Pas de `class_weight` : la probabilité de défaut doit rester réaliste.

    Les paramètres sont choisis par la librairie optuna : 50 essais, chacun entraîné sur le train et noté par
    l'AUC sur la validation 2014. Le test n'est pas utilisé ici.
    """)
    return


@app.cell
def _(X_train, X_val, pl, y_train, y_val):
    # Recherche des paramètres : chaque essai = un modèle entraîné sur le train, noté sur la validation
    import lightgbm as lgb
    import optuna
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objectif(trial):
        """Un essai : des paramètres tirés par Optuna -> AUC sur la validation."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 600),                      # nombre d'arbres
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),       # pas de chaque arbre
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),                            # taille des arbres
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True), # prêts min. par feuille
        }
        m = lgb.LGBMClassifier(**params, random_state=42, verbose=-1).fit(X_train, y_train)
        return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

    etude = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    etude.optimize(objectif, n_trials=50)

    print(f"Meilleure AUC validation : {etude.best_value:.4f} (essai {etude.best_trial.number} sur 50)")
    pl.DataFrame([{"paramètre": k, "valeur": round(v, 4) if isinstance(v, float) else v}
                  for k, v in etude.best_params.items()])
    return etude, lgb, roc_auc_score


@app.cell
def _(etude, pl):
    # Les 10 meilleurs essais : les bons paramètres se ressemblent-ils ?
    pl.DataFrame([{"essai": t.number, "auc_val": round(t.value, 4), **t.params} for t in etude.trials]).sort(
        "auc_val", descending=True).head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Les meilleures convergent toutes vers des petits arbres, beaucoup de prêts par feuille
    """)
    return


@app.cell
def _(X_train, etude, lgb, y_train):
    # Modèle final avec les meilleurs paramètres
    modele = lgb.LGBMClassifier(**etude.best_params, random_state=42, verbose=-1)
    modele.fit(X_train, y_train)
    return (modele,)


@app.cell
def _(X_val, modele, roc_auc_score, y_val):
    p_val = modele.predict_proba(X_val)[:, 1]   # probabilité de défaut de chaque prêt
    auc_val = roc_auc_score(y_val, p_val)
    print("AUC validation (2014) :", round(auc_val, 3), " (0,5 = hasard, 1 = classement parfait)")
    return (p_val,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Seuil de décision (validation 2014)

    Refus si la probabilité de défaut dépasse le seuil. Le seuil est celui qui maximise le F1 sur la validation.
    """)
    return


@app.cell
def _(p_val, y_val):
    from sklearn.metrics import precision_recall_curve

    _prec, _rec, _seuils = precision_recall_curve(y_val, p_val)
    _f1 = 2 * _prec * _rec / (_prec + _rec + 1e-12)
    SEUIL = float(_seuils[_f1[:-1].argmax()])
    print(f"Seuil retenu : {SEUIL:.3f}  (refus si probabilité de défaut >= seuil)")
    return (SEUIL,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Sauvegarde et fonctions pour le chatbot

    Le chatbot ne reçoit jamais la probabilité ni le seuil (le client ne doit pas pouvoir « viser » le seuil),
    et une valeur manquante n'est jamais citée comme point fort ou faible.
    """)
    return


@app.cell
def _(FEATURES, SEUIL, X_train, modele):
    import os        # pour créer le dossier
    import joblib    # pour écrire le fichier

    CHEMIN_MODELE = "output/modele_credit.joblib"
    ARTEFACT = {
        "modele": modele,   # le LightGBM entrainé: proba de défaut
        "seuil": SEUIL,     # e.g. 0.139 : au-dessus, refus
        "features": FEATURES,  # les 9 features, dans l'ordre attendu par le modèle
        "categories": {c: list(X_train[c].cat.categories) for c in ["purpose", "home_ownership"]},
        # les valeurs possibles de purpose et home_ownership
    }
    os.makedirs("output", exist_ok=True)
    joblib.dump(ARTEFACT, CHEMIN_MODELE)
    print("Modèle sauvegardé :", CHEMIN_MODELE)
    return (CHEMIN_MODELE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Example 101 d'utilisation du modèle par le chatbot

    Le chatbot :
    1. charge le modèle sauvegardé (`ct.charger_modele`) ;
    2. envoie les réponses du client à `ct.score_loan`, qui renvoie la décision, les points forts et les points faibles ;
    3. en cas de refus, il appelle `ct.montant_max` pour proposer un montant plus petit.

    Notes sur les valeurs manquantes :
    - Champs obligatoires : loan_amnt et annual_inc
    - une valeur absente n’est jamais citée comme point fort ou faible, puisqu’on ne la connaît pas ;
    - la décision est moins fiable avec peu d’informations. Le chatbot doit donc essayer d’obtenir un maximum de réponses avant d’appeler l’outil
    """)
    return


@app.cell
def _(CHEMIN_MODELE, ct):
    # Les fonctions du chatbot sont dans credit_tool.py : on recharge le modèle comme le fera le chatbot
    art = ct.charger_modele(CHEMIN_MODELE)
    # Exemple :  client  demande 25 000 avec 45 000 de revenu
    exemple = {"loan_amnt": 25000, "annual_inc": 45000, "mensualites_dettes": 900, "fico": 675,
               "mort_acc": 0, "inq_last_6mths": 2, "revol_util": 80,
               "purpose": "debt_consolidation", "home_ownership": "RENT"}
    resultat = ct.score_loan(exemple, art)
    if resultat["decision"] == "refus":
        resultat |= ct.montant_max(exemple, art)
    resultat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO meilleure gestion des valeurs manquantes
    """)
    return


if __name__ == "__main__":
    app.run()
