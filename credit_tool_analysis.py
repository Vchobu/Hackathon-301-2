import marimo

__generated_with = "0.25.2"
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
    return corr, px


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

    Les paramètres sont choisis par la librairie **optuna** : 50 essais, chacun entraîné sur le train et noté par
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
    ### 4.c Un AUC de 0.65, est-ce bien ?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####4c.1: Préparation du modèle de regression logistique
    """)
    return


@app.cell
def _(FEATURES):
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    def regression_logistique():
        """Régression logistique + préparation des données (LightGBM n'en a pas besoin)."""

        # traitement par type de données
        MONTANTS = ["annual_inc", "loan_to_income"]                  # très étalés -> log
        TEXTE = ["purpose", "home_ownership"]                        # catégories -> colonnes 0/1
        AUTRES = [f for f in FEATURES if f not in MONTANTS + TEXTE]  # fico, dti, mort_acc, ...

        # pour les montants, on impute les médianes, prend le log et standardise
        prep_montants = make_pipeline(
            SimpleImputer(strategy="median"),     # manquant -> médiane
            FunctionTransformer(np.log1p),        # log(1 + x)
            StandardScaler(),                     # moyenne 0, écart-type 1
        )

        # pour les autres numériques manquantes (e.g. fico, dti), on prend la mediane et on l'indique et standardise
        prep_autres = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),  # manquant -> médiane + colonne « était manquant »
            StandardScaler(),
        )

        # pour les textes (purpose, home_ownership), on crée une colonne par catégorie
        # catégories rares (< 1000 prêts) regroupées 
        # catégorie inconnue après l'entrainement -> renvoie 0
        prep_texte = OneHotEncoder(
            min_frequency=1000,                   # catégories de moins de 1000 prêts regroupées
            handle_unknown="ignore",              # catégorie jamais vue après l'entrainement -> on renvoie 0
        )

        # chaque groupe de colonnes reçoit sa préparation
        preparation = ColumnTransformer([         
            ("montants", prep_montants, MONTANTS),
            ("autres", prep_autres, AUTRES),
            ("texte", prep_texte, TEXTE),
        ])

        # construire le modèle: il sera utilisé par les fonctions .fit(X, y) et .predict_proba(X)
        # max_iter=2000 : les coefficients sont trouvés par itérations successives
        return make_pipeline(preparation, LogisticRegression(max_iter=2000))

    return np, regression_logistique


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 4.c.2 Régression logistique contre LightGBM : validation croisée 5-fold
    """)
    return


@app.cell
def _(
    X_train,
    X_val,
    etude,
    lgb,
    np,
    pl,
    regression_logistique,
    roc_auc_score,
    y_train,
    y_val,
):
    # Validation croisée 5-fold sur 2012-2014, et AUC validation 2014 (même protocole que 4.b)
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # on recolle train (2012-13) et validation (2014)
    _X = pd.concat([X_train, X_val])
    _y = np.concatenate([y_train, y_val])

    # les deux modèles à comparer (pas encore entraînés)
    _modeles = {
        "LightGBM (paramètres Optuna)": lgb.LGBMClassifier(**etude.best_params, random_state=42, verbose=-1),
        "Régression logistique": regression_logistique(),
    }

    # 5 plis, chacun avec le même taux de défaut
    _plis = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    _lignes = []
    for _nom, _modele in _modeles.items():
        # CV : entraîné sur 4 plis, AUC sur le pli de validation, 5 fois
        _aucs = cross_val_score(_modele, _X, _y, cv=_plis, scoring="roc_auc")

        # même protocole que 4.b : entraîné sur 2012-13, évalué sur 2014
        _modele.fit(X_train, y_train)
        _auc_val = roc_auc_score(y_val, _modele.predict_proba(X_val)[:, 1])

        _lignes.append({"modèle": _nom,
                        "AUC CV moyenne": round(_aucs.mean(), 3),
                        "écart-type": round(_aucs.std(), 3),
                        "AUC validation 2014": round(_auc_val, 3)})
    pl.DataFrame(_lignes)
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Lecture** (résultats obtenus : LightGBM 0.656 ± 0.003, régression logistique 0.652 ± 0.003) :

    - l'écart-type est très faible : l'AUC de 0.65 est **stable**, ce n'est pas un hasard du découpage ;
    - la régression logistique fait presque aussi bien : **la limite vient des 9 informations demandées au client**,
      pas du choix du modèle. Un modèle plus complexe ne ferait pas beaucoup mieux avec ces données.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.d Importance des features (SHAP)

    Contribution moyenne de chaque feature au risque de défaut, sur 20 000 prêts de validation.
    Le chatbot utilise les mêmes valeurs pour les points forts et faibles du client.
    """)
    return


@app.cell
def _(FEATURES, X_val, modele, np, pl, px):
    _contrib = modele.predict(X_val.sample(20_000, random_state=42), pred_contrib=True)[:, :-1]  # sans le biais
    _importance = pl.DataFrame({"feature": FEATURES, "importance": np.abs(_contrib).mean(axis=0)}).sort("importance")
    px.bar(_importance.to_pandas(), x="importance", y="feature", orientation="h")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Seuil de décision (validation 2014)

    Refus si la probabilité de défaut dépasse le seuil. Le seuil est celui qui maximise le F1 sur la validation.

    - Précision : parmi les prêts refusés, la part qui aurait vraiment fait défaut. Si elle est basse, on refuse beaucoup de bons clients.
    - Recall : parmi tous les défauts, la part que le modèle refuse. Si il est bas, beaucoup de défauts passent.
    """)
    return


@app.cell
def _(p_val, y_val):
    from sklearn.metrics import precision_recall_curve

    # essaie tous les seuils possibles, 
    _prec, _rec, _seuils = precision_recall_curve(y_val, p_val)
    _f1 = 2 * _prec * _rec / (_prec + _rec + 1e-12)  # F1 : moyenne harmonique des deux
    SEUIL = float(_seuils[_f1[:-1].argmax()])        # seuil qui donne le meilleur F1
    print(f"Seuil retenu : {SEUIL:.3f}  (refus si probabilité de défaut >= seuil)")
    return (SEUIL,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Évaluation finale sur le test (2015)

    On compare les modèles sur la **VAN** (valeur actuelle nette) de chaque prêt, taux 1%
    (montant prêté + paiements reçus). Un prêt refusé a une VAN de 0.
    """)
    return


@app.cell
def _(ANNEES, X_train, ct, pd, pl):
    # Prêts de test relus avec les colonnes nécessaires à la VAN (pas données au modèle!)
    COLONNES_CLIENT = ["loan_amnt", "annual_inc", "dti", "fico_range_low", "fico_range_high",
                       "mort_acc", "inq_last_6mths", "revol_util", "purpose", "home_ownership"]
    COLONNES_VAN = ["funded_amnt", "total_pymnt", "recoveries", "collection_recovery_fee", "last_pymnt_d", "sub_grade"]

    # 1. Prêts de 36 mois, terminés, émis l'année de test
    test_2015 = (
        pl.read_csv(ct.DATA_PATH, columns=["issue_d", "loan_status", "term"] + COLONNES_CLIENT + COLONNES_VAN,
                    infer_schema_length=200_000, ignore_errors=True)
        .with_columns(
            pl.col("issue_d").str.strptime(pl.Date, "%b-%Y", strict=False).alias("debut"),
            pl.col("last_pymnt_d").str.strptime(pl.Date, "%b-%Y", strict=False).alias("fin"),
        )
        .filter(
            pl.col("term").str.contains("36"),
            pl.col("debut").dt.year().is_in(ANNEES["test"]),
            pl.col("loan_status").is_in(["Fully Paid", "Charged Off", "Default"]),
        )
    )

    # 2. Cible et features, formatting comme en 4.a
    test_2015 = test_2015.with_columns(
        (pl.col("loan_status") != "Fully Paid").cast(pl.Int8).alias("defaut"),
        ((pl.col("fico_range_low") + pl.col("fico_range_high")) / 2).alias("fico"),
        (pl.col("loan_amnt") / pl.col("annual_inc").clip(lower_bound=1)).alias("loan_to_income"),
    )

    # 3. VAN de chaque prêt
    TAUX_ACTU = 0.05                                  # obligations d’entreprises notées Baa, moyenne 2015
    _r = (1 + TAUX_ACTU) ** (1 / 12) - 1              # taux mensuel

    # a. Nombre de mois de paiement, entre l'émission et le dernier paiement (au moins 1)
    test_2015 = test_2015.with_columns(
        ((pl.col("fin").dt.year() - pl.col("debut").dt.year()) * 12
         + pl.col("fin").dt.month() - pl.col("debut").dt.month()).fill_null(1).clip(1).alias("mois")
    )

    # b. Facteur d'actualisation : valeur aujourd'hui d'1 $ reçu au dernier mois
    test_2015 = test_2015.with_columns(((1 + _r) ** -pl.col("mois")).alias("actu"))

    # c. Argent reçu : payé par le client, et récupéré après un défaut (moins les frais de recouvrement)
    test_2015 = test_2015.with_columns(
        (pl.col("total_pymnt") - pl.col("recoveries")).alias("paye"),
        (pl.col("recoveries") - pl.col("collection_recovery_fee")).alias("recupere"),
    )

    # d. VAN = - montant prêté + mensualités actualisées + recouvrement actualisé
    test_2015 = test_2015.with_columns(
        (-pl.col("funded_amnt")
         + pl.col("paye") / pl.col("mois") * (1 - pl.col("actu")) / _r
         + pl.col("recupere") * pl.col("actu")).alias("van")
    )

    # 4. Note de la banque en rang : A1 = 0 (sûr) ... G5 = 34 (risqué)
    _notes = [f"{lettre}{chiffre}" for lettre in "ABCDEFG" for chiffre in range(1, 6)]
    test_2015 = test_2015.with_columns(pl.col("sub_grade").replace_strict(_notes, list(range(35))).alias("note_lc"))

    # 5. Pandas, avec les mêmes catégories que le train
    test_2015 = test_2015.to_pandas()
    for _c in ["purpose", "home_ownership"]:
        test_2015[_c] = pd.Categorical(test_2015[_c], categories=X_train[_c].cat.categories)

    print(f"Test {ANNEES['test']} : {len(test_2015)} prêts, {test_2015['defaut'].mean() * 100:.1f} % de défauts, "
          f"VAN totale si tout est accepté {test_2015['van'].sum() / 1e6:.1f} M$")
    return (test_2015,)


@app.cell
def _(FEATURES, SEUIL, modele, np, pl, roc_auc_score, test_2015):
    # Décision accepter / refuser sur le test 2015 : le modèle contre les autres façons de décider

    # 1. Ce qui s'est réellement passé pour chaque prêt
    _defaut = test_2015["defaut"].to_numpy()          # 1 = défaut
    _van = test_2015["van"].to_numpy()                # ce que le prêt a rapporté

    # 2. Décision du modèle : accepté si la probabilité de défaut est sous le seuil
    _p = modele.predict_proba(test_2015[FEATURES])[:, 1]
    _accepte_modele = _p < SEUIL
    _nb_acceptes = _accepte_modele.sum()

    # 3. Les autres façons de décider (True = accepté)
    _accepte_tout = np.ones(len(_defaut), dtype=bool)
    _accepte_regle = ~((test_2015["dti"] > 30) | (test_2015["fico"] < 670)).to_numpy()
    # la banque accepte autant de prêts que le modèle : les mieux notés (A1, A2, ...)
    _accepte_banque = (test_2015["note_lc"].rank(method="first") <= _nb_acceptes).to_numpy()

    # 4. Bilan de chaque décision
    def _bilan(nom, accepte):
        return {"décision": nom,
                "acceptés %": round(accepte.mean() * 100, 1),
                "défaut chez les acceptés %": round(_defaut[accepte].mean() * 100, 1),
                "VAN totale M$": round(_van[accepte].sum() / 1e6, 1)}

    print(f"AUC test : modèle {roc_auc_score(_defaut, _p):.3f}, "
          f"note de la banque {roc_auc_score(_defaut, test_2015['note_lc']):.3f}")
    pl.DataFrame([
        _bilan("Tout accepter", _accepte_tout),
        _bilan("Règle simple (DTI > 30 ou FICO < 670 -> refus)", _accepte_regle),
        _bilan("Note de la banque, même taux d'acceptation", _accepte_banque),
        _bilan(f"Modèle (seuil {SEUIL:.3f})", _accepte_modele),
    ])
    return


@app.cell
def _(FEATURES, SEUIL, modele, np, pl, px, test_2015):
    # Courbe : VAN totale selon la part des dossiers acceptés, notre modèle contre la note de la banque

    _van = test_2015["van"].to_numpy()
    _scores = {
        "Notre modèle": modele.predict_proba(test_2015[FEATURES])[:, 1],   # probabilité de défaut
        "Note de la banque": test_2015["note_lc"].to_numpy(),              # A1 = 0 ... G5 = 34
    }

    # Pour chaque part acceptée : on garde les dossiers les moins risqués selon chaque score
    _lignes = []
    for _part in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for _nom, _score in _scores.items():
            _ordre = np.argsort(_score, kind="stable")                    # du moins au plus risqué
            _acceptes = _ordre[: int(_part * len(_score))]
            _lignes.append({"Part des dossiers acceptés": int(_part * 100), "décision": _nom,
                            "VAN totale (M$)": round(_van[_acceptes].sum() / 1e6, 1)})

    _fig = px.line(pl.DataFrame(_lignes).to_pandas(), x="Part des dossiers acceptés", y="VAN totale (M$)",
                   color="décision", markers=True,
                   title="VAN totale 2015 selon la part des dossiers acceptés",
                   color_discrete_map={"Notre modèle": "#D9692A", "Note de la banque": "#2F6FB0"})

    # Ligne verticale : part des dossiers acceptés au seuil du modèle (environ 61.5 %)
    _part_seuil = (_scores["Notre modèle"] < SEUIL).mean() * 100
    _fig.add_vline(x=_part_seuil, line_dash="dash", line_color="#6B7280",
                   annotation_text=f"seuil du modèle ({_part_seuil:.1f} %)", annotation_position="top")

    # Point : la règle simple (refus si DTI > 30 ou FICO < 670), une seule décision
    _regle = ~((test_2015["dti"] > 30) | (test_2015["fico"] < 670)).to_numpy()
    _fig.add_scatter(x=[_regle.mean() * 100], y=[round(_van[_regle].sum() / 1e6, 1)],
                     mode="markers", name="Règle DTI > 30 ou FICO < 670",
                     marker=dict(symbol="diamond", size=18, color="#14213D"))

    # Mise en forme pour la présentation : textes plus grands, lignes plus épaisses
    _fig.update_layout(
        font=dict(size=22),                     # taille de base : axes, légende, annotation
        title_font_size=28,
        legend=dict(font=dict(size=22), title_text=""),
        width=1100, height=650,                 # même proportion que l'emplacement sur la slide
        margin=dict(t=110),                     # place pour le titre et l'annotation du seuil
        plot_bgcolor="white",
    )
    _fig.update_traces(line_width=4, marker_size=12)
    _fig.update_xaxes(title_font_size=24, tickfont_size=22, gridcolor="#E5E5E5", ticksuffix=" %")
    _fig.update_yaxes(title_font_size=24, tickfont_size=22, gridcolor="#E5E5E5")

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Conclusion** : l’AUC du modèle (0.65) est un peu plus bas que celui de la note de la banque (0.68), qui utilise le dossier de crédit complet. Mais pour décider qui financer, le modèle fait mieux : à taux d’acceptation égal, sa VAN est de 38 M$ contre 30 M$ pour la banque, et 29 M$ si on accepte tout.

    Limite : on ne connaît que les prêts déjà acceptés par la banque. (on ne pouvait pas faire autrement. Un dossier refusé n’a jamais reçu de prêt, donc on ne sait pas s’il aurait remboursé. Il n’a pas de cible (défaut ou non), ni de VAN.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Sauvegarde et fonctions pour le chatbot

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
    ## 8. Example 101 d'utilisation du modèle par le chatbot

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
    # Les fonctions du chatbot sont dans credit_tool.py : ici on recharge le modèle comme fera le chatbot
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
