# Hackathon 301-2 : chatbot de pré-évaluation de crédit

Données : [Lending Club 2007-2018](https://www.kaggle.com/datasets/wordsforthewise/lending-club),
on le trouve dans `data/accepted_2007_to_2018Q4.csv.gz`.

## Partie 1 : modèle de crédit

### Installation

```
uv sync
uv run marimo edit credit_tool_analysis.py
```

### Modèle de crédit (`credit_tool_analysis.py`)

| Étape | Choix |
|---|---|
| Features | 9 informations que le client connaît au moment de la demande : FICO, revenu, DTI, montant / revenu, hypothèques, demandes de crédit récentes, utilisation du crédit renouvelable, objet du prêt, logement |
| Données | Prêts de 36 mois, 2012-2015, tous terminés |
| Split temporel | Train 2012-2013, validation 2014, test 2015 |
| Modèle | LightGBM, paramètres choisis par Optuna sur la validation |
| Décision | Refus si la probabilité de défaut >= seuil (max F1 sur la validation, ~0.14) |
| Explications | SHAP : points forts et faibles de chaque client |

**Résultats**
- AUC ~0.65, stable en validation croisée (+/- 0.003). La régression logistique fait pareil : la limite vient des données, pas du modèle.
- Sur le test 2015, à taux d'acceptation égal, le modèle choisit des prêts plus rentables que la note de la banque : VAN de 38 M$ contre 30 M$ (actualisation à 5 %).
- Limite : on ne connaît que les prêts déjà acceptés par la banque.

### Outil du chatbot (fonction `outil_credit`)

Le modèle est sauvegardé dans `output/modele_credit.joblib` et utilisé par le chatbot via `credit_tool.py` (`outil_credit`).

