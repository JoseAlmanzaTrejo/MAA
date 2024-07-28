import pandas as pd
import numpy as np
import optuna

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay, auc

import matplotlib.pyplot as plt


def fnc_cv_classifier(X: pd.DataFrame, y: pd.Series, classifier: any, cv: StratifiedKFold, modelo: str, print: bool= False) -> list:

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    n_splits=cv.get_n_splits()

    fig, ax = plt.subplots(figsize=(10, 10))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y.iloc[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[test],
            y[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == n_splits - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(float(viz.roc_auc))
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability\n(Positive label '{y.name}')",
    )
    ax.legend(loc="lower right")
    if print:
        plt.savefig(f"PIA/img/roc_auc_cv_{modelo}.png")
    plt.close()

    return aucs

def fnc_modelo_lr(data: pd.DataFrame, X: list, y: str) -> list:

    scaling = preprocessing.StandardScaler().fit_transform(data[X])
    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=123)
    
    params = []
    for penalty in ['l1','l2','elasticnet']:
        for solver in ['lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga']:
            for C in [0.001, 0.1, 1.0, 100, 10000]:
                for l1_ratio in [0.1, 0.25, 0.5, 0.75, 0.9, None]:

                    if solver in ['lbfgs','newton-cg','newton-cholesky','sag'] and penalty in ['l1','elasticnet']:
                        continue
                    if (solver != 'saga') and (penalty == 'elasticnet'):
                        continue
                    if (penalty == 'elasticnet') and (l1_ratio in [None]):
                        continue
                    if (penalty != 'elasticnet') and (l1_ratio not in [None]):
                        continue
                    params = params + [{'penalty':penalty,'solver':solver,'C':C, 'l1_ratio':l1_ratio}]

    auc_val = 0
    bst_params = {}
    for i, parametros in enumerate(params):
        lr_logreg = LogisticRegression(**parametros, random_state=123)
        aucs = fnc_cv_classifier(X = scaling, y = data[y], classifier = lr_logreg, cv = cv, modelo=f'Regresión Logística{i}', print=True)

        if np.mean(aucs) > auc_val:
            auc_val = np.mean(aucs)
            bst_params = parametros
    
    lr_logreg = LogisticRegression(**bst_params, random_state=123)
    aucs = fnc_cv_classifier(X = scaling, y = data[y], classifier = lr_logreg, cv = cv, modelo=f'Regresión Logística bst', print=True)
    return auc_val, bst_params

def fnc_objective_dt(trial):
    nlp = pd.read_csv('csv/train.csv')
    nlp.engagement = nlp.engagement.apply(lambda x: 0 if x==False else 1)
    X = list(nlp.columns[1:-1])
    y = 'engagement'

    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=123)

    params = {}
    params['max_depth'] = trial.suggest_int('max_depth', 1, 50)
    params['criterion'] = trial.suggest_categorical('criterion', ['gini','entropy','log_loss'])
    params['min_samples_split'] = trial.suggest_float('min_samples_split', 0.01,1.00)
    params['min_samples_leaf'] = trial.suggest_float('min_samples_leaf', 0.01,1.00)

    classifier = DecisionTreeClassifier(**params)
    
    aucs = fnc_cv_classifier(X = nlp[X].values, y = nlp[y], classifier = classifier, cv = cv, modelo=f'Decision Tree Classifier{trial._trial_id}', print=True)

    return np.mean(aucs)

def fnc_objective_rf(trial):
    nlp = pd.read_csv('csv/train.csv')
    nlp.engagement = nlp.engagement.apply(lambda x: 0 if x==False else 1)
    X = list(nlp.columns[1:-1])
    y = 'engagement'

    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=123)

    params = {}
    params['n_estimators'] = trial.suggest_int('n_estimators', 1, 200)
    params['max_depth'] = trial.suggest_int('max_depth', 1, 200)
    params['criterion'] = trial.suggest_categorical('criterion', ['gini','entropy','log_loss'])
    params['min_samples_split'] = trial.suggest_float('min_samples_split', 0.01,1.00)
    params['min_samples_leaf'] = trial.suggest_float('min_samples_leaf', 0.01,1.00)
    params['max_leaf_nodes'] = trial.suggest_int('max_leaf_nodes', 2, 50)

    classifier = RandomForestClassifier(**params)
    
    aucs = fnc_cv_classifier(X = nlp[X].values, y = nlp[y], classifier = classifier, cv = cv, modelo=f'Random Forest Classifier{trial._trial_id}', print=True)

    return np.mean(aucs)

def fnc_objective_gb(trial):
    nlp = pd.read_csv('csv/train.csv')
    nlp.engagement = nlp.engagement.apply(lambda x: 0 if x==False else 1)
    X = list(nlp.columns[1:-1])
    y = 'engagement'

    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=123)

    params = {}
    params['learning_rate'] = trial.suggest_float('learning_rate', 0.001, 0.5)
    params['n_estimators'] = trial.suggest_int('n_estimators', 1, 200)
    params['max_depth'] = trial.suggest_int('max_depth', 1, 200)
    params['subsample'] = trial.suggest_float('subsample', 0.1, 1.0)
    params['min_samples_split'] = trial.suggest_float('min_samples_split', 0.01,1.00)
    params['min_samples_leaf'] = trial.suggest_float('min_samples_leaf', 0.01,1.00)
    params['max_leaf_nodes'] = trial.suggest_int('max_leaf_nodes', 2, 50)

    classifier = GradientBoostingClassifier(**params)
    
    aucs = fnc_cv_classifier(X = nlp[X].values, y = nlp[y], classifier = classifier, cv = cv, modelo=f'Gradient Booster Classifier{trial._trial_id}', print=True)

    return np.mean(aucs)


def fnc_bst_params(modelo: str) -> dict:

    study = optuna.create_study(direction='maximize')
    
    if modelo == 'Decision Tree Classifier':
        study.optimize(fnc_objective_dt, n_trials=250)
    elif modelo == 'Random Forest Classifier':
        study.optimize(fnc_objective_rf, n_trials=250)
    elif modelo == 'Gradient Boosting Classifier':
        study.optimize(fnc_objective_gb, n_trials=250)
    
    print("Best hyperparameters:", study.best_params)
    print("Best value:", study.best_value)
    return study.best_params
    
def fnc_modelo_dt(data: pd.DataFrame, X: list, y: str) -> list:

    best_params = fnc_bst_params(modelo = 'Decision Tree Classifier')
    
    scaling = preprocessing.StandardScaler().fit_transform(data[X])
    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=123)

    regression = DecisionTreeClassifier(**best_params, random_state=123)
    aucs = fnc_cv_classifier(X = scaling, y = data[y], classifier = regression, cv = cv, modelo=f'Decision Tree bst', print=True)

    return np.mean(aucs), best_params

def fnc_modelo_rf(data: pd.DataFrame, X: list, y: str) -> list:

    best_params = fnc_bst_params(modelo = 'Random Forest Classifier')
    
    scaling = preprocessing.StandardScaler().fit_transform(data[X])
    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=123)

    regression = RandomForestClassifier(**best_params, random_state=123)
    aucs = fnc_cv_classifier(X = scaling, y = data[y], classifier = regression, cv = cv, modelo=f'Random Forest bst', print=True)

    return np.mean(aucs), best_params

def fnc_modelo_gb(data: pd.DataFrame, X: list, y: str) -> list:

    best_params = fnc_bst_params(modelo = 'Gradient Boosting Classifier')
    
    scaling = preprocessing.StandardScaler().fit_transform(data[X])
    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=123)

    regression = GradientBoostingClassifier(**best_params, random_state=123)
    aucs = fnc_cv_classifier(X = scaling, y = data[y], classifier = regression, cv = cv, modelo=f'Gradient Boosting bst', print=True)

    return np.mean(aucs), best_params

def fnc_descipcion_caracteristicas(data, X):

    for columna in X:
        plt.hist(data[columna], bins=96)

        # Adding labels and title
        plt.xlabel('Valores')
        plt.ylabel('Frecuencia')
        plt.title(f'Histograma de variable {columna}')

        plt.savefig(f"PIA/img/histograma {columna}.png")
        plt.close()


if __name__ == "__main__":
    #Leemos la información para entrenar
    nlp = pd.read_csv('csv/train.csv')
    nlp.engagement = nlp.engagement.apply(lambda x: 0 if x==False else 1)
    X = list(nlp.columns[1:-1])

    fnc_descipcion_caracteristicas(data=nlp, X=X)
    results_lr, bst_params_lr = fnc_modelo_lr(data = nlp, X = X, y = 'engagement')
    results_dt, bst_params_dt = fnc_modelo_dt(data = nlp, X = X, y = 'engagement')
    results_rf, bst_params_rf = fnc_modelo_rf(data = nlp, X = X, y = 'engagement')
    results_gb, bst_params_gb = fnc_modelo_gb(data = nlp, X = X, y = 'engagement')

    for modelos in [['Regresión Logística',float(results_lr),bst_params_lr],
                ['Árbol de Decisiones',float(results_dt),bst_params_dt],
                ['Bosque Aleatorio',float(results_rf),bst_params_rf],
                ['Potenciación del Gradiente',float(results_gb),bst_params_gb]]:
        print(f'Algoritmo: {modelos[0]}, ROC AUC promedio: {modelos[1]}, parámetros: {modelos[2]}')
