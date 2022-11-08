#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220809
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# install scikit-optimize
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import cohen_kappa_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from imblearn.pipeline import Pipeline
from sklearn.metrics import get_scorer
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from datetime import datetime
from sklearn import tree
import lightgbm as lgb
import pandas as pd

# scoring dict
scoring_dict = {
    'Accuracy': get_scorer('accuracy'),
    'sensitivity': get_scorer('recall'),
    'specificity': make_scorer(recall_score, pos_label=0),
    'Balanced_Accuracy': get_scorer('balanced_accuracy'),
    'AUC': get_scorer('roc_auc'),
    'Kappa': make_scorer(cohen_kappa_score),
    'F1': get_scorer('f1')
}


def calculate_score(y_t, y_p):
    sc_tb = pd.DataFrame()
    for sc_name, sc in scoring_dict.items():
        if sc_name == 'specificity':
            p = sc._kwargs
            p['y_true'] =y_t
            p['y_pred'] = y_p
            res = sc._score_func(**p)
        else:
            res = sc._score_func(y_t, y_p)

        df_temp = pd.DataFrame({'score_name': [sc_name],
                                'score': [res]})
        sc_tb = pd.concat([sc_tb, df_temp])
        print(sc_name)

    return sc_tb


def grid_info(grid, m):
    """
    取得 grid search info

    :param grid: object, gird sarch
    :param m: string, model name
    :return: data.frame
    """

    df_info = pd.DataFrame(grid.cv_results_)
    df_info = df_info[['params', 'mean_test_score',
                       'rank_test_score']]
    df_info['model'] = m
    return df_info


def customize_classifier(X, y, model_name, model_params='',
                         search_method='grid',
                         score='AUC',
                         imbalance=False,
                         imbalance_process={'over': {'sampling_strategy': 0.5},
                                            'under': {'sampling_strategy': 1}},
                         val_score=False):
    """
    classifier estimator

    :param X: DataFrame/array, x variable data
    :param y: DataFrame/1.d array, y variable data only 0,1
    :param model_name: list, need train model method name
    :param search_method: string, search_method name grid/bayes
    :param score: string, score name
    :param model_params: dict, customize model params
    :param imbalance: boolean, imbalance adjust
    :param imbalance_process: dict, imbalance process
    :param val_score: boolean, return val score
    :return: best_est: object, best model,
             res:DataFrame, grid search detail info
             val info :DataFrame, each val score
    """

    # method dict
    method_dict = {
        "tree": tree.DecisionTreeClassifier(),
        "rf": RandomForestClassifier(),
        "gbm": GradientBoostingClassifier(random_state=0),
        "xgb": XGBClassifier(
            colsample_bytree=0.8,
            objective='binary:logistic',
            scale_pos_weight=1,
            eval_metric='error',
            use_label_encoder=False),
        "lgb": lgb.LGBMClassifier(),
        "lg": LogisticRegression(solver='saga')
    }

    # grid Search dict
    grid_dict = {
        "tree": {
            "step1": {"model__max_depth": [3, 5]}
        },
        "rf": {
            "step1": {"model__max_depth": [3, 5]}
        },
        "gbm": {
            "step1": {"model__n_estimators": [10, 30, 50]},
            "step2": {"model__max_depth": [3, 5]}
        },
        "xgb": {
            "step1": {"model__max_depth": [3, 5],
                      "model__min_child_weight": [1e-1, 1, 1e1]},
            "step2": {"model__gamma": [0, 0.2]},
            "step3": {"model__subsample": [0.8, 0.9]},
            "step4": {"model__reg_alpha": [0, 1e-1, 1, 2]},
            "step5": {"model__learning_rate": [0.1, 1]}
        },
        "lgb": {
            "step1": {"model__num_leaves": [5, 10, 20]},
            "step2": {"model__min_child_weight": [1e-1, 1, 1e1]},
            "step3": {"model__subsample": [0.5, 0.8, 0.9, 1],
                      "model__colsample_bytree": [0.5, 0.8, 0.9, 1]},
            "step4": {"model__reg_alpha": [0, 1e-1, 1]}
        },
        "lg": {
            "step1": {"model__penalty": ["l2"]}
        }
    }

    # Bayes Search dict
    bayes_dict = {
        "tree": {
            "step1": {"model__max_depth":  Integer(1, 15),
                      "model__min_samples_split": Integer(2, 15),
                      "model__min_samples_leaf": Integer(1, 10)}
        },
        "rf": {
            "step1": {"model__max_depth": Integer(1, 15),
                      "model__n_estimators": Integer(100, 500)}
        },
        "gbm": {
            "step1": {"model__n_estimators": Integer(10, 60),
                      "model__max_depth": Integer(1, 15)}
        },
        "xgb": {
            "step1": {
                "model__max_depth": Integer(1, 15),
                "model__min_child_weight": Real(0.001, 10, prior='uniform'),
                "model__gamma": Real(0, 5, prior='uniform'),
                "model__subsample": Real(0.6, 1, prior='uniform'),
                "model__reg_alpha": Real(0, 3, prior='uniform'),
                "model__learning_rate": Real(0.001, 100, prior='uniform')}
        },
        "lgb": {
            "step1": {
                "model__num_leaves": Integer(5, 30),
                "model__min_child_weight": Real(0.001, 10, prior='uniform'),
                "model__subsample": Real(0.6, 1, prior='uniform'),
                "model__colsample_bytree": Real(0.6, 1, prior='uniform'),
                "model__reg_alpha": Real(0, 3, prior='uniform')}
        },
        "lg": {
            "step1": {
                "model__penalty": Categorical(["l2"]),
                "model__C": Real(1, 10, prior='uniform')}
        }
    }

    # imbalance
    imbalance_dict = {
        "over": SMOTE(sampling_strategy=0.1),
        "under":  RandomUnderSampler(sampling_strategy=1)
    }

    # set search params
    if model_params == '' and search_method == 'grid':
        model_params = {your_key: grid_dict[your_key] for your_key in model_name}
    elif search_method == 'bayes':
        model_params = {your_key: bayes_dict[your_key] for your_key in model_name}

    # balance adjust
    if imbalance:
        model_pipe = [(i, imbalance_dict[i].set_params(**j))
                      for i, j in imbalance_process.items()]
    else:
        model_pipe = []

    # change to boolean
    y = np.where(y == 1, True, False)

    # run search
    md_dict, cv_result = {}, pd.DataFrame()
    for i in model_name:
        print("========== " + i + " Begin ==========")
        md = model_pipe + \
             [('model', method_dict[i])]
        pipeline = Pipeline(steps=md)
        for step, param in model_params[i].items():
            # start time
            start = datetime.now()

            # grid search
            if search_method == 'grid':
                search = GridSearchCV(estimator=pipeline,
                                      param_grid=param,
                                      scoring=scoring_dict[score],
                                      cv=3)
            else:
                search = BayesSearchCV(estimator=pipeline,
                                       search_spaces=param,
                                       n_iter=70,
                                       cv=3,
                                       n_jobs=8,
                                       scoring=scoring_dict[score],
                                       random_state=123)
            search.fit(X, y)
            # end time
            res = grid_info(search, i)
            t = datetime.now() - start
            res['time'] = t
            # save
            cv_result = pd.concat([cv_result,
                                   res])

            # update
            pipeline.set_params(**search.best_params_)
            print(step + ": ",
                  search.best_params_, ", time:", t,
                  "==========")

        md_dict[i] = pipeline
        print(i + " Done ==========")

    # select best
    cv_result.reset_index(drop=True, inplace=True)
    best_score = cv_result['mean_test_score'].max()
    best_res = cv_result[
        cv_result['mean_test_score'] == best_score]
    best_md_name = best_res.loc[
        best_res['time'].idxmin(), 'model']

    # best model
    best_est = md_dict[best_md_name]

    # val score
    if val_score:
        val_info = cross_validate(best_est, X, y,
                                  scoring=scoring_dict,
                                  cv=5,
                                  return_train_score=True)
        val_info = pd.DataFrame(val_info)
    else:
        val_info = pd.DataFrame()

    # fit best model
    best_est.fit(X, y)

    # score
    sc = scoring_dict[score]

    # add all
    res = pd.concat(
        [cv_result,
         pd.DataFrame({'params': [best_est.get_params()],
                       'mean_test_score': [sc(best_est['model'], X, y)],
                       'rank_test_score': ['all'],
                       'model': [best_md_name]})])

    return best_est, res, val_info


if __name__ == '__main__':
    # sample
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.9],
        flip_y=0,
        random_state=1)

    X = pd.DataFrame(X)

    best_m, report, val_tb = customize_classifier(
        X=X,
        y=y,
        model_name=['lg'],
        model_params='',
        search_method='bayes',
        score='Balanced_Accuracy',
        imbalance=True,
        imbalance_process={'under': {'sampling_strategy': 1}},
        val_score=True
    )
