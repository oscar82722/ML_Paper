#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220809

from sklearn.ensemble import GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from datetime import datetime
from sklearn import tree
import lightgbm as lgb
import pandas as pd


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
                         imbalance=False):
    """
    classifier estimator

    :param X: DataFrame/array, x variable data
    :param y: DataFrame/1.d array, y variable data only 0,1
    :param model_name: list, need train model method name
    :param model_params: dict, customize model params
    :param imbalance: boolean, imbalance adjust
    :return: best_est: object, best model,
             res:DataFrame, grid search detail info
    """

    # model method
    model_dict = {

        "tree": {"model": tree.DecisionTreeClassifier(),
                 "params": {
                     "step1": {'model__max_depth': [3, 5]}}
                 },

        "rf": {"model": RandomForestClassifier(),
               "params": {
                   "step1": {'model__max_depth': [3, 5]}}
               },

        "gbm": {"model": GradientBoostingClassifier(
            random_state=0),
                "params": {"step1": {
                    'model__n_estimators': [10, 30, 50]},
                           "step2": {
                               'model__max_depth': [3, 5]}}
                },

        "xgb": {"model": XGBClassifier(
            colsample_bytree=0.8,
            objective='binary:logistic',
            scale_pos_weight=1,
            eval_metric='error',
            use_label_encoder=False),
            "params": {
                    'step1': {'model__max_depth': [3, 5],
                              'model__min_child_weight': [
                                  1e-1, 1, 1e1]},
                    'step2': {'model__gamma': [0, 0.2]},
                    'step3': {
                        'model__subsample': [0.8, 0.9]},
                    'step4': {
                        'model__reg_alpha': [0, 1e-1, 1,
                                             2]},
                    'step5': {
                        'model__learning_rate': [0.1, 1]}}
                },

        "lgb": {"model": lgb.LGBMClassifier(),
                "params": {
                    'step1': {
                        'model__num_leaves': [5, 10, 20]},
                    'step2': {
                        'model__min_child_weight': [
                            1e-1, 1, 1e1]},
                    'step3': {
                        'model__subsample': [
                            0.5, 0.8, 0.9, 1],
                        'model__colsample_bytree': [
                            0.5, 0.8, 0.9, 1]},
                    'step4': {
                        'model__reg_alpha': [0, 1e-1, 1]}}
                }

    }

    # set params
    if model_params == '':
        model_params = {your_key: model_dict[your_key][
            'params'] for your_key in model_name}

    # balance adjust
    if imbalance:
        model_pipe = [
            ('over', SMOTE(sampling_strategy=0.1)),
            ('under', RandomUnderSampler(
                sampling_strategy=0.5))]
    else:
        model_pipe = []

    # grid search
    md_dict, cv_result = {}, pd.DataFrame()
    for i in model_name:
        print("========== " + i + " Begin ==========")
        md = model_pipe + \
             [('model', model_dict[i]['model'])]
        pipeline = Pipeline(steps=md)
        for step, param in model_params[i].items():
            # start time
            start = datetime.now()

            # grid search
            gs = GridSearchCV(estimator=pipeline,
                              param_grid=param,
                              scoring='roc_auc',
                              cv=3)
            gs.fit(X, y)
            # end time
            res = grid_info(gs, i)
            t = datetime.now() - start
            res['time'] = t
            # save
            cv_result = pd.concat([cv_result,
                                   res])

            # update
            pipeline.set_params(**gs.best_params_)
            print(step + ": ",
                  gs.best_params_, ", time:", t,
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

    # fit best model
    best_est = md_dict[best_md_name]
    best_est.fit(X, y)

    # add all
    res = pd.concat(
        [cv_result,
         pd.DataFrame({'params': [best_est.get_params()],
                       'mean_test_score': [
                           roc_auc_score(
                               y, best_est.predict(X))],
                       'rank_test_score': ['all'],
                       'model': [best_md_name]})])

    return best_est, res


if __name__ == '__main__':
    # sample
    X, y = make_classification(
        n_samples=10000,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.1],
        flip_y=0,
        random_state=1)

    X = pd.DataFrame(X)

    best_m, report = customize_classifier(
        X=X,
        y=y,
        model_name=['tree', 'rf', 'gbm', 'xgb', 'lgb'],
        model_params=''
    )
