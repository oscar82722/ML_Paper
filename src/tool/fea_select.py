#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220809

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn import tree
import lightgbm as lgb
import pandas as pd
import numpy as np


def fea_select(X, y, method):

    select_method = {
        'tree': tree.DecisionTreeClassifier(),
        'rf': RandomForestClassifier(),
        'xgb': XGBClassifier(
            colsample_bytree=0.8,
            objective='binary:logistic',
            scale_pos_weight=1,
            eval_metric='error',
            use_label_encoder=False),
        'lgb': lgb.LGBMClassifier()
    }

    if method != 'all':
        # feature select
        selector = SelectFromModel(
            estimator=select_method[method],
            threshold='1.25*mean')
        selector.fit(X.values, np.ravel(y.values))
        keep_col = X.columns
        keep_col = list(keep_col[selector.get_support()])
    else:
        keep_col = list(X.columns)

    return keep_col


if __name__ == '__main__':
    # sample
    X, y = make_classification(
        n_samples=10000,
        n_features=10,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.5],
        flip_y=0,
        random_state=1)

    X = pd.DataFrame(X)
    X.columns = ['col_' + str(x) for x in range(X.shape[1])]
    y = pd.DataFrame({'target': y})
    for m in ['all', 'tree', 'rf', 'xgb', 'lgb']:
        print(m, fea_select(X, y, method=m))






