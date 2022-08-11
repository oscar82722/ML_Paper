#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220809

from sklearn.datasets import make_classification
import pandas as pd
import shap

import src.estimator.classifier as c
shap.initjs()


def run(X, y, model, size, train=True):
    """
    run shap
    :param X: DataFrame/array, x variable data
    :param y: DataFrame/1.d array, y variable data only 0,1
    :param model: list, need train model method name
    :param size: use to plot sample size
    :param train: boolean, need training model
    :return: None
    """

    if train:
        best_md, result = c.customize_classifier(
            X=X,
            y=y,
            model_name=model)
    else:
        best_md = model

    # check size
    s = X.shape[0] if X.shape[0] > size else size
    shap_df = X.sample(n=s, random_state=1)

    # shape
    explainer = shap.TreeExplainer(best_md['model'])
    shap_values = explainer.shap_values(shap_df)
    shap_values = shap_values[1] if len(shap_values) == 2 \
        else shap_values

    # plot
    shap.summary_plot(
        shap_values,
        shap_df,
        feature_names=shap_df.columns, show=False)


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
    run(X=X, y=y, model=['lgb'], size=1000)


