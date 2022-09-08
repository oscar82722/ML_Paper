#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220809

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import pandas as pd

import src.estimator.classifier as c


def run(X, y, model, size, train=False, out_folder=''):
    """
    run shap
    :param X: DataFrame/array, x variable data
    :param y: DataFrame/1.d array, y variable data only 0,1
    :param model: list, need train model method name
    :param size: use to plot sample size
    :param train: boolean, need training model
    :param out_folder: str, save plot folder
    :return: None
    """
    import shap

    if train:
        best_md, result, val_info = c.customize_classifier(
            X=X,
            y=y,
            model_name=model)
    else:
        best_md = model

    # check size
    s = X.shape[0] if X.shape[0] < size else size
    shap_df = X.sample(n=s, random_state=1)

    # shape
    explainer = shap.TreeExplainer(best_md['model'])
    exp_res = explainer(shap_df)
    shap_v = exp_res.values
    if len(shap_v.shape) == 3:
        shap_v = shap_v.transpose()
        shap_v = shap_v[0].transpose()
        exp_res.values = shap_v

    # plot bar plot
    plt.subplots(figsize=(12, 12))
    shap.plots.bar(exp_res, max_display=100, show=False)
    plt.gcf().axes[-1].set_aspect('auto')
    plt.tight_layout()
    if out_folder != '':
        plt.savefig(out_folder + '/shap_bar.png')

    # plot
    fig, ax = plt.subplots(figsize=(12, 12))
    shap.summary_plot(
        shap_v,
        shap_df,
        feature_names=shap_df.columns, show=False)
    plt.gcf().axes[-1].set_aspect('auto')
    plt.tight_layout()
    if out_folder != '':
        plt.savefig(out_folder + '/shap.png')

    return best_md


if __name__ == '__main__':
    # sample
    X, y = make_classification(
        n_samples=5000,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.5],
        flip_y=0,
        random_state=1)

    X = pd.DataFrame(X)
    X.columns = ['a', 'b']
    for m in ['tree', 'xgb', 'rf', 'lgb']:
        run(X=X, y=y, model=[m],
            size=1000, train=True)


