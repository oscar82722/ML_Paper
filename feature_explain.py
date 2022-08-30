#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220809

import matplotlib.pyplot as plt
from tableone import TableOne
from pathlib import Path
import pandas as pd
import joblib

import src.tool.shap_func as sp


def run(params):

    # create output folder
    Path(params['output_folder'] + '/explain').mkdir(
        parents=True, exist_ok=True)

    # step 1 read data
    df = pd.read_csv(params['data'])
    print('    ' + 'Read Data Done')

    # step 2 prepare x variable
    df_fea = pd.read_csv(params['fea'])
    x_variable = df_fea['Col'].to_list()
    print('    ' + 'Read Feature Done')

    # step3 read model
    md = joblib.load(params['model'])
    t = False
    print('    ' + 'Read Model Done')

    # step 4.1 run shap
    if params['output']['SHAP']:
        fig, ax = plt.subplots(figsize=(12, 8))
        sp.run(X=df[x_variable],
               y=df[params['target']],
               model=md,
               train=t,
               size=1000)
        plt.savefig(params['output_folder'] +
                    '/explain/shap.png')
        print('    ' + 'SHAP Done')

    # step 4.2 run importance
    if params['output']['Importance_plot']:
        importances = md['model'].feature_importances_
        feature_names = x_variable
        plot_data = pd.Series(importances,
                              index=feature_names)
        plot_data = plot_data.sort_values(ascending=True)

        fig, ax = plt.subplots()
        plot_data.plot.barh()
        ax.set_title("Feature importances")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.savefig(params['output_folder'] +
                    '/explain/Importance_plot.png')
        print('    ' + 'Importance plot Done')

    # step 4.3 tableone
    if params['output']['tableone']:
        tb1 = TableOne(df,
                       columns=x_variable,
                       categorical=df_fea[
                           df_fea['col_type'] == 'cate'][
                           'Col'].to_list(),
                       groupby=params['target'],
                       pval=True)
        tb1.to_csv(params['output_folder'] +
                   '/explain/table_one.csv')
        print('    ' + 'TableOne Done')


if __name__ == '__main__':
    explain_params = {
        "data": "D:/測試資料/train_min/PRED_2M_DATA_D105.csv",
        "fea": 'D:/測試資料/result/fea/D105__fea_comb1__all.csv',
        "target": "NPC_D",
        "model": "D:/測試資料/result/model/D105__fea_comb1__all__lgb.sav",
        "output": {
            "SHAP": 1,
            "Importance_plot": 1,
            "tableone": 1
        },
        "output_folder": "D:/測試資料/result/plot/",
    }
    run(explain_params)


