#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220809

import matplotlib.pyplot as plt
import pandas as pd
import joblib

import src.tool.shap as shap

params = {
    "data": "D:/測試資料/train_min/PRED_2M_DATA_D105.csv",
    "fea": 'D:/測試資料/result/fea/D105__fea_comb1__all.csv',
    "target": "NPC_D",
    "model": "D:/測試資料/result/model/D105__fea_comb1__all__tree.sav",
    "output": {
        "SHAP": 1,
        "Importance_plot": 1,
        "tableone": 1
    },
    "output_folder": "C:/Users/admin/Desktop/ML_Paper/result/NPC_20220811/plot/",
}

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
    shap.run(X=df[x_variable],
             y=df[params['target']],
             model=md,
             train=t,
             size=1000)
    plt.savefig(params['output_folder'] + '/shap.png')
    print('    ' + 'SHAP Done')

# step 4.2 run importance
if params['output']['Importance_plot']:
    importances = md['model'].feature_importances_
    feature_names = md['model'].feature_names_in_
    plot_data = pd.Series(importances, index=feature_names)
    plot_data = plot_data.sort_values(ascending=True)

    fig, ax = plt.subplots()
    plot_data.plot.barh()
    ax.set_title("Feature importances")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(params['output_folder'] +
                '/Importance_plot.png')
    print('    ' + 'Importance plot Done')

# step 4.3 tableone
if params['output']['tableone']:
    shap.run(X=df[x_variable],
             y=df[params['target']],
             model=md,
             train=t,
             size=1000)
    plt.savefig(params['output_folder'] + '/shap.png')
    print('    ' + 'SHAP Done')








