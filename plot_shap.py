#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220809

import matplotlib.pyplot as plt
import pandas as pd
import joblib

import src.tool.shap as shap


def run(params):
    # step 1 read data
    df = pd.read_csv(params['data'])

    # step 2 prepare x variable
    df_fea = pd.read_csv(params['fea'])
    x_variable = df_fea['Col'].to_list()

    # step3 read model
    if isinstance(params['model'], list):
        md = params['model']
        t = True
    else:
        md = joblib.load(params['model'])
        t = False

    # step 4 run shap
    shap.run(X=df[x_variable],
             y=df[params['target']],
             model=md,
             train=t,
             size=1000)

    plt.savefig(params['output_folder'] + '/shap.png')


if __name__ == '__main__':

    test_params = {
        'data': 'E:/NPC/Train/PRED_2M_DATA_D105.csv',
        'fea': 'C:/Users/admin/Desktop/ML_Paper/result/NPC_20220811/fea/D105__fea_comb1__all.csv',
        'target': 'NPC_D',
        'model': 'C:/Users/admin/Desktop/ML_Paper/result/NPC_20220811/model/D105__fea_comb1__all__tree.sav',
        'output_folder': 'C:/Users/admin/Desktop/ML_Paper/result/NPC_20220811/plot/'
    }
