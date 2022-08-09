#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220809

import pandas as pd

import src.tool.shap as shap

params = {
    'data_path': 'D:/測試資料/PREDICTED_DATA_20220621/PRED_DATA_D195_11093.csv',
    'fea_path': 'D:/測試資料/fea_comb_file_test1.csv',
    'model': ['rf'],
    'fea_group': 'fea_comb3',
    'target': 'NPC_D'}

# step 1 read data
df = pd.read_csv(params['data_path'])

# step 2 prepare x variable
df_fea = pd.read_csv(params['fea_path'])
x_variable = df_fea[df_fea['Test_name'] ==
                    params['fea_group']]['Col'].to_list()

# step3 replace column name
d = params['data_path'].split('/')[-1].split('_')[-2]
df_col_name = list(df.columns)
df.columns = [x.replace(d + '_', '') for x in df_col_name]

# step 4 run shap
shap.run(X=df[x_variable],
         y=df[params['target']],
         model=params['model'],
         size=1000)
