#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @202200811

import pandas as pd
import joblib

predict_params = {
    'data': 'E:/NPC/Test/FORECAST_2M_DATA_D105.csv',
    'fea': 'C:/Users/admin/Desktop/ML_Paper/result/NPC_20220811/fea/D105__fea_comb1__all.csv',
    'model': 'C:/Users/admin/Desktop/ML_Paper/result/NPC_20220811/model/D105__fea_comb1__all__tree.sav',
    'output_folder': 'C:/Users/admin/Desktop/ML_Paper/result/NPC_20220811/'
}


# step 1 read data
df = pd.read_csv(predict_params['data'])

# step 2 prepare x variable
df_fea = pd.read_csv(predict_params['fea'])
x_variable = df_fea['Col'].to_list()

# step3 read model
md = joblib.load(predict_params['model'])

# step4 predict
y_hat = md.predict_proba(df[x_variable])
y_hat = y_hat[:, 1]
df['predict_prob'] = y_hat

# output
out_file_name = predict_params['data'].split('.')[0].split('/')[-1] + '__predict_prob.csv'
df.to_csv(predict_params['output_folder'] + '/' + out_file_name, index=False)
