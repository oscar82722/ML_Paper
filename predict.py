#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @202200811

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import joblib

import src.estimator.classifier as clf
import src.tool.km_plt as km


def run(params):

    # step 0 create folder
    Path(params['output_folder'] + '/predict').mkdir(
        parents=True, exist_ok=True)

    # step 1 read data
    df = pd.read_csv(params['data'])
    print('    Read Data Done')

    # step 2 prepare x variable
    df_fea = pd.read_csv(params['fea'])
    x_variable = df_fea['Col'].to_list()
    print('    Read Feature Done')

    # step3 read model
    md = joblib.load(params['model'])
    print('    Read Feature Done')

    # step4 predict
    y_hat = md.predict_proba(df[x_variable])
    y_hat = y_hat[:, 1]
    df['predict_prob'] = y_hat
    df['predict'] = md.predict(df[x_variable])
    print('    Predict Done')

    # evaluation
    df['Level'] = pd.cut(df['predict_prob'],
                         params['prob'])
    df.to_csv(params['output_folder'] +
              '/predict/predict.csv', index=False)

    # km plot
    p = {
        "df": df,
        "time_col": params["time_col"],
        "event_col": params["target"],
        "group": "Level"
    }
    km.plot(**p)
    plt.savefig(params['output_folder'] +
                '/predict/KM.png')
    print('    KM Curve Done')

    # calculate score
    sc_tb = clf.calculate_score(
        y_t=df[params["target"]],
        y_p=df['predict'])
    sc_tb.to_csv(params['output_folder'] +
                 '/predict/score.csv', index=False)
    print('    Score Done')


if __name__ == '__main__':
    pred_params = {
        "data": 'D:/test_data/test.csv',
        "fea": "D:/test_data/result/fea/D105__fea_comb1__all.csv",
        "model": "D:/test_data/result/model/D105__fea_comb1__all__rf.sav",
        "target": "NPC_D",
        "prob": [0, 0.5, 0.6, 1],
        "time_col": ['INDEX_TIME', 'END_TIME'],
        "output_folder": "D:/test_data/result/"
    }

    run(params=pred_params)
