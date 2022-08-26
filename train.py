#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220811

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import os

import src.estimator.classifier as clf
import src.tool.fea_select as f_s
import src.tool.heatmap as hm


def run(params):
    if params['process'][0]:
        # step1. get train folder data
        file_v = os.listdir(params['train_data_folder'])

        # step2. read feature file
        fea_file = pd.read_csv(params['feature_file'])

        # step3. set output folder
        for i in ['/model', '/fea', '/plot']:
            f = params['output_folder'] + i
            Path(f).mkdir(parents=True, exist_ok=True)

        # step4. run
        train_folder = params['train_data_folder']
        df_report, df_val = pd.DataFrame(), pd.DataFrame()
        for d in params['data_group']:
            print('data group: ', d)
            # get date group df
            train_path = [x for x in file_v if d in x]
            if len(train_path) > 1:
                break
            else:
                train_file = pd.read_csv(
                    train_folder + '/' + train_path[0])

            for f_n, f_m in params['fea_group'].items():
                print('    fea group: ', f_n)
                fea_all = fea_file[
                    fea_file['Test_name'] ==
                    f_n]['Col'].to_list()

                for m in f_m:
                    print('        fea method: ', m)
                    # run fea method
                    fea_now = f_s.fea_select(
                        X=train_file[fea_all],
                        y=train_file[params['target']],
                        method=m)

                    # save select fea
                    fea_select_path = \
                        params['output_folder'] + \
                        '/fea/' + d + '__' + f_n + \
                        '__' + m + '.csv'
                    fea_out = pd.DataFrame({'Col': fea_now})
                    fea_out.to_csv(fea_select_path,
                                   index=False)

                    for md in params['model']:
                        print('            model: ', md)
                        md_params = params[
                            'model_params'].copy()

                        md_params['X'] = train_file[fea_now]
                        md_params['y'] = train_file[
                            params['target']]
                        md_params['model_name'] = [md]

                        # run model
                        best_m, report, val_tb = \
                            clf.customize_classifier(
                                **md_params)

                        # save model
                        md_out_path = \
                            params['output_folder'] + \
                            '/model/' + d + '__' + \
                            f_n + '__' + \
                            m + '__' + md + '.sav'
                        joblib.dump(best_m, md_out_path)

                        # save report
                        report[
                            ['data_group', 'fea_group',
                             'fea_method', 'model']] = \
                            d,  f_n, m, md

                        # save val
                        val_tb.index = \
                            val_tb.index.set_names(
                                ['val_time'])
                        val_tb = val_tb.reset_index()
                        val_tb[
                            ['data_group', 'fea_group',
                             'fea_method', 'model']] = \
                            d,  f_n, m, md

                        # update
                        df_report = pd.concat([
                            df_report, report])
                        df_val = pd.concat(
                            [df_val, val_tb])

        # save report
        df_report.to_csv(params['output_folder'] +
                         '/report.csv', index=False)
        df_val.to_csv(params['output_folder'] +
                      '/val_tb.csv', index=False)

        # step5. evaluation
        id_col = ['data_group', 'fea_group',
                  'fea_method', 'model']
        s = ['Accuracy', 'sensitivity',
             'specificity', 'specificity',
             'Balanced_Accuracy', 'AUC',
             'Kappa', 'F1']

        df_val = pd.read_csv(params['output_folder'] +
                             '/val_tb.csv')
        df_m = pd.melt(df_val,
                       id_vars=id_col,
                       value_vars=['test_' + x for x in s] +
                                  ['train_' + x for x in s])
        df_m[['train_test', 's']] = \
            df_m['variable'].str.split('_', 1, expand=True)
        df_ev = df_m.groupby(
            ['data_group', 'fea_group',
             'fea_method', 'model',
             'train_test', 's'])['value'].mean().\
            reset_index()

        df_ev.to_csv(params['output_folder'] +
                     '/evaluation.csv', index=False)

    if params['process'][1]:
        # step6. plot heatmap
        if params['plot_file'] != '':
            df_ev = pd.read_csv(params['plot_file'])

        for d in params['data_group']:
            for t in ['train', 'test']:
                df_plot = df_ev[
                    (df_ev['data_group'] == d) &
                    (df_ev['s'] ==
                     params['fea_plot_score']) &
                    (df_ev['train_test'] == t)].copy()

                df_plot['fea'] = \
                    df_plot['fea_group'] + '__' + \
                    df_plot['fea_method']

                df_plot['value'] = np.round(
                    df_plot['value'], 3)

                df_plot = df_plot.pivot(index='model',
                                        columns='fea',
                                        values='value')

                # plot
                hm.plot(
                    df=df_plot,
                    value_name=params['fea_plot_score'],
                    title="Feature Selection(" + d + ')',
                    r=[])

                plt.show()
                plt.savefig(
                    params['output_folder'] +
                    '/plot/fea_select__' +
                    d + '__' + t + '.png')

                print('data_group: ', d, '_', t, 'done')


if __name__ == '__main__':
    train_param = {
        # process [training, plot]
        "process": [1, 1],
        # data params
        "train_data_folder": "E:/NPC/Train/",
        "data_group": ["D105", "D135", "D165", "D195",
                       "D375"],

        # feature selection params
        "feature_file": "C:/Users/admin/Desktop/ML_Paper/params/npc/fea_comb_file_2.csv",
        "fea_group": {
            "fea_comb1": ["all"],
            "fea_comb2": ["all"],
            "fea_comb3": ["all"],
            "fea_comb4": ["all"],
            "fea_comb5": ["all"],
            "fea_comb6": ["all"]
        },

        # model params
        "target": "NPC_D",
        "model": ["lg", "tree", "rf", "xgb", "lgb"],
        "model_params": {
            "model_params": {
                "tree": {
                    "step1": {"model__max_depth": [3, 5]}
                },
                "rf": {
                    "step1": {"model__max_depth": [3, 5]}
                },
                "gbm": {
                    "step1": {
                        "model__n_estimators": [10, 30,
                                                50]},
                    "step2": {"model__max_depth": [3, 5]}
                },
                "xgb": {
                    "step1": {"model__max_depth": [3, 5],
                              "model__min_child_weight": [
                                  1e-1, 1, 1e1]},
                    "step2": {"model__gamma": [0, 0.2]},
                    "step3": {
                        "model__subsample": [0.8, 0.9]},
                    "step4": {
                        "model__reg_alpha": [0, 1e-1, 1,
                                             2]},
                    "step5": {
                        "model__learning_rate": [0.1, 1]}
                },
                "lgb": {
                    "step1": {
                        "model__num_leaves": [5, 10, 20]},
                    "step2": {
                        "model__min_child_weight": [1e-1, 1,
                                                    1e1]},
                    "step3": {
                        "model__subsample": [0.5, 0.8, 0.9,
                                             1],
                        "model__colsample_bytree": [0.5,
                                                    0.8,
                                                    0.9,
                                                    1]},
                    "step4": {
                        "model__reg_alpha": [0, 1e-1, 1]}
                },
                "lg": {
                    "step1": {"model__penalty": ["l2"]}
                }
            },
            "search_method": 'grid',
            "score": "Balanced_Accuracy",
            "imbalance": 1,
            "imbalance_process": {
                'under': {'sampling_strategy': 1}},
            "val_score": 1
        },

        # output params
        "output_folder": "C:/Users/admin/Desktop/ML_Paper/result/NPC_20220811/",

        # plot params
        "plot_file": "",
        "fea_plot_score": "AUC"
    }

    run(params=train_param)

