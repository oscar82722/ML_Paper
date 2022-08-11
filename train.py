#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220811

import src.estimator.classifier as clf
import src.tool.fea_select as f_s
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import os


train_params = {
    'train_data_folder': 'E:/NPC/Train/',
    'date_group': ['D105', 'D135', 'D165', 'D195', 'D375'],
    'feature_file': 'C:/Users/admin/Desktop/ML_Paper/params/npc/fea_comb_file_2.csv',
    'fea_group': {
        'fea_comb1': ['all'],
        'fea_comb2': ['all'],
        'fea_comb3': ['all'],
        'fea_comb4': ['all'],
        'fea_comb5': ['all'],
        'fea_comb6': ['all']
    },
    'target': 'NPC_D',
    'model': ['lg', 'tree', 'rf', 'xgb', 'lgb'],
    'output_folder': 'C:/Users/admin/Desktop/ML_Paper/result/NPC_20220811/',
    'fea_plot_score': 'AUC'
}


# step1. get train folder data
file_v = os.listdir(train_params['train_data_folder'])

# step2. read feature file
fea_file = pd.read_csv(train_params['feature_file'])

# step3. set output folder
for i in ['/model', '/fea', '/plot']:
    f = train_params['output_folder'] + i
    Path(f).mkdir(parents=True, exist_ok=True)


# step4. run
df_report = pd.DataFrame()
df_val_tb = pd.DataFrame()
train_folder = train_params['train_data_folder']
for d_group in train_params['date_group']:
    print('date group: ', d_group)
    # get date group df
    train_path = [x for x in file_v if d_group in x]
    if len(train_path) > 1:
        break
    else:
        train_file = pd.read_csv(train_folder + '/' + train_path[0])

    for f_name, fea_method_v in train_params['fea_group'].items():
        print('    fea group: ', f_name)
        fea_all = fea_file[fea_file['Test_name'] == f_name]['Col'].to_list()
        for fea_method in fea_method_v:
            print('        fea method: ', fea_method)
            fea_now = f_s.fea_select(X=train_file[fea_all],
                                     y=train_file[train_params['target']],
                                     method=fea_method)
            # save select fea
            fea_select_path = train_params['output_folder'] + '/fea/' + d_group + '__' + f_name + '__' + \
                              fea_method + '.csv'
            fea_out = pd.DataFrame({'Col': fea_now})
            fea_out.to_csv(fea_select_path, index=False)

            for md in train_params['model']:
                print('            model: ', md)
                best_m, report, val_tb = clf.customize_classifier(
                    X=train_file[fea_now],
                    y=train_file[train_params['target']],
                    model_name=[md],
                    model_params='',
                    imbalance=True,
                    val_score=True
                )

                # save model
                md_out_path = train_params['output_folder'] + '/model/' + d_group + '__' + f_name + '__' + \
                              fea_method + '__' + md + '.sav'
                joblib.dump(best_m, md_out_path)

                # save report
                report['date_group'] = d_group
                report['fea_group'] = f_name
                report['fea_method'] = fea_method
                report['model'] = md

                # save val
                val_tb.index = val_tb.index.set_names(['val_time'])
                val_tb['date_group'] = d_group
                val_tb['fea_group'] = f_name
                val_tb['fea_method'] = fea_method
                val_tb['model'] = md
                val_tb = val_tb.reset_index()

                # update
                df_report = pd.concat([df_report, report])
                df_val_tb = pd.concat([df_val_tb, val_tb])

df_report.to_csv(train_params['output_folder'] + '/report.csv', index=False)
df_val_tb.to_csv(train_params['output_folder'] + '/val_tb.csv', index=False)

# step5. evaluation
id_col = ['date_group', 'fea_group', 'fea_method', 'model']
s = ['Accuracy', 'sensitivity', 'specificity', 'specificity',
     'Balanced_Accuracy', 'AUC', 'Kappa', 'F1']
df_val = pd.read_csv(train_params['output_folder'] + '/val_tb.csv')
df_m = pd.melt(df_val, id_vars=id_col, value_vars=['test_' + x for x in s] + ['train_' + x for x in s])
df_m[['train_test', 's']] = df_m['variable'].str.split('_', 1, expand=True)
df_ev = df_m.groupby(['date_group', 'fea_group', 'fea_method',
                      'model', 'train_test', 's'])['value'].mean().reset_index()
df_ev.to_csv(train_params['output_folder'] + '/evaluation.csv', index=False)

# step6. plot heatmap
for d_group in train_params['date_group']:
    for t in ['train', 'test']:
        df_plot = df_ev[(df_ev['date_group'] == d_group) &
                        (df_ev['s'] == train_params['fea_plot_score']) &
                        (df_ev['train_test'] == t)].copy()
        df_plot['fea'] = df_plot['fea_group'] + '__' + df_plot['fea_method']
        df_plot['value'] = np.round(df_plot['value'], 3)

        df_plot = df_plot.pivot(index='model', columns='fea', values='value')
        harvest = df_plot.to_numpy()
        # plot
        fig, ax = plt.subplots()
        im = ax.imshow(harvest, cmap='YlGnBu')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(train_params['fea_plot_score'], rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(df_plot.columns)), labels=df_plot.columns)
        ax.set_yticks(np.arange(len(df_plot.index)), labels=df_plot.index)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        m = np.median(harvest)
        for i in range(len(df_plot.index)):
            for j in range(len(df_plot.columns)):
                c = 'w' if harvest[i, j] > m else 'black'
                text = ax.text(j, i, harvest[i, j],
                               ha="center", va="center", color=c)

        ax.set_title("Feature Selection(" + d_group + ')')
        fig.tight_layout()
        plt.show()
        plt.savefig(train_params['output_folder'] + '/plot/fea_select__' + d_group + '__' + t + '.png')
        print('date_group: ', d_group, '_', t, 'done')

