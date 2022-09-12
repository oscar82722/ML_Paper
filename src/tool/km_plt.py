#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220912

from lifelines import KaplanMeierFitter
import pandas as pd


def plot(df, time_col, event_col, group):
    # check time column and compute time
    if (len(time_col) == 2) and isinstance(time_col, list):
        for t in time_col:
            df[t] = pd.to_datetime(df[t])
        df['time'] = (df[time_col[1]] -
                      df[time_col[0]]).dt.days
        time_col = 'time'

    colours = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
    i = 0
    # plot
    for cat in sorted(df[group].unique(), reverse=True):
        idx = df[group] == cat
        kmf = KaplanMeierFitter()
        kmf.fit(df[idx][time_col],
                event_observed=df[idx][event_col],
                label=cat)
        p = kmf.plot_cumulative_density(label=cat,
                                        ci_show=False,
                                        c=colours[i])
        i += 1
    p.set_xlabel("Days")


if __name__ == '__main__':
    data = pd.read_csv('D:/test_data/test.csv')
    plot(df=data,
         time_col=['INDEX_TIME', 'END_TIME'],
         event_col='NPC_D',
         group='ID_S')
