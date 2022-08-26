#!/usr/bin/python
# -*- coding: utf-8 -*-
# ver1.0 create @20220826

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot(df, value_name, title, r=[]):

    # data transfer
    harvest = df.to_numpy()

    # plot
    f_size = 15
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.rcParams['font.size'] = f_size
    if len(r) == 0:
        im = ax.imshow(harvest, cmap='YlGnBu')
    else:
        im = ax.imshow(harvest, cmap='YlGnBu',
                       vmin=-4, vmax=4)

    c_bar = ax.figure.colorbar(im, ax=ax)
    c_bar.ax.set_ylabel(value_name,
                        rotation=-90, va="bottom")

    # Show all ticks and label them
    # with the respective list entries
    ax.set_xticks(
        np.arange(len(df.columns)),
        labels=df.columns,
        fontsize=f_size)

    ax.set_yticks(np.arange(len(df.index)),
                  labels=df.index,
                  fontsize=f_size)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    m = (np.min(harvest) + np.max(harvest)) / 2
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            c = 'w' if harvest[i, j] >= m else 'black'
            text = ax.text(j, i, harvest[i, j],
                           ha="center",
                           va="center", color=c)

    ax.set_title(title)
    fig.tight_layout()


if __name__ == '__main__':

    # test
    vegetables = ["cucumber", "tomato", "lettuce",
                  "asparagus",
                  "potato", "wheat", "barley"]
    farmers = ["Farmer Joe", "Upland Bros.",
               "Smith Gardening",
               "Agrifun", "Organiculture", "BioGoods Ltd.",
               "Cornylee Corp."]

    value = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                      [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                      [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                      [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                      [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                      [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                      [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

    df_test = pd.DataFrame(value)
    df_test.columns = vegetables
    df_test.index = farmers

    plot(df=df_test,
         value_name='value',
         title='Test HeatMap',
         r=[])
