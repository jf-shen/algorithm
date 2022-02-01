import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.figure(figsize=[8, 6])

from IPython import display

display.set_matplotlib_formats('svg')

import seaborn as sns

sns.set()


def get_reward(df, signal, label, a_min, a_max):
    """
    Args:
        df: dataframe containing signal and label
        signal: df columns
        label: df columns
        a_min: signal bucket lower bound
        a_max: signal bucket upper bound
    Return:
        mean(label) for rows satisfying: a_min <= signal < a_max
    """
    a_min = -np.inf if a_min is None else a_min
    a_max = np.inf if a_max is None else a_max
    df['is_candidate'] = (df[signal] >= a_min) & (df[signal] < a_max)
    return df[df['is_candidate']][label].mean()


def plot_reward(df, signal, label, bucket_num):
    signal_list = np.sort(df[signal])
    x_list = []
    y_list = []
    percentage_list = []

    for i in range(bucket_num):
        idx_min = int(i / float(bucket_num) * len(signal_list))
        idx_max = int((i + 1) / float(bucket_num) * len(signal_list))
        idx_max = min(idx_max, len(signal_list) - 1)

        x_min = signal_list[idx_min]
        x_max = signal_list[idx_max]

        y = get_reward(df, signal=signal, label=label, a_min=x_min, a_max=x_max)
        percentage_list.append(i / float(bucket_num) * 100)
        x_list.append(x_min)
        y_list.append(y)

    plt.title('[%s] average reward' % (signal))
    plt.scatter(percentage_list, y_list, s=10, label=signal)
    plt.xlabel(signal + ' percentage')
    plt.ylabel('reward')


def plot_rewards(df, signals, label, bucket_num):
    for signal in signals:
        signal_list = np.sort(df[signal])
        x_list = []
        y_list = []
        percentage_list = []

        for i in range(bucket_num):
            idx_min = int(i / float(bucket_num) * len(signal_list))
            idx_max = int((i + 1) / float(bucket_num) * len(signal_list))
            idx_max = min(idx_max, len(signal_list) - 1)

            x_min = signal_list[idx_min]
            x_max = signal_list[idx_max]
            y = get_reward(df, signal=signal, label=label, a_min=x_min, a_max=x_max)
            percentage_list.append(i / float(bucket_num) * 100)
            x_list.append(x_min)
            y_list.append(y)

        plt.scatter(percentage_list, y_list, s=10, label=signal)

    plt.legend(loc='lower right', labels=signals, frameon=True, edgecolor='black')
    plt.title('%s average reward' % signals)
    plt.xlabel('%s percentage' % signals)
    plt.ylabel('reward')



