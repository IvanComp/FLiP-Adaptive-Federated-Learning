import os


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import matplotlib.pyplot as plt
import pandas as pd

pattern = 'selector'
persistence = 'new'
iid_percentage = 100
pairs = [(3, 3), (5, 5), (10, 10), (2, 4), (4, 2), (4, 8), (8, 4), (2, 8)]

filter_1 = (lambda tup: tup[0] == tup[1], 'Nhigh-eq-Nlow')
filter_2 = (lambda tup: tup[0] > tup[1], 'Nhigh-gt-Nlow')
filter_3 = (lambda tup: tup[0] < tup[1], 'Nhigh-lt-Nlow')

filters = [filter_1, filter_2, filter_3]


def get_exp_data(n_high, n_low, iid_percentage, data_persistence):
    ratio = '{}high-{}low'.format(n_high, n_low)
    folders = ['vm/{}/{}/{}iid'.format(data_persistence, ratio, iid_percentage)]
    exp_data = []

    for folder in folders:
        experiments = [subfolder for subfolder in os.listdir(os.getcwd() + '/' + folder)]
        for exp in experiments:
            exp_path = os.getcwd() + '/' + folder + '/' + exp
            if 'FLwithAP_performance_metrics.csv' in os.listdir(exp_path):
                df = pd.read_csv(exp_path + '/FLwithAP_performance_metrics.csv')
                df = df[df['Val F1'] >= 0]
                df['Cumulative Time'] = df['Total Time of FL Round'].cumsum()
                print(df)
                df['F1 Score Over Total Time for FL Round'] = df['Val F1'] / df['Cumulative Time']
                model_data = df[df['Val F1'] >= 0]
                exp_data.append((folder + '/' + exp, model_data))
    return exp_data


def plot_by_filter(filter):
    exp_data = []
    for pair in pairs:
        if filter[0](pair):
            exp_data.extend(get_exp_data(pair[0], pair[1], iid_percentage, persistence))

    selected_confs = ['no-selector', 'random-selector', 'always-selector', 'fixed-selector', 'tree-selector',
                      'bo-adaptive']

    metric = 'F1 Score Over Total Time for FL Round'
    colors = ['red', 'purple', 'green', 'blue', 'orange', 'pink'][:len(selected_confs)]
    labels = ['never', 'random', 'all-high', 'fixed', 'tree', 'bo'][:len(selected_confs)]
    d = []
    for conf in selected_confs:
        d.append([])
        data = [model_data[metric].tolist() for exp, model_data in exp_data if exp.split('/')[-1].split('_')[0] == conf]
        for i in range(len(data)):
            d[-1].append(data[i][-1])
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    meanprops = dict(marker='^', markerfacecolor='white', markeredgecolor='black', markersize=8)
    bp = ax.boxplot(d, labels=labels, patch_artist=True, showmeans=True, meanprops=meanprops)
    # fill with colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median in bp['medians']:
        median.set_color('black')
    # ax.set_title('{} high - {} low - {}% iid'.format(N_high, N_low, iid_percentage))
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_verticalalignment('bottom')  # Align text top to bottom position
    offset_text.set_position((-0.12, -1.0))  # (x, y) in axis coordinates; tweak x as needed
    fig.tight_layout()
    fig.savefig('plots/exp/{}/{}-{}-{}.png'.format(persistence, pattern, filter[1], iid_percentage), dpi=300)


for filter in filters:
    plot_by_filter(filter)
