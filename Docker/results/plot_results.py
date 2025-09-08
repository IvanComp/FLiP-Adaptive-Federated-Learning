import os


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bisect import bisect_left
from typing import List

import scipy.stats as ss
from scipy.stats import mannwhitneyu
import itertools


def VD_A(treatment: List[float], control: List[float]):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    :param treatment: a numeric list
    :param control: another numeric list
    :returns the value estimate and the magnitude
    """
    m = len(treatment)
    n = len(control)

    if m != n:
        raise ValueError("Data d and f must have the same length")

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude


metric = {('selector', 'same'): 'F1 Score Over Total Time for FL Round',
          ('selector', 'new'): 'F1 Score Over Total Time for FL Round',
          ('hdh', 'same'): 'Val F1', ('hdh', 'new'): 'Val F1'}

label_dict = {'selector': ['never', 'random', 'all-high', 'fixed', 'tree', 'bo'],
              'hdh': ['never', 'random', 'round1', 'fixed', 'tree', 'bo']}

patterns = ['selector', 'hdh']
persistences = ['same', 'new']
iid_percentages = [100, 0]
pairs = [(3, 3), (5, 5), (10, 10), (2, 4), (4, 2), (4, 8), (8, 4), (2, 8)]

selected_confs = ['no-{}', 'random-{}', 'always-{}', 'fixed-{}', 'tree-{}', 'bo-{}']

filter_1 = (lambda tup: tup[0] == tup[1], 'Nhigh-eq-Nlow', '$\mathsf{N_{high}}=\mathsf{N_{low}}$')
filter_2 = (lambda tup: tup[0] > tup[1], 'Nhigh-gt-Nlow', '$\mathsf{N_{high}}>\mathsf{N_{low}}$')
filter_3 = (lambda tup: tup[0] < tup[1], 'Nhigh-lt-Nlow', '$\mathsf{N_{high}}<\mathsf{N_{low}}$')

filters = [filter_1, filter_2, filter_3]

setups = list(itertools.product(patterns, persistences, iid_percentages, filters))


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
                df['F1 Score Over Total Time for FL Round'] = df['Val F1'] / df['Cumulative Time']
                model_data = df[df['Val F1'] >= 0]
                exp_data.append((folder + '/' + exp, model_data))
    return exp_data


def plot_by_filter(pattern, persistence, iid_percentage, filter):
    exp_data = []
    for pair in pairs:
        if filter[0](pair):
            exp_data.extend(get_exp_data(pair[0], pair[1], iid_percentage, persistence))

    colors = ['red', 'purple', 'green', 'blue', 'orange', 'pink'][:len(selected_confs)]
    labels = label_dict[pattern][:len(selected_confs)]
    d = []
    for conf in selected_confs:
        d.append([])
        data = [model_data[metric[(pattern, persistence)]].tolist() for exp, model_data in exp_data if
                exp.split('/')[-1].split('_')[0] == conf.format(pattern)]
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


# GENERATES BOX PLOTS
for setup in setups:
    if setup[0] == 'hdh' and setup[2] == 100:
        continue
    print(f'Generating box plot for {setup[0]}, {setup[1]}, {setup[2]}, {setup[3][1]}')
    plot_by_filter(setup[0], setup[1], setup[2], setup[3])


def run_statistical_tests(pattern, persistence, iid_percentage, filter):
    exp_data = []
    for pair in pairs:
        if filter[0](pair):
            exp_data.extend(get_exp_data(pair[0], pair[1], iid_percentage, persistence))

    d = []
    for conf in selected_confs:
        d.append([])
        data = [model_data[metric[(pattern, persistence)]].tolist() for exp, model_data in exp_data if
                exp.split('/')[-1].split('_')[0] == conf.format(pattern)]
        for i in range(len(data)):
            d[-1].append(data[i][-1])

    effect_size = {'negligible': 'negl.', 'small': 'small', 'medium': 'med.', 'large': 'large'}

    with open('plots/exp/{}/{}-{}-{}-VD_A.txt'.format(persistence, pattern, filter[1], iid_percentage), 'w') as f:
        # 'no-', 'random-', 'all-high-', 'fixed-', 'tree-', 'bo-'
        conf_to_compare = [(0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4), (0, 5), (1, 5), (2, 5)]
        latex_str = f"\n{filter[2]} & {iid_percentage}"
        for conf_pair in conf_to_compare:
            d_1 = d[conf_pair[0]]
            d_2 = d[conf_pair[1]]
            try:
                U1, p = mannwhitneyu(d_1, d_2, method="auto")
                estimate, magnitude = VD_A(d_1, d_2)
                conf_a = selected_confs[conf_pair[0]].split('/')[-1].replace('{}iid-'.format(iid_percentage), '')
                conf_b = selected_confs[conf_pair[1]].split('/')[-1].replace('{}iid-'.format(iid_percentage), '')
                f.write('{}\t{}\t{:.3f}\t{}\t{}\n'.format(conf_a, conf_b, p, estimate, magnitude))
                if p < 0.05:
                    if np.mean(d_1) < np.mean(d_2):
                        latex_str += f" & \\better{{<0.05 ({effect_size[magnitude]})}}"
                    else:
                        latex_str += f" & \\worse{{<0.05 ({effect_size[magnitude]})}}"
                else:
                    latex_str += f" & {p:.2f} ({effect_size[magnitude]})"
            except ValueError:
                print('Selected conf. do not have the same number of replications.')
                latex_str += " & TODO"
        f.write(latex_str + '\\\\\n')


# PERFORMS STATISTICAL TESTS AND GENERATES LATEX TABLE
for setup in setups:
    if setup[0] == 'hdh' and setup[2] == 100:
        continue

    print(f'Performing statistical tests for {setup[0]}, {setup[1]}, {setup[2]}, {setup[3][1]}')
    run_statistical_tests(setup[0], setup[1], setup[2], setup[3])
