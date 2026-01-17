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
    try:
        A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors
    except ZeroDivisionError:
        return 0, 'negligible'

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude


metric = {('all', 'same'): 'Cumulative F1',
          ('all', 'new'): 'Cumulative F1',
          ('all-text', 'same'): 'Cumulative F1',
          ('all-text', 'new'): 'Cumulative F1',
          ('selector-text', 'same'): 'F1 Score Over Total Time for FL Round',
          ('selector-text', 'new'): 'F1 Score Over Total Time for FL Round',
          ('selector', 'same'): 'F1 Score Over Total Time for FL Round',
          ('selector', 'new'): 'F1 Score Over Total Time for FL Round',
          ('hdh', 'same'): 'Cumulative F1',
          ('hdh', 'new'): 'Cumulative F1',
          ('hdh-text', 'same'): 'Cumulative F1',
          ('hdh-text', 'new'): 'Cumulative F1',
          ('compressor', 'same'): 'Cumulative Communication Time',
          ('compressor-2', 'same'): 'Total Time of FL Round',
          ('compressor-delay', 'same'): 'Cumulative Communication Time',
          ('compressor-text', 'same'): 'Cumulative Communication Time',
          ('compressor-text-delay', 'same'): 'Cumulative Communication Time',
          }

metric_per_round = {('all', 'same'): 'Cumulative F1',
                    ('all', 'new'): 'Cumulative F1',
                    ('all-text', 'same'): 'Cumulative F1',
                    ('all-text', 'new'): 'Cumulative F1',
                    ('selector-text', 'same'): 'F1 Score Over Total Time for FL Round',
                    ('selector-text', 'new'): 'F1 Score Over Total Time for FL Round',
                    ('selector', 'same'): 'F1 Score Over Total Time for FL Round',
                    ('selector', 'new'): 'F1 Score Over Total Time for FL Round',
                    ('hdh', 'same'): 'Cumulative F1',
                    ('hdh', 'new'): 'Cumulative F1',
                    ('hdh-text', 'same'): 'Cumulative F1',
                    ('hdh-text', 'new'): 'Cumulative F1',
                    ('compressor', 'same'): 'Total Time of FL Round',
                    ('compressor-2', 'same'): 'Total Time of FL Round',
                    ('compressor-delay', 'same'): 'Total Time of FL Round',
                    ('compressor-text', 'same'): 'Total Time of FL Round',
                    ('compressor-text-delay', 'same'): 'Total Time of FL Round'
                    }

should_increase = ['F1 Score Over Total Time for FL Round', 'Val F1', 'Cumulative F1']
should_decrease = ['Cumulative Communication Time', 'Cumulative Training Time', 'Cumulative Time With HDH',
                   'Total Time With HDH', 'Cumulative Total Time', 'Cumulative Communication Bottleneck']

label_dict = {('all', 'same'): ['never', 'random', 'all-high+once', r'$\mathrm{FliP_{rule}}$',
                                r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                r'$\mathrm{FliP_{online}}$'],
              ('all', 'new'): ['never', 'random', 'all-high+always', r'$\mathrm{FliP_{rule}}$',
                               r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                               r'$\mathrm{FliP_{online}}$'],
              ('all-text', 'same'): ['never', 'random', 'all-high+once', r'$\mathrm{FliP_{rule}}$',
                                     r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                     r'$\mathrm{FliP_{online}}$'],
              ('all-text', 'new'): ['never', 'random', 'all-high+always', r'$\mathrm{FliP_{rule}}$',
                                    r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                    r'$\mathrm{FliP_{online}}$'],
              ('selector-text', 'same'): ['never', 'random', 'all-high', r'$\mathrm{FliP_{rule}}$',
                                          r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                          r'$\mathrm{FliP_{online}}$'],
              ('selector-text', 'new'): ['never', 'random', 'all-high', r'$\mathrm{FliP_{rule}}$',
                                         r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                         r'$\mathrm{FliP_{online}}$'],
              ('selector', 'same'): ['never', 'random', 'all-high', r'$\mathrm{FliP_{rule}}$',
                                     r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                     r'$\mathrm{FliP_{online}}$'],
              ('selector', 'new'): ['never', 'random', 'all-high', r'$\mathrm{FliP_{rule}}$',
                                    r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                    r'$\mathrm{FliP_{online}}$'],
              ('hdh', 'same'): ['never', 'random', 'once', r'$\mathrm{FliP_{rule}}$',
                                r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                r'$\mathrm{FliP_{online}}$'],
              ('hdh', 'new'): ['never', 'random', 'always', r'$\mathrm{FliP_{rule}}$',
                               r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                               r'$\mathrm{FliP_{online}}$'],
              ('hdh-text', 'same'): ['never', 'random', 'once', r'$\mathrm{FliP_{rule}}$',
                                     r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                     r'$\mathrm{FliP_{online}}$'],
              ('hdh-text', 'new'): ['never', 'random', 'always', r'$\mathrm{FliP_{rule}}$',
                                    r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                    r'$\mathrm{FliP_{online}}$'],
              ('compressor', 'same'): ['never', 'random', 'always', r'$\mathrm{FliP_{rule}}$',
                                       r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                       r'$\mathrm{FliP_{online}}$'],
              ('compressor-2', 'same'): ['never', 'random', 'always', r'$\mathrm{FliP_{rule}}$',
                                         r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                         r'$\mathrm{FliP_{online}}$'],
              ('compressor-delay', 'same'): ['never', 'random', 'always', r'$\mathrm{FliP_{rule}}$',
                                             r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                             r'$\mathrm{FliP_{online}}$'],
              ('compressor-text', 'same'): ['never', 'random', 'always', r'$\mathrm{FliP_{rule}}$',
                                            r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                            r'$\mathrm{FliP_{online}}$'],
              ('compressor-text-delay', 'same'): ['never', 'random', 'always', r'$\mathrm{FliP_{rule}}$',
                                                  r'$\mathrm{FliP_{pred}}$', r'$\mathrm{FliP_{bo}}$',
                                                  r'$\mathrm{FliP_{online}}$'],
              }

patterns = ['compressor-2']
persistences = ['same', 'new']
iid_percentages = [100, 0]
pairs = [(3, 3), (5, 5), (10, 10), (2, 4), (4, 2), (4, 8), (8, 4), (2, 8)]

selected_confs = ['no-{}', 'random-{}', 'always-{}', 'fixed-{}', 'tree-{}', 'bo-{}', 'online-{}']

filter_1 = (lambda tup: tup[0] == tup[1], 'Nhigh-eq-Nlow', '$\mathsf{N_{high}}=\mathsf{N_{low}}$')
filter_2 = (lambda tup: tup[0] > tup[1], 'Nhigh-gt-Nlow', '$\mathsf{N_{high}}>\mathsf{N_{low}}$')
filter_3 = (lambda tup: tup[0] < tup[1], 'Nhigh-lt-Nlow', '$\mathsf{N_{high}}<\mathsf{N_{low}}$')
filter_4 = (lambda tup: tup[0] > 0 and tup[1] > 0, 'any-Nhigh-Nlow', '$\\text{any}\\nhigh,\\nlow$')

filters = {
    'all-text': [filter_4],
    'all': [filter_4],
    'selector-text': [filter_4],
    'selector': [filter_4],
    'hdh': [filter_4],
    'hdh-text': [filter_4],
    'compressor': [filter_4],
    'compressor-2': [filter_4],
    'compressor-text': [filter_4],
    'compressor-delay': [filter_4],
    'compressor-text-delay': [filter_4]
}

setups = []
for pattern in patterns:
    setups.extend(list(itertools.product([pattern], persistences, iid_percentages, filters[pattern])))


def get_exp_data(n_high, n_low, iid_percentage, data_persistence):
    ratio = '{}high-{}low'.format(n_high, n_low)
    folders = ['vm/{}/{}/{}iid'.format(data_persistence, ratio, iid_percentage)]
    exp_data = []

    for folder in folders:
        experiments = [subfolder for subfolder in os.listdir(os.getcwd() + '/' + folder)]
        for exp in experiments:
            if exp.startswith('.'):
                continue

            exp_path = os.getcwd() + '/' + folder + '/' + exp
            if 'FLwithAP_performance_metrics.csv' in os.listdir(exp_path):
                df = pd.read_csv(exp_path + '/FLwithAP_performance_metrics.csv')
                df['Cumulative Training Time'] = df['Training Time'].cumsum()
                df['Cumulative Communication Time'] = df['Communication Time'].cumsum()

                grouping_key = df.index // (n_high + n_low)
                # Use groupby() and transform() to get the max value for each group
                df['Communication Bottleneck'] = df.groupby(grouping_key)['Communication Time'].transform('max')
                df['Training Bottleneck'] = df.groupby(grouping_key)['Training Time'].transform('max')
                df['Communication Time per Round'] = df.groupby(grouping_key)['Communication Time'].transform('sum')

                try:
                    df['Cumulative HDH Time'] = df['HDH Time'].cumsum()
                except KeyError:
                    df['Cumulative HDH Time'] = 0

                df = df[df['Val F1'] >= 0]
                df['Cumulative F1'] = df['Val F1'].cumsum()
                df['Cumulative Total Time'] = df['Total Time of FL Round'].cumsum()

                df['Cumulative Communication Bottleneck'] = df['Communication Bottleneck'].cumsum()

                df['Total Time Without Training'] = df['Total Time of FL Round'] - df['Training Bottleneck']
                df['Cumulative Time Without Training'] = df['Total Time Without Training'].cumsum()

                df['Total Time With HDH'] = df['Cumulative Total Time'] + df['Cumulative HDH Time']
                df['F1 Score Over Total Time for FL Round'] = df['Val F1'] / df['Cumulative Total Time']
                df['F1 Score Over Total Time With HDH'] = df['Val F1'] / df['Total Time With HDH']
                df['F1 Over Total Training Time'] = df['Val F1'] / df['Cumulative Training Time']
                model_data = df[df['Val F1'] >= 0]
                exp_data.append((folder + '/' + exp, model_data))
    return exp_data


def filter_outliers_iqr(data):
    """
    Filters outliers from a list of lists using the IQR method.

    Args:
      data: A list of lists, where each sublist is a population.

    Returns:
      A new list of lists with outliers removed from each sublist.
    """
    filtered_data = []
    for sublist in data:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        q1, q3 = np.percentile(sublist, [10, 90])

        # Calculate the Interquartile Range (IQR)
        iqr = q3 - q1

        # Define the lower and upper bounds for outlier detection
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        # Filter the sublist to keep only the values within the bounds
        filtered_sublist = [x for x in sublist if lower_bound <= x <= upper_bound]
        filtered_data.append(filtered_sublist)

    return filtered_data


def plot_by_filter_and_round(pattern, persistence, iid_percentage, filter):
    exp_data = []
    for pair in pairs:
        if filter[0](pair):
            exp_data.extend(get_exp_data(pair[0], pair[1], iid_percentage, persistence))

    colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600', '#f5e979'][:len(selected_confs)]
    labels = label_dict[(pattern, persistence)][:len(selected_confs)]
    fig, ax = plt.subplots(figsize=(5, 3))

    baseline = [model_data[metric_per_round[(pattern, persistence)]].tolist() for exp, model_data in exp_data if
                exp.split('/')[-1].split('_')[0] == 'always-{}'.format(pattern)]
    means_baseline = [np.mean([exp[round] for exp in baseline]) for round in range(len(baseline[0]))]

    compared_confs = ['fixed-{}', 'tree-{}', 'bo-{}']
    for conf_i, conf in enumerate(compared_confs):
        data = [model_data[metric_per_round[(pattern, persistence)]].tolist() for exp, model_data in exp_data if
                exp.split('/')[-1].split('_')[0] == conf.format(pattern)]
        a, b, c = pattern, conf, persistence

        try:
            means = [np.mean([(exp[round] - means_baseline[round]) / means_baseline[round] * 100 for exp in data])
                     for round in range(1, len(data[0]))]

            stds = [np.std([(exp[round] - means_baseline[round]) / means_baseline[round] * 100 for exp in data]) for
                    round in range(1, len(data[0]))]

            means = np.array(means)
            stds = np.array(stds)
            ax.plot(range(2, len(data[0]) + 1), means, label=labels[selected_confs.index(conf)],
                    marker='o', markersize=2, color=colors[selected_confs.index(conf)])
        except:
            print('insufficient data')
    ax.plot(range(1, len(data[0]) + 1), [0] * len(data[0]), linestyle='--', color='black', linewidth=0.5)
    ax.set_xticks(range(1, len(data[0]) + 1))
    ax.set_xticklabels(range(1, len(data[0]) + 1))
    ax.set_xlim([2, len(data[0])])
    ax.grid(True, axis='y', linestyle='-', which='major', color='lightgrey',
            alpha=0.7, zorder=0)
    ax.legend(loc='best', fontsize=12)
    fig.savefig('plots/rq1/{}/{}-{}-{}-round.pdf'.format(persistence, pattern, filter[1], iid_percentage),
                dpi=300, bbox_inches='tight')


def adjacent_values(sorted_vals, q1, q3):
    """
    Compute Tukey whiskers (1.5 IQR rule).
    """
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr

    upper_adjacent = np.clip(upper, q3, sorted_vals[-1])
    lower_adjacent = np.clip(lower, sorted_vals[0], q1)

    return lower_adjacent, upper_adjacent


def draw_iqr_whiskers(ax, data, positions):
    """
    Draw:
    - dot at median
    - thick vertical line for IQR
    - thin vertical whiskers (Tukey)
    `data` is a list of lists with variable lengths.
    """

    for vals, pos in zip(data, positions):
        vals = np.asarray(vals)
        if len(vals) == 0:
            continue

        q1, median, q3 = np.percentile(vals, [25, 50, 75])

        lower, upper = adjacent_values(sorted(vals), q1, q3)

        # median dot
        ax.scatter(
            pos,
            median,
            marker='o',
            color='white',
            edgecolor='black',
            s=30,
            zorder=4
        )

        # IQR (thick)
        ax.vlines(
            pos,
            q1,
            q3,
            color='black',
            linestyle='-',
            lw=5,
            zorder=3
        )

        # whiskers (thin)
        ax.vlines(
            pos,
            lower,
            upper,
            color='black',
            linestyle='-',
            lw=1,
            zorder=2
        )


def plot_by_filter(pattern, persistence, iid_percentage, filter):
    exp_data = []
    for pair in pairs:
        if filter[0](pair):
            exp_data.extend(get_exp_data(pair[0], pair[1], iid_percentage, persistence))

    colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600', '#f5e979'][:len(selected_confs)]
    labels = label_dict[(pattern, persistence)][:len(selected_confs)]
    d = []
    for conf in selected_confs:
        d.append([])
        data = [model_data[metric[(pattern, persistence)]].tolist() for exp, model_data in exp_data if
                exp.split('/')[-1].split('_')[0] == conf.format(pattern)]
        for i in range(len(data)):
            d[-1].append(data[i][-1])
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'sans-serif'  # Often looks good with LaTeX
    plt.rcParams['font.serif'] = ['Helvetica']  # Specify LaTeX font
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    meanprops = dict(marker='^', markerfacecolor='white', markeredgecolor='black', markersize=8)
    bp = ax.boxplot(d, labels=labels, patch_artist=True, showmeans=True, meanprops=meanprops)
    # fill with colors
    for box in bp['boxes']:
        box.set(linewidth=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
    for median in bp['medians']:
        median.set_color('black')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7, zorder=0)
    if pattern == 'hdh-text' and iid_percentage == 0 and persistence == 'new':
        ax.set_ylim([8.0, 10.0])
    if pattern == 'selector-text' and iid_percentage == 0 and persistence == 'same':
        ax.set_ylim([1.8 * 10 ** -3, 2.65 * 10 ** -3])
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_verticalalignment('bottom')
    offset_text.set_position((-0.12, -1.0))
    fig.tight_layout()
    fig.savefig('plots/rq1/{}/{}-{}-{}.pdf'.format(persistence, pattern, filter[1], iid_percentage),
                dpi=300, bbox_inches='tight')

    d = []
    valid_labels = []
    valid_colors = []

    for conf, label, color in zip(selected_confs, labels, colors):
        values = []
        data = [
            model_data[metric[(pattern, persistence)]].tolist()
            for exp, model_data in exp_data
            if exp.split('/')[-1].split('_')[0] == conf.format(pattern)
        ]
        for i in range(len(data)):
            values.append(data[i][-1])

        if len(values) > 0:  # <-- key fix
            d.append(values)
            valid_labels.append(label)
            valid_colors.append(color)

    labels = valid_labels
    colors = valid_colors

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Create violin plot
    vp = ax.violinplot(
        d,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    # Color violins
    for body, color in zip(vp['bodies'], colors):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(1.0)
        body.set_linewidth(0.5)
        body.set_zorder(2)

    positions = np.arange(1, len(d) + 1)
    draw_iqr_whiskers(ax, d, positions)

    # Style median lines
    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(1.0)

    # Add mean markers manually
    means = [np.mean(vals) for vals in d]
    # ax.scatter(
    #    range(1, len(means) + 1),
    #    means,
    #    marker='^',
    #    color='white',
    #    edgecolor='black',
    #    s=60,
    #    zorder=4
    # )

    # X-axis labels
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    ax.yaxis.grid(
        True,
        linestyle='-',
        which='major',
        color='lightgrey',
        alpha=0.7,
        zorder=0
    )

    # Special-case limits (unchanged)
    if pattern == 'hdh-text' and iid_percentage == 0 and persistence == 'new':
        ax.set_ylim([8.0, 10.0])
    if pattern == 'selector-text' and iid_percentage == 0 and persistence == 'same':
        ax.set_ylim([1.8 * 10 ** -3, 2.65 * 10 ** -3])

    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_verticalalignment('bottom')
    offset_text.set_position((-0.12, -1.0))

    fig.tight_layout()
    fig.savefig(
        'plots/rq1/{}/violin_{}-{}-{}.pdf'.format(persistence, pattern, filter[1], iid_percentage),
        dpi=300,
        bbox_inches='tight'
    )


# GENERATES BOX PLOTS
for setup in setups:
    exclude = (('all' in setup[0] and setup[2] == 100) or
               ('hdh' in setup[0] and setup[2] == 100) or
               ('compressor' in setup[0] and setup[1] == 'new'))
    if exclude:
        continue

    print(f'Generating box plot for {setup[0]}, {setup[1]}, {setup[2]}, {setup[3][1]}')
    plot_by_filter(setup[0], setup[1], setup[2], setup[3])
    # plot_by_filter_and_round(setup[0], setup[1], setup[2], setup[3])


def plot_pattern_vs_all(pattern, persistence, iid_percentage, filter):
    """
    Bar plot comparing a single pattern vs the 'all' (selector + HDH) configuration.
    """
    if 'text' in pattern:
        patterns = [pattern, 'all-text']
    else:
        patterns = [pattern, 'all']
    data_by_pattern = []

    for p in patterns:
        exp_data = []
        for pair in pairs:
            if filter[0](pair):
                exp_data.extend(get_exp_data(pair[0], pair[1], iid_percentage, persistence))

        d = []
        for conf in selected_confs:
            if 'all' in p and 'no-' in conf:
                continue
            values = [
                model_data[metric[(pattern, persistence)]].tolist()[-1]
                for exp, model_data in exp_data
                if exp.split('/')[-1].split('_')[0] == conf.format(p)
            ]
            d.append(values)

        data_by_pattern.append(d)

    # ---- plotting ----
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['text.usetex'] = True

    fig, ax = plt.subplots(figsize=(6, 3))

    positions = []
    labels = []
    pos = 1

    for i, p in enumerate(patterns):
        for j in range(len(selected_confs)):
            if 'all' in p and 'no-' in selected_confs[j]:
                continue
            positions.append(pos)
            labels.append(label_dict[(p, persistence)][j])
            pos += 1
        pos += 1  # gap between pattern groups

    flat_data = [item for sublist in data_by_pattern for item in sublist]
    meanprops = dict(marker='^', markerfacecolor='white', markeredgecolor='black', markersize=8)

    bp = ax.boxplot(
        flat_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanprops=meanprops
    )

    colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600', '#f5e979']

    for patch, color in zip(bp['boxes'], colors + colors[1:]):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set_color('black')

    if pattern == 'hdh-text' and iid_percentage == 0 and persistence == 'new':
        ax.set_ylim([8.0, 10.0])

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.yaxis.grid(True, linestyle='-', alpha=0.7)

    fig.tight_layout()
    fig.savefig(
        f'plots/rq1/{persistence}/{pattern}-vs-all-{filter[1]}-{iid_percentage}.pdf',
        dpi=300,
        bbox_inches='tight'
    )

    # ---- plotting ----
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['text.usetex'] = True

    fig, ax = plt.subplots(figsize=(6, 3))

    labels = []
    pos = 1

    valid_data = []
    valid_positions = []
    valid_colors = []

    colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600', '#f5e979']
    color_idx = 0

    for i, p in enumerate(patterns):
        for j, values in enumerate(data_by_pattern[i]):
            if len(values) == 0:
                continue

            valid_data.append(values)
            valid_positions.append(pos)
            labels.append(label_dict[(p, persistence)][j])
            valid_colors.append(colors[color_idx % len(colors)])

            pos += 1
            color_idx += 1

        pos += 1  # gap between pattern groups

    # --- grid BELOW everything ---
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='-', alpha=0.7, zorder=0)

    # --- violin plot ---
    vp = ax.violinplot(
        valid_data,
        positions=valid_positions,
        widths=0.6,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    # style violins
    for body, color in zip(vp['bodies'], valid_colors):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(1.0)
        body.set_linewidth(0.5)
        body.set_zorder(2)

    draw_iqr_whiskers(ax, valid_data, valid_positions)

    # medians
    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(1.0)
    vp['cmedians'].set_zorder(3)

    # means (same as boxplot)
    means = [np.mean(vals) for vals in valid_data]
    # ax.scatter(
    #    valid_positions,
    #    means,
    #    marker='^',
    #    color='white',
    #    edgecolor='black',
    #    s=60,
    #    zorder=4
    # )

    # axis formatting
    if pattern == 'hdh-text' and iid_percentage == 0 and persistence == 'new':
        ax.set_ylim([8.0, 10.0])

    ax.set_xticks(valid_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    fig.tight_layout()
    fig.savefig(
        f'plots/rq1/{persistence}/violin_{pattern}-vs-all-{filter[1]}-{iid_percentage}.pdf',
        dpi=300,
        bbox_inches='tight'
    )


# MULTI-PATTERN COMPARISON PLOTS
for setup in setups:
    pattern, persistence, iid_percentage, filter = setup

    if pattern in ['selector', 'hdh', 'selector-text', 'hdh-text'] and iid_percentage == 0:
        print(f'Generating {pattern} vs all comparison plot')
        plot_pattern_vs_all(pattern, persistence, iid_percentage, filter)


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

    effect_size = {'negligible': 'N', 'small': 'S', 'medium': 'M', 'large': 'L'}

    with (open('plots/rq1/{}/{}-{}-{}-VD_A.txt'.format(persistence, pattern, filter[1], iid_percentage), 'w') as f):
        # 'no-', 'random-', 'all-high-', 'fixed-', 'tree-', 'bo-'
        conf_to_compare = [(0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4), (0, 5), (1, 5), (2, 5)]
        latex_str = f"\n{filter[2]} & {iid_percentage}"

        the_lower_the_better = metric[(pattern, persistence)] in should_decrease
        the_higher_the_better = metric[(pattern, persistence)] in should_increase

        for conf_pair in conf_to_compare:
            d_1 = d[conf_pair[0]]
            d_2 = d[conf_pair[1]]

            if len(d_1) != len(d_2):
                if the_higher_the_better:
                    d_1 = sorted(d_1)[:min(len(d_1), len(d_2))]
                    d_2 = sorted(d_2, reverse=True)[:min(len(d_1), len(d_2))]
                else:
                    d_1 = sorted(d_1, reverse=True)[:min(len(d_1), len(d_2))]
                    d_2 = sorted(d_2)[:min(len(d_1), len(d_2))]

            try:
                U1, p = mannwhitneyu(d_1, d_2, method="auto")
                estimate, magnitude = VD_A(d_1, d_2)
                conf_a = selected_confs[conf_pair[0]].split('/')[-1].replace('{}iid-'.format(iid_percentage), '')
                conf_b = selected_confs[conf_pair[1]].split('/')[-1].replace('{}iid-'.format(iid_percentage), '')
                f.write('{}\t{}\t{:.3f}\t{}\t{}\n'.format(conf_a, conf_b, p, estimate, magnitude))

                if p < 0.05:
                    if (the_higher_the_better and np.mean(d_2) > np.mean(d_1)) or (
                            the_lower_the_better and np.mean(d_2) < np.mean(d_1)):
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
    exclude = (('all' in setup[0] and setup[2] == 100) or
               ('hdh' in setup[0] and setup[2] == 100) or
               ('compressor' in setup[0] and setup[1] == 'new'))
    if exclude:
        continue

    print(f'Performing statistical tests for {setup[0]}, {setup[1]}, {setup[2]}, {setup[3][1]}')
    # run_statistical_tests(setup[0], setup[1], setup[2], setup[3])
