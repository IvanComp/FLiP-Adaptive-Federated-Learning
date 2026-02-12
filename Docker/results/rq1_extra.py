import itertools
import os
import random

import pandas as pd

metric = {('all', 'same'): 'Cumulative F1',
          ('all', 'new'): 'Cumulative F1',
          ('all-2', 'same'): 'Cumulative F1',
          ('all-2', 'new'): 'Cumulative F1',
          ('all-text', 'same'): 'Cumulative F1',
          ('all-text', 'new'): 'Cumulative F1',
          ('selector-text', 'same'): 'F1 Score Over Total Time for FL Round',
          ('selector-text', 'new'): 'F1 Score Over Total Time for FL Round',
          ('selector', 'same'): 'F1 Score Over Total Time for FL Round',
          ('selector', 'new'): 'F1 Score Over Total Time for FL Round',
          ('selector-2', 'same'): 'F1 Score Over Total Time for FL Round',
          ('selector-2', 'new'): 'F1 Score Over Total Time for FL Round',
          ('hdh', 'same'): 'Cumulative F1',
          ('hdh', 'new'): 'Cumulative F1',
          ('hdh-2', 'same'): 'Cumulative F1',
          ('hdh-2', 'new'): 'Cumulative F1',
          ('hdh-text', 'same'): 'Cumulative F1',
          ('hdh-text', 'new'): 'Cumulative F1',
          ('compressor-2', 'same'): 'Cumulative Time Without Training',
          ('compressor-2', 'new'): 'Cumulative Time Without Training',
          ('compressor-2-delay', 'same'): 'Cumulative Time Without Training',
          ('compressor-2-delay', 'new'): 'Cumulative Time Without Training',
          }

should_increase = ['F1 Score Over Total Time for FL Round', 'Val F1', 'Cumulative F1']
should_decrease = ['Cumulative Communication Time', 'Cumulative Training Time', 'Cumulative Time With HDH',
                   'Total Time With HDH', 'Cumulative Total Time', 'Cumulative Communication Bottleneck',
                   'Total Time of FL Round', 'Total Time Without Training',
                   'Cumulative Time Without Training']

metrics_for_all_conf = ["Cumulative F1", "Val F1",
                        "Cumulative Total Time", "Cumulative HDH Time",
                        "Cumulative Time Without Training"]

random.seed(10)

patterns = ['selector-2', 'hdh-2', 'compressor-2', 'all-2']
persistences = ['new']
iid_percentages = [0]
pairs = [(3, 3), (5, 5), (10, 10), (2, 4), (4, 2), (4, 8), (8, 4), (2, 8)]

selected_confs = ['no-{}', 'random-{}', 'always-{}', 'fixed-{}', 'tree-{}', 'bo-{}', 'online-{}']

filter_1 = (lambda tup: tup[0] == tup[1], 'Nhigh-eq-Nlow', '$\mathsf{N_{high}}=\mathsf{N_{low}}$')
filter_2 = (lambda tup: tup[0] > tup[1], 'Nhigh-gt-Nlow', '$\mathsf{N_{high}}>\mathsf{N_{low}}$')
filter_3 = (lambda tup: tup[0] < tup[1], 'Nhigh-lt-Nlow', '$\mathsf{N_{high}}<\mathsf{N_{low}}$')
filter_4 = (lambda tup: tup[0] > 0 and tup[1] > 0, 'any-Nhigh-Nlow', '$\\text{any}\\nhigh,\\nlow$')

filters = {
    'all-text': [filter_4],
    'all': [filter_4],
    'all-2': [filter_4],
    'selector-text': [filter_4],
    'selector': [filter_4],
    'selector-2': [filter_4],
    'hdh': [filter_4],
    'hdh-2': [filter_4],
    'hdh-text': [filter_4],
    'compressor': [filter_4],
    'compressor-2': [filter_4],
    'compressor-text': [filter_4],
    'compressor-delay': [filter_4],
    'compressor-2-delay': [filter_4],
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
                    df['Cumulative HDH Time'] = df['HDH Time'].cumsum() / (n_high + n_low)
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


def analyze_by_filter(pattern, persistence, iid_percentage, filter):
    exp_data = []
    for pair in pairs:
        if filter[0](pair):
            exp_data.extend(get_exp_data(pair[0], pair[1], iid_percentage, persistence))

    means = {}
    for conf in selected_confs:
        for metric in metrics_for_all_conf:
            values = []
            data = [
                model_data[metric].tolist()
                for exp, model_data in exp_data
                if exp.split('/')[-1].split('_')[0] == conf.format(pattern)
            ]
            for i in range(len(data)):
                values.append(data[i][-1])

            if len(values) > 0:
                means[(conf.format(pattern), metric)] = sum(values) / len(values)
            else:
                means[(conf.format(pattern), metric)] = 0

    for metric in metrics_for_all_conf:
        print('\n' + metric)
        for tup in means:
            if metric == tup[1]:
                print(f'{tup[0]}: {means[tup]:.5f}', end=', ')


# GENERATES BOX PLOTS
for setup in setups:
    exclude = (('all' in setup[0] and setup[2] == 100) or
               ('hdh' in setup[0] and setup[2] == 100))
    if exclude:
        continue

    print(f'\n\nAnalyzing data for {setup[0]}, {setup[1]}, {setup[2]}, {setup[3][1]}')
    analyze_by_filter(setup[0], setup[1], setup[2], setup[3])
