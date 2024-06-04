import pickle
from collections import defaultdict
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib import rcParams

all_cluster_types = ['lda', 'lr', 'DBSCAN', 'DBSCAN_half', 'DBSCAN_double', 'DBSCAN_triple', 'DBSCAN_quad', 'k_means',
                     'k_means_inc', 'k_means_dec', 'k_means_double', 'k_means_4', 'k_means_6', 'k_means_8', 'OPTICS',
                     'mean_shift', 'mean_shift_2', 'mean_shift_4', 'mean_shift_8']

redacted_cluster_types = ['lda', 'lr', 'DBSCAN', 'DBSCAN_double', 'k_means_6', 'k_means_8', 'OPTICS', 'mean_shift_4',
                          'mean_shift_8']


def get_statistics(exp_set, ambiguity_values, accuracy_values, cluster_types_to_keep, measurement_metric, min_accuracy,
                   network_type=None, keep_only_perfect_linear_separability=False):
    assert exp_set in {'regular', 'stackless', 'stackful', 'top_of_stack'}
    assert measurement_metric in {'amb', 'wamb', 'size'}
    measurement_index_dict = {'amb': 0, 'wamb': 1, 'size': 4}
    measurement_index = measurement_index_dict[measurement_metric]

    if network_type and network_type not in {'relu', 'tanh', 'lstm', 'gru'}:
        assert False

    filtered_measurements = defaultdict(list)

    for exp_name, values in ambiguity_values.items():
        # sort by network type and accuracy
        # TODO REVERT
        # if exp_set not in exp_name or accuracy_values[exp_name] < min_accuracy:
        #     continue
        # if enabled, keep only perfect those experiments with perfect linear separability
        if keep_only_perfect_linear_separability and (values['lda'][1] > 0 or values['lr'][1] > 0):
            continue

        # sort by network type
        if network_type and network_type not in exp_name:
            continue

        for c in cluster_types_to_keep:
            filtered_measurements[c].append(values[c][measurement_index])

    len_filtered = len(list(filtered_measurements.values())[0])
    if len_filtered == 0:
        print('No experiments meet filter requirements')
        return

    fig1, ax1 = plt.subplots()
    plt.xticks(rotation=90)

    print(f'Number of all experiments:      {len([x for x in ambiguity_values.keys() if exp_set in x])}')
    print(f'Number of filtered experiments: {len_filtered}')
    for i, (k, v) in enumerate(filtered_measurements.items()):
        ax1.boxplot(v, labels=[k], positions=[i+1], sym='x', flierprops={'alpha': 0.2})
        print(f'Cluster Type {k}: Avg {mean(v)}, Stddev: {stdev(v)}, Max: {max(v)}, Zeros: {v.count(0)}')

    plt.ylabel(measurement_metric)
    plt.title(f'Exp. set: {exp_set}, Metric: {measurement_metric}, Min Acc: {min_accuracy}')
    plt.show()

    import tikzplotlib
    tikzplotlib.save('pda_wamb_0.2_top_of_stack.tex')


def get_relationship_between_accuracy_and_wamb(accuracy_values, ambiguity_values):
    with open('experiment_results/additional_ambiguity_analysis.pickle', 'rb') as handle:
        data = pickle.load(handle)
        for k, v in data.items():
            key = k + 'regular' if 'CFG' not in k else k + 'stackless'
            ambiguity_values[key] = {'k_means_8': v}

    with open('experiment_results/additional_accurcy_analysis.pickle', 'rb') as handle:
        data = pickle.load(handle)
        for k, v in data.items():
            key = k + 'regular' if 'CFG' not in k else k + 'stackless'
            accuracy_values[key] = v[0]

    regular_feature, regular_target = [], []
    pda_feature, pda_target = [], []
    for k, v in ambiguity_values.items():
        if 'regular' in k:
            regular_feature.append([v['k_means_8'][1]])
            regular_target.append(accuracy_values[k])
        if 'stackless' in k:
            pda_feature.append([v['k_means_8'][1]])
            pda_target.append(accuracy_values[k])

    # Calculate the Spearman correlation coefficient
    spearman_corr, _ = scipy.stats.pearsonr(np.array(regular_feature).flatten(), regular_target)
    print("Pearson Correlation Coefficient (REG):", spearman_corr)

    spearman_corr, _ = scipy.stats.pearsonr(np.array(pda_feature).flatten(), pda_target)
    print("Pearson Correlation Coefficient (PDA):", spearman_corr)

    for exp_name, targets, features in [('regular', regular_target, regular_feature),
                                        ('pda', pda_target, pda_feature)]:
        # Perform linear regression
        slope, intercept = np.polyfit(np.array(targets).flatten(), np.array(features).flatten(), 1)
        # Create the regression line using the slope and intercept
        regression_line = slope * np.array(targets).flatten() + intercept

        plt.plot(targets, regression_line, color='red', label='Regression Line')
        plt.scatter(targets, features, alpha=0.05, s=10 * rcParams['lines.markersize'] ** 2)

        plt.xlabel('Accuracy')
        plt.ylabel('Wamb')
        plt.legend(loc='upper left')
        plt.title(f'RNNs trained on {exp_name}')
        plt.show()


if __name__ == '__main__':
    paper_results = True
    if paper_results:
        # values obtained with experiment_runner and used in the paper
        with open('experiment_results/new_accuracy_results_top_of_stack.pickle', 'rb') as handle:
            accuracy_values = pickle.load(handle)
        with open('experiment_results/ambiguity_results_top_of_stack.pickle', 'rb') as handle:
            ambiguity_values = pickle.load(handle)
    else:
        # in case you recompute all values with automated_trainer script
        with open('experiment_results/new_accuracy_results.pickle', 'rb') as handle:
            accuracy_values = pickle.load(handle)
        with open('experiment_results/new_ambiguity_results.pickle', 'rb') as handle:
            ambiguity_values = pickle.load(handle)

    # get weighted ambiguity and size plots for all experiments
    for mm in ['wamb', ]:
        # for exp_set in ['regular', 'stackful', 'stackless']:
            exp_set = 'top_of_stack'
            get_statistics(exp_set, ambiguity_values, accuracy_values,
                           cluster_types_to_keep=all_cluster_types,
                           measurement_metric=mm,
                           min_accuracy=0.8,
                           network_type=None,
                           keep_only_perfect_linear_separability=False)

    exit()
    # show only GRUs for regular languages
    get_statistics('regular', ambiguity_values, accuracy_values,
                   cluster_types_to_keep=all_cluster_types,
                   measurement_metric='wamb',
                   min_accuracy=0.8,
                   network_type='gru',
                   keep_only_perfect_linear_separability=False)

    # show only regural languages with perfect linear separability
    get_statistics('regular', ambiguity_values, accuracy_values,
                   cluster_types_to_keep=all_cluster_types,
                   measurement_metric='wamb',
                   min_accuracy=0.8,
                   network_type=None,
                   keep_only_perfect_linear_separability=True)

    # get_relationship_between_accuracy_and_wamb(accuracy_values, ambiguity_values)
