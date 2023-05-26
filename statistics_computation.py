import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import tikzplotlib


def read_measurement_picklefile(file_name, measurements):
    file_content = []
    with open(file_name, 'rb') as file:
        try:
            while True:
                file_content.append(pickle.load(file))
        except EOFError:
            pass
        file_content = file_content[0]
        # print(file_content)
        for entry in file_content:
            exp_name = entry[0]
            # print(exp_name)
            if exp_name_select and exp_name_select not in exp_name:
                continue
            if exp_name_exclude and exp_name_exclude in exp_name:
                continue
            results = entry[1]
            for result in results:
                cluster_type = result[0]
                if cluster_type == "conf_test":
                    actual_data = [result[1], exp_name]
                    measurements[cluster_type].append(actual_data)
                else:
                    # print(cluster_type)
                    (avg_ambiguity, weighted_average, max_ambiguity, min_ambiguity) = result[1]
                    n_clusters = result[2]
                    data = []
                    data.extend(result[1])
                    data.append(n_clusters)
                    data.append(exp_name)
                    measurements[cluster_type].append(data)


def read_file(file_name, measurements, exclude_cluster=None):
    with open(file_name) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

        exp_names = lines[1:len(lines):3]
        exp_results = lines[2:len(lines):3]
        exps = zip(exp_names, exp_results)
        for (exp_name, result_string) in exps:
            if exp_name_select and exp_name_select not in exp_name:
                continue

            result_json = result_string.replace("(", "[").replace(")", "]").replace("'", '"')
            import json
            parsed_result = json.loads(result_json)
            for cluster_data in parsed_result:
                cluster_type = cluster_data[0]
                if cluster_type == "conf_test":
                    actual_data = [cluster_data[1], exp_name]
                else:
                    if cluster_type in exclude_cluster:
                        continue
                    actual_data = cluster_data[1]
                    n_clusters = cluster_data[2]
                    actual_data.append(n_clusters)
                    actual_data.append(exp_name)
                measurements[cluster_type].append(actual_data)


def compute_statistics():
    measurements = defaultdict(list)
    if retrained:
        if path_to_retrained_pickle_results is None:
            read_measurement_picklefile("experiment_results/retraining_clustering.pickle", measurements)
            read_measurement_picklefile("experiment_results/retraining_clustering_2.pickle", measurements)
        else:
            read_measurement_picklefile(path_to_retrained_pickle_results, measurements)
    else:
        if path_to_normal_results is None:
            read_file('experiment_results/stats_standard_kmeans_dbscan.txt', measurements, exclude_cluster=[])
            read_file('experiment_results/stats_add_dbscan_kmeans_optics_meanshift.txt', measurements,
                      exclude_cluster=['lda'])
            read_file('experiment_results/stats_mean_shift_lr.txt', measurements, exclude_cluster=['lda'])
        else:
            read_file(path_to_normal_results, measurements)

    filtered_exp_names = list()
    keep_only_perfect_lda_and_la = False

    for meas in measurements['lda']:
        meas_lr = next(filter(lambda meas_lr: meas_lr[5] == meas[5], measurements['lr']))
        if keep_only_perfect_lda_and_la and meas[1] <= eps and meas_lr[1] <= eps:
            filtered_exp_names.append(meas[5])
        elif not keep_only_perfect_lda_and_la:
            filtered_exp_names.append(meas[5])

    if conf_test_threshold is not None:
        assert 'conf_test' in measurements.keys()
        filtered_threshold_exps = []
        for meas in measurements['conf_test']:
            if keep_only_perfect_lda_and_la and meas[1] in filtered_exp_names:
                filtered_threshold_exps.append(meas[1])
            elif not keep_only_perfect_lda_and_la and meas[0] <= conf_test_threshold:
                filtered_threshold_exps.append(meas[1])
        filtered_exp_names = filtered_threshold_exps

    print(f"Number of filtered exp names {len(filtered_exp_names)}")
    measurements_filtered = dict()

    for key, value in measurements.items():
        if key == "conf_test":
            continue
        measurements_filtered[key] = list(
            map(lambda data: data[measurement_index], filter(lambda data: data[5] in filtered_exp_names, value)))

    measurements = measurements_filtered

    fig1, ax1 = plt.subplots()
    # ax1.set_ylabel('Ambiguity')
    plt.xticks(rotation=90)

    i = 0

    # cluster_types = measurements.keys()  # ['DBSCAN_double', 'DBSCAN_triple', 'k_means', 'k_means_4',
    # too keep the order
    cluster_types = ['lda', 'lr', 'DBSCAN', 'DBSCAN_half', 'DBSCAN_double', 'DBSCAN_triple', 'DBSCAN_quad', 'k_means',
                     'k_means_inc', 'k_means_dec', 'k_means_double', 'k_means_4', 'k_means_6', 'k_means_8', 'OPTICS',
                     'mean_shift', 'mean_shift_2', 'mean_shift_4', 'mean_shift_8']

    # for GRUs
    # cluster_types = ['lda', 'lr', 'DBSCAN', 'DBSCAN_double', 'k_means_6', 'k_means_8', 'mean_shift_4', 'mean_shift_8']

    print(f"#Experiments: {len(measurements['lr'])}")
    for cluster_type in cluster_types:
        i += 1
        meas_datapoints = measurements[cluster_type]
        avg_value = sum(meas_datapoints) / len(meas_datapoints)
        import math

        stddev = math.sqrt(1 / len(meas_datapoints) *
                           sum(map(lambda v: (v - avg_value) ** 2, meas_datapoints)))
        count_min = len(list(filter(lambda v: v < eps, meas_datapoints)))
        max_val = max(meas_datapoints)
        ax1.boxplot(meas_datapoints, labels=[cluster_type], positions=[i], sym='x', )

        print(f"Cluster type: {cluster_type}, avg: {avg_value}, stddev: {stddev}, max: {max_val}, #zeros: {count_min}")

    tikzplotlib.save("statistics_output.tex")

    plt.show()


if __name__ == '__main__':
    # default values
    measurement_metric = "wamb"  # "amb" "size"
    measurement_index = 1 if measurement_metric == "wamb" else (0 if measurement_metric == "amb" else 4)
    conf_test_threshold = 0.0
    retrained = False
    exp_name_select = "retrained" if retrained else None
    exp_name_exclude = None
    eps = 1e-6

    exp_name_options = {'all': None, 'retrained': 'retrained', 'lstm': 'lstm', 'gru': 'gru', 'tanh': 'tanh', 'relu': 'relu'}
    import argparse

    parser = argparse.ArgumentParser(description='Compute experiment statistic')
    parser.add_argument('-metric', type=str, help='Measurement metric: either amb (ambiguity), wamb (weighted '
                                                  'ambiguity), or size (cluster size)', nargs='?')
    parser.add_argument('-min_accuracy', type=float, help='Consider only RNNs whose accuracy is greater or equal to '
                                                          'selected value', nargs='?')
    parser.add_argument('-experiment_set', type=str,
                        help='all for all normal RNNs, retrained for noisy constructed RNNs, or one of '
                             '{lstm, relu, tanh, gru} for only that network', nargs='?')
    parser.add_argument('-data_file_normal', type=str,
                        help='Path to a .txt file containing results of normal training and extraction ('
                             'clustering_comparison results)',
                        nargs='?')
    parser.add_argument('-data_file_retrained', type=str, help='Path to a .pickle file containing results of '
                                                               'retraining noisy networks ('
                                                               'retraining_noisy_constructed_RNNs results)', nargs='?')

    args = parser.parse_args()

    path_to_normal_results, path_to_retrained_pickle_results = None, None

    if args.experiment_set:
        exp_name_select = exp_name_options[args.experiment_set]
        if exp_name_select == 'retrained':
            retrained = True
    if args.metric:
        measurement_metric = args.metric
        measurement_index = 1 if measurement_metric == "wamb" else (0 if measurement_metric == "amb" else 4)
    if args.min_accuracy:
        conf_test_threshold = 1.0 - args.min_accuracy
    if args.data_file_normal:
        path_to_normal_results = args.data_file_normal
    if args.data_file_retrained:
        path_to_retrained_pickle_results = args.data_file_retrained

    compute_statistics()
