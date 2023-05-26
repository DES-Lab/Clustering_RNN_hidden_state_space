from collections import defaultdict, Counter
from os import listdir

import torch
import torch.optim as optim
from aalpy.utils import load_automaton_from_file
from sklearn.cluster import estimate_bandwidth

from RNN import get_model, Optimization
from automata_data_generation import get_tomita, get_mqtt_mealy, generate_data_from_automaton, AutomatonDataset
from clustering import compute_linear_separability_classifier, \
    compute_clusters_and_map_to_states, compute_linear_separability_and_map_to_states
from dim_reduction import reduce_dimensions
from methods import extract_hidden_states, compute_lda_separation
from util import compute_ambiguity


# Unused function
def compute_lda_and_separation(automaton, model, validation_data):
    hs_processing_fun = 'flatten' if model.model_type != 'lstm' else 'flatten_lstm'

    import time
    before_extract = time.time()
    hidden_states = extract_hidden_states(model, validation_data, process_hs_fun=hs_processing_fun,
                                          save=False, load=False)

    before_lda = time.time()
    lda = compute_linear_separability_classifier(automaton, hidden_states, validation_data)
    before_separation = time.time()
    separation_values = compute_lda_separation(lda, automaton, hidden_states, validation_data)
    end = time.time()

    print(f"Separation: {separation_values}")
    print(f"Times: {before_lda - before_extract}, {before_separation - before_lda}, {end - before_separation}")
    return separation_values


# Unused function
def map_clusters_to_lda(automaton, model, validation_data):
    clustering_functions = ['k_means', 'mean_shift', 'spectral', 'DBSCAN', 'OPTICS']

    hs_processing_fun = 'flatten' if model.model_type != 'lstm' else 'flatten_lstm'
    hidden_states = extract_hidden_states(model, validation_data, process_hs_fun=hs_processing_fun,
                                          save=False, load=False)

    cluster_to_lda_maps = []
    lda = compute_linear_separability_classifier(automaton, hidden_states, validation_data)
    lda_predictions = [lda.predict(x.reshape(1, -1)) for x in hidden_states.copy()]
    for cf in clustering_functions:
        cf_predictions = compute_clusters_and_map_to_states(automaton, validation_data, hidden_states, cf,
                                                            return_fit_predict=True)
        lda_to_cf_map = defaultdict(Counter)
        for l, c in zip(lda_predictions, cf_predictions):
            lda_to_cf_map[f'l{l}'][f'c{c}'] += 1
            cluster_to_lda_maps.append(('cf', lda_to_cf_map))

    ambiguity_results = []
    for cf, lda_cluster_map in cluster_to_lda_maps:
        ambiguity_results.append((cf, compute_ambiguity(lda_cluster_map)))

    print(ambiguity_results)


def compare_clustering_methods(automaton, model, validation_data):
    clustering_functions = ['DBSCAN_half', 'DBSCAN', 'DBSCAN_double', 'DBSCAN_triple', 'DBSCAN_quad', 'k_means',
                            'k_means_dec', 'k_means_inc', 'k_means_double', 'k_means_4', 'k_means_6', 'k_means_8',
                            'OPTICS', 'mean_shift', 'mean_shift_2', 'mean_shift_4', 'mean_shift_8']

    model.eval()

    hs_processing_fun = 'flatten' if model.model_type != 'lstm' else 'flatten_lstm'
    hidden_states = extract_hidden_states(model, validation_data, process_hs_fun=hs_processing_fun,
                                          save=False, load=False)

    cluster_to_state_maps = []
    lda_state_map = compute_linear_separability_and_map_to_states(automaton, hidden_states, validation_data,
                                                                  method='lda')
    cluster_to_state_maps.append(('lda', lda_state_map, automaton.size))
    lr_state_map = compute_linear_separability_and_map_to_states(automaton, hidden_states, validation_data, method='lr')
    cluster_to_state_maps.append(('lr', lr_state_map, automaton.size))

    mean_shift_default_bandwidth = estimate_bandwidth(hidden_states)

    for cf in clustering_functions:
        cf_state_map, nr_clusters = compute_clusters_and_map_to_states(automaton, validation_data, hidden_states,
                                                                       cf, mean_shift_default_bandwidth)
        cluster_to_state_maps.append((cf, cf_state_map, nr_clusters))

    ambiguity_results = []
    for cf, state_cluster_map, nr_clusters in cluster_to_state_maps:
        ambiguity_results.append((cf, compute_ambiguity(state_cluster_map, weighted=True), nr_clusters))

    print(ambiguity_results)
    return ambiguity_results


if __name__ == '__main__':

    save_dir = 'rnn_data'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = 'gpu' if device != 'cpu' else 'cpu'

    experiments = [(get_tomita(3), 'tomita3'),
                   (get_tomita(5), 'tomita5'),
                   (get_tomita(7), 'tomita7'),
                   (get_mqtt_mealy(), 'mqtt'),
                   (load_automaton_from_file('automata_models/regex_paper.dot', automaton_type='dfa'), 'regex')]

    randomly_generated_dfa = [f for f in listdir('automata_models/') if f[:3] == 'dfa']
    randomly_generated_moore_machines = [f for f in listdir('automata_models/') if f[:5] == 'moore']

    for aut in randomly_generated_dfa:
        experiments.append((load_automaton_from_file(f'automata_models/{aut}', automaton_type='dfa'), aut))

    for aut in randomly_generated_moore_machines:
        experiments.append((load_automaton_from_file(f'automata_models/{aut}', automaton_type='moore'), aut))

    rnn_types = ['tanh', 'relu', 'gru', 'lstm']
    num_repeats_per_config = 2

    num_training_samples = 50 * 1000
    num_validation_samples = 4 * 1000

    total_exp_configs = len(experiments) * len(rnn_types) * 3 * num_repeats_per_config
    print(f'Total number of trainings: {total_exp_configs}')

    current_iteration = 0
    with open('experiment_results/stats_mean_shift_lr.txt', 'a') as stats_file:
        for automaton, exp_name in experiments:
            for rnn in rnn_types:
                optimal_size = len(automaton.get_input_alphabet()) * automaton.size
                sizes = ((1, optimal_size), (1, int(optimal_size * 1.5)), (2, optimal_size))
                for layers, nodes in sizes:
                    for i in range(num_repeats_per_config):
                        current_iteration += 1
                        print(f'Automated driver progress: {round((current_iteration / total_exp_configs) * 100, 2)}%')

                        exp_rnn_config = f'{exp_name}_{rnn}_{layers}_{nodes}_{i + 1}_{device_type}'
                        model_weights_name = f'{save_dir}/models/exp_models/{exp_rnn_config}.pt'
                        print(exp_rnn_config)

                        training_data, input_al, output_al = generate_data_from_automaton(automaton,
                                                                                          num_examples=num_training_samples)
                        validation_data, _, _ = generate_data_from_automaton(automaton,
                                                                             num_examples=num_validation_samples)

                        model_type = rnn if rnn in {'gru', 'lstm'} else 'rnn'
                        activation_fun = rnn
                        input_dim = len(input_al)
                        output_dim = len(output_al)
                        layer_dim = layers
                        hidden_dim = nodes
                        batch_size = 128
                        dropout = 0.1 if layer_dim > 1 else 0
                        n_epochs = 200
                        learning_rate = 0.0005
                        weight_decay = 1e-6

                        data_handler = AutomatonDataset(input_al, output_al, batch_size)

                        train, val = data_handler.create_dataset(training_data), data_handler.create_dataset(
                            validation_data)

                        model_params = {'input_dim': input_dim,
                                        'hidden_dim': hidden_dim,
                                        'layer_dim': layer_dim,
                                        'output_dim': output_dim,
                                        'nonlinearity': activation_fun,
                                        'dropout_prob': dropout,
                                        'data_handler': data_handler, }

                        model = get_model(model_type, model_params)
                        model.model_name = model_weights_name

                        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                        opt = Optimization(model=model, optimizer=optimizer)
                        # opt.train(train, val, n_epochs=n_epochs, exp_name=model_weights_name, early_stop=True, verbose=True,
                        #          save=False, load=True)

                        load_status = opt.load(model_weights_name)
                        if not load_status:
                            print(f'Can not find weights file of: {model_weights_name}')
                            continue

                        validation_data, _, _ = generate_data_from_automaton(automaton, num_examples=1000)

                        res = compare_clustering_methods(automaton, model, validation_data)
                        # res = compute_lda_and_separation(automaton, model, validation_data)
                        from methods import conformance_test

                        conf_test_res = conformance_test(model, automaton, min_test_len=5, max_test_len=13)
                        res.append(('conf_test', conf_test_res))
                        stats_file.write(
                            f'Automated driver progress: iteration {current_iteration} of {total_exp_configs}\n')
                        stats_file.write(f'{exp_rnn_config}\n')
                        stats_file.write(f'{res}\n')
                        stats_file.flush()
