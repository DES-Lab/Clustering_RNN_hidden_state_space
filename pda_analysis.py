import pickle
from collections import defaultdict, Counter
from collections import defaultdict, Counter
import random
import torch
import torch.optim as optim
from aalpy.utils import load_automaton_from_file
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import precision_score, recall_score

from PushDownAutomaton import pda_for_L1, pda_for_L2, pda_for_L4, generate_data_from_pda, pda_for_L3, pda_for_L6, \
    pda_for_L7, pda_for_L8, pda_for_L9, pda_for_L10, pda_for_L11, pda_for_L12, pda_for_L13, pda_for_L15, pda_for_L14, \
    pda_for_L5
from RNN import get_model, Optimization
from automata_data_generation import generate_data_from_automaton, AutomatonDataset
from clustering import compute_clusters_and_map_to_states
from methods import extract_hidden_states, conformance_test
from util import compute_ambiguity


def get_fpa(validation_data, cluster_labels, pda):
    fpa_state_input_frequencies = defaultdict(Counter)
    fpa_state_input_reached_state_frequencies = defaultdict(Counter)

    cluster_index = 0
    initial_state = (f'c{cluster_labels[0]}', pda.initial_state.is_accepting)
    for inputs, _ in validation_data:
        pda.reset_to_initial()

        for i in inputs[:-1]:
            origin = cluster_labels[cluster_index]
            is_accepting = pda.current_state.is_accepting
            origin_state = (f'c{origin}', is_accepting)
            fpa_state_input_frequencies[origin_state][i] += 1

            o = pda.step(i)
            reached_state = (f'c{cluster_labels[cluster_index + 1]}', o)

            fpa_state_input_reached_state_frequencies[(origin_state, i)][reached_state] += 1
            cluster_index += 1

        cluster_index += 1

    assert cluster_index == len(cluster_labels)

    # for k, v in fpa_state_input_frequencies.items():
    #     print(k, v)
    # print('--------------------')
    # for k, v in fpa_state_input_reached_state_frequencies.items():
    #     print(k, v)
    return fpa_state_input_frequencies, fpa_state_input_reached_state_frequencies, initial_state


def generate_data(fpa_state_input_frequencies, fpa_state_input_reached_state_frequencies, initial_state,
                  num_sequances=4000, max_len=15):
    data = []
    for _ in range(num_sequances):
        test_case = []
        state, is_accepting = initial_state

        for _ in range(random.randint(4, max_len)):
            if (state, is_accepting) not in fpa_state_input_frequencies.keys():
                break
            inputs, freq = zip(*fpa_state_input_frequencies[(state, is_accepting)].items())
            random_input = random.choices(inputs, weights=freq, k=1)[0]

            reachable_states, freq = zip(
                *fpa_state_input_reached_state_frequencies[((state, is_accepting), random_input)].items())
            reached_state = random.choices(reachable_states, weights=freq, k=1)[0]

            state, is_accepting = reached_state

            test_case.append((random_input, is_accepting))

        data.append(test_case)

    return data


def get_similarity_score(pda, data):
    num_diff_sequances = 0

    actual_labels, predicted_labels = [], []

    for i_o_sequence in data:
        pda.reset_to_initial()
        faulty_seq = False
        for i, predicted_output in i_o_sequence:
            o = pda.step(i)

            actual_labels.append(o)
            predicted_labels.append(predicted_output)

            if predicted_output != o:
                faulty_seq = True

        if faulty_seq:
            num_diff_sequances += 1

    accuracy = round(1 - (num_diff_sequances / len(data)), 4)
    precision = precision_score(actual_labels, predicted_labels)
    recall = recall_score(actual_labels, predicted_labels)
    print(f'Sequance-wise match  : {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall   : {recall}')

    return accuracy, precision, recall


def get_cf_and_ambiguity_prime(automaton, model, validation_data, pda_stack_limit=None):
    clustering_functions = ['k_means_8', ]

    model.eval()

    hs_processing_fun = 'flatten' if model.model_type != 'lstm' else 'flatten_lstm'
    hidden_states = extract_hidden_states(model, validation_data, process_hs_fun=hs_processing_fun,
                                          save=False, load=False)

    mean_shift_default_bandwidth = estimate_bandwidth(hidden_states)

    cf_state_map, nr_clusters, cf, cluster_labels = compute_clusters_and_map_to_states(automaton, validation_data,
                                                                                       hidden_states,
                                                                                       'k_means_8',
                                                                                       mean_shift_default_bandwidth,
                                                                                       pda_stack_limit=pda_stack_limit,
                                                                                       return_cf=True)

    ambiguity_results = compute_ambiguity(cf_state_map, weighted=True)
    print(ambiguity_results)
    return cf, ambiguity_results, hidden_states, cluster_labels


if __name__ == '__main__':

    save_dir = 'rnn_data'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = 'gpu' if device != 'cpu' else 'cpu'

    # experiments = [(get_tomita(3), 'tomita3'),
    #                (get_tomita(5), 'tomita5'),
    #                (get_tomita(7), 'tomita7'),
    #                (get_mqtt_mealy(), 'mqtt'),
    #                (load_automaton_from_file('automata_models/regex_paper.dot', automaton_type='dfa'), 'regex')]
    #
    # randomly_generated_dfa = [f for f in listdir('automata_models/') if f[:3] == 'dfa']
    # randomly_generated_moore_machines = [f for f in listdir('automata_models/') if f[:5] == 'moore']
    #
    # for aut in randomly_generated_dfa:
    #     experiments.append((load_automaton_from_file(f'automata_models/{aut}', automaton_type='dfa'), aut))
    #
    # for aut in randomly_generated_moore_machines:
    #     experiments.append((load_automaton_from_file(f'automata_models/{aut}', automaton_type='moore'), aut))
    experiments = [(pda_for_L1(), 'CFG_L_1', load_automaton_from_file('automata_models/pda_L1.dot', 'dfa')),
                   (pda_for_L2(), 'CFG_L_2', load_automaton_from_file('automata_models/pda_L2.dot', 'dfa')),
                   (pda_for_L3(), 'CFG_L_3', load_automaton_from_file('automata_models/pda_L3.dot', 'dfa')),
                   (pda_for_L4(), 'CFG_L_4', load_automaton_from_file('automata_models/pda_L4.dot', 'dfa')),
                   (pda_for_L5(), 'CFG_L_5', load_automaton_from_file('automata_models/pda_L5.dot', 'dfa')),
                   (pda_for_L6(), 'CFG_L_6', load_automaton_from_file('automata_models/pda_L6.dot', 'dfa')),
                   (pda_for_L7(), 'CFG_L_7', load_automaton_from_file('automata_models/pda_L7.dot', 'dfa')),
                   (pda_for_L8(), 'CFG_L_8', load_automaton_from_file('automata_models/pda_L8.dot', 'dfa')),
                   (pda_for_L9(), 'CFG_L_9', load_automaton_from_file('automata_models/pda_L9.dot', 'dfa')),
                   (pda_for_L10(), 'CFG_L_10', load_automaton_from_file('automata_models/pda_L10.dot', 'dfa')),
                   (pda_for_L11(), 'CFG_L_11', load_automaton_from_file('automata_models/pda_L11.dot', 'dfa')),
                   (pda_for_L12(), 'CFG_L_12', load_automaton_from_file('automata_models/pda_L12.dot', 'dfa')),
                   (pda_for_L13(), 'CFG_L_13', load_automaton_from_file('automata_models/pda_L13.dot', 'dfa')),
                   (pda_for_L14(), 'CFG_L_14', load_automaton_from_file('automata_models/pda_L14.dot', 'dfa')),
                   (pda_for_L15(), 'CFG_L_15', load_automaton_from_file('automata_models/pda_L15.dot', 'dfa')),
                   ]


    rnn_types = ['tanh', 'relu', 'gru', 'lstm']
    num_repeats_per_config = 2
    pda_stack_limit = 3

    num_training_samples = 50 * 1000
    num_validation_samples = 4 * 1000

    total_exp_configs = len(experiments) * len(rnn_types) * 2 * num_repeats_per_config
    print(f'Total number of trainings: {total_exp_configs}')

    current_iteration = 0

    results = dict()

    with open('experiment_results/pfa_precision_recall.txt', 'a') as stats_file:
        for pda, exp_name, regular_approximation in experiments:
            for rnn in rnn_types:
                optimal_size = len(pda.get_input_alphabet()) * pda.size
                # EDI
                sizes = [(1, optimal_size), (1, int(optimal_size * 1.5)),
                         (2, optimal_size), (1, optimal_size * 3), (1, int(optimal_size * 5))]

                for layers, nodes in sizes:
                    for i in range(num_repeats_per_config):
                        current_iteration += 1
                        print(f'Automated driver progress: {round((current_iteration / total_exp_configs) * 100, 2)}%')

                        exp_rnn_config = f'{exp_name}_{rnn}_{layers}_{nodes}_{i + 1}_{device_type}'
                        model_weights_name = f'{save_dir}/models/exp_models/{exp_rnn_config}.pt'
                        print(exp_rnn_config)

                        if "CFG" in exp_name:
                            training_data, input_al, output_al = generate_data_from_pda(pda,
                                                                                        num_examples=num_training_samples)

                            validation_data, _, _ = generate_data_from_pda(pda,
                                                                           num_examples=num_validation_samples)
                        else:
                            training_data, input_al, output_al = generate_data_from_automaton(pda,
                                                                                              num_examples=num_training_samples)
                            validation_data, _, _ = generate_data_from_automaton(pda,
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

                        load_status = opt.load(model_weights_name)
                        if not load_status:
                            print(f'Can not find weights file of: {model_weights_name}')
                            continue

                        conf_test_res = conformance_test(model, pda, min_test_len=5, max_test_len=13)
                        if conf_test_res > 0.2:
                            print(f'Skipping due to low accuracy: {conf_test_res}')
                            continue

                        validation_data, _, _ = generate_data_from_pda(pda, num_examples=6000)

                        cf, amb, hidden_states, cluster_labels = get_cf_and_ambiguity_prime(pda, model, validation_data,
                                                                                            3)

                        state_input_frequencies, transitions, initial_state = get_fpa(validation_data, cluster_labels,
                                                                                      pda)
                        generated_i_o_seq = generate_data(state_input_frequencies, transitions, initial_state,
                                                          num_sequances=26000, max_len=20)
                        acc, prec, recall = get_similarity_score(pda, generated_i_o_seq)
                        results[exp_rnn_config] = (acc, prec, recall)

                        with open('pda_acc_prec_recall.pickle', 'wb') as handle:
                            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        print('---------------')
