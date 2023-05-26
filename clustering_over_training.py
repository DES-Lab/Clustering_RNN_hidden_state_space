import pickle
from collections import defaultdict

import torch
import torch.optim as optim
from aalpy.utils import load_automaton_from_file

from RNN import get_model, Optimization
from automata_data_generation import get_tomita, generate_data_from_automaton, AutomatonDataset
from clustering import compute_linear_separability_and_map_to_states, compute_clusters_and_map_to_states
from util import extract_hidden_states, compute_ambiguity

epochs = list(range(10, 51, 5))

def examine_clustering_over_training(opt, model_weight_name):
    clustering_functions = ['k_means_4', 'DBSCAN', ]

    # epochs = list(range(10, 101, 10))
    model_weights_paths = []
    epochs = list(range(10, 51, 5))

    for e in epochs:
        model_weights_paths.append(model_weight_name + f'_epoch_{e}')

    print(model_weight_name)

    hs_processing_fun = 'flatten' if model.model_type != 'lstm' else 'flatten_lstm'
    validation_data, _, _ = generate_data_from_automaton(automaton, num_examples=1000)

    amb_over_training = defaultdict(list)

    for epochs, weights_file in zip(epochs, model_weights_paths):
        load_status = opt.load(weights_file)
        if not load_status:
            print(f'Could not load {weights_file}')
            assert False

        cluster_to_state_maps = []
        hidden_states = extract_hidden_states(opt.model, validation_data, process_hs_fun=hs_processing_fun,
                                              save=False, load=False)

        lda_state_map = compute_linear_separability_and_map_to_states(automaton, hidden_states, validation_data)
        cluster_to_state_maps.append(('lda', lda_state_map))

        for cf in clustering_functions:
            cf_state_map = compute_clusters_and_map_to_states(automaton, validation_data, hidden_states, cf)[0]
            cluster_to_state_maps.append((cf, cf_state_map))

        for cf, state_cluser_map in cluster_to_state_maps:
            amb_over_training[cf].append(compute_ambiguity(state_cluser_map, weighted=True)[1])

    for k, v in amb_over_training.items():
        print(k, v)

    with open(f'{model_weight_name}.pickle', 'wb') as handle:
        pickle.dump(amb_over_training, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('-------------------------------------------------------------')


def visualize_results_over_training(model_weight_name):
    import matplotlib.pyplot  as plt

    amb_over_training = None
    with open(f'{model_weight_name}.pickle', 'rb') as handle:
        amb_over_training = pickle.load(handle)

    epochs = list(range(10, 51, 5))
    for key, values in amb_over_training.items():
        plt.plot(epochs, values, label=key)

    plt.title(model_weight_name)
    plt.xlabel('Epoch')
    plt.ylabel('Ambiguity')
    plt.legend()
    plt.show()


save_dir = "rnn_data"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_type = 'gpu' if device != 'cpu' else 'cpu'

experiments = [(load_automaton_from_file('automata_models/moore_size_8_inputs_2_output3_2.dot', automaton_type='moore'), 'random_moore'),
               (get_tomita(5), 'tomita5'), ]

rnn_types = ['gru', 'lstm'] # 'tanh', 'relu',

num_repeats_per_config = 1

num_training_samples = 50 * 1000
num_validation_samples = 4 * 1000

total_exp_configs = len(experiments) * len(rnn_types) * 1 * num_repeats_per_config
print(f'Total number of trainings: {total_exp_configs}')

current_iteration = 0

perform_training = False

for automaton, exp_name in experiments:
    for rnn in rnn_types:
        optimal_size = len(automaton.get_input_alphabet()) * automaton.size
        # sizes = ((1, optimal_size), (1, int(optimal_size*1.5)), (2, optimal_size))
        sizes = ((1, int(optimal_size * 1.5)),)

        for layers, nodes in sizes:
            for i in range(num_repeats_per_config):
                current_iteration += 1
                print(f'Automated driver progress: {round((current_iteration / total_exp_configs) * 100, 2)}%')

                exp_rnn_config = f'{exp_name}_{rnn}_{layers}_{nodes}_{i + 1}_{device_type}'
                model_weights_name = f'{save_dir}/models/exp_models/{exp_rnn_config}.pt'

                training_data, input_al, output_al = generate_data_from_automaton(automaton,
                                                                                  num_examples=num_training_samples)
                validation_data, _, _ = generate_data_from_automaton(automaton, num_examples=num_validation_samples)

                model_type = rnn if rnn in {'gru', 'lstm'} else 'rnn'
                activation_fun = rnn
                input_dim = len(input_al)
                output_dim = len(output_al)
                layer_dim = layers
                hidden_dim = nodes
                batch_size = 128
                dropout = 0.1 if layer_dim > 1 else 0
                n_epochs = 50
                learning_rate = 0.0005
                weight_decay = 1e-6

                data_handler = AutomatonDataset(input_al, output_al, batch_size)

                train, val = data_handler.create_dataset(training_data), data_handler.create_dataset(validation_data)

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
                if perform_training:
                    opt.train(train, val, n_epochs=n_epochs, exp_name=model_weights_name, early_stop=False,
                              verbose=True,
                              save_interval=5, save=False, load=False)

                # examine_clustering_over_training(opt, model_weights_name)
                visualize_results_over_training(model_weights_name)
