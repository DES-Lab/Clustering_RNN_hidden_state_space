import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle
from os import listdir

import torch
import torch.optim as optim
from aalpy.utils import get_Angluin_dfa, load_automaton_from_file

from PushDownAutomaton import *
from RNN import get_model, Optimization
from automata_data_generation import get_tomita, get_mqtt_mealy, generate_data_from_automaton, AutomatonDataset
from clustering_comparison import compare_clustering_methods
from methods import conformance_test

save_dir = 'rnn_data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_type = 'gpu' if device != 'cpu' else 'cpu'

# experiments = [(get_tomita(1), 'tomita1'),
#                (get_tomita(3), 'tomita3'),
#                (get_tomita(5), 'tomita5'),
#                (get_tomita(7), 'tomita7'),
#                (get_Angluin_dfa(), 'angluin'),
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

experiments = []
# Add all context free languages
experiments.extend([
                    # (pda_for_L1(), 'CFG_L_1'),
                    # (pda_for_L2(), 'CFG_L_2'),
                    # (pda_for_L3(), 'CFG_L_3'),
                    # (pda_for_L4(), 'CFG_L_4'),
                    # (pda_for_L5(), 'CFG_L_5'),
                    # (pda_for_L6(), 'CFG_L_6'),
                    # (pda_for_L7(), 'CFG_L_7'),
                    # (pda_for_L8(), 'CFG_L_8'),
                    # (pda_for_L9(), 'CFG_L_9'),
                    # (pda_for_L10(), 'CFG_L_10'),
                    (pda_for_L11(), 'CFG_L_11'),
                    (pda_for_L12(), 'CFG_L_12'),
                    (pda_for_L13(), 'CFG_L_13'),
                    (pda_for_L14(), 'CFG_L_14'),
                    (pda_for_L15(), 'CFG_L_15'),
                    ])

rnn_types = ['relu', 'tanh', 'gru', 'lstm']
num_repeats_per_config = 2

num_training_samples = 50 * 1000
num_validation_samples = 2 * 1000

total_exp_configs = len(experiments) * len(rnn_types) * num_repeats_per_config
# Include sizes
total_exp_configs = total_exp_configs * 3 + total_exp_configs * 2
print(f'Total number of trainings: {total_exp_configs}')

current_iteration = 0

# comment out if you want to see if loading of models works
perform_training = False

accuracy_results = dict()
clustering_results = dict()

accuracy_file = 'new_accuracy_results'
ambiguity_file = 'new_ambiguity_results'

for automaton, exp_name in experiments:
    print(exp_name)
    for rnn in rnn_types:
        optimal_size = len(automaton.get_input_alphabet()) * automaton.size
        sizes = [(1, optimal_size), (1, int(optimal_size * 1.5)), (2, optimal_size)]
        if 'CFG' in exp_name:
            sizes.extend([(1, optimal_size * 3), (1, int(optimal_size * 5))])

        for layers, nodes in sizes:
            for i in range(num_repeats_per_config):
                current_iteration += 1
                print(f'Automated driver progress: {round((current_iteration / total_exp_configs) * 100, 2)}%')

                exp_rnn_config = f'{exp_name}_{rnn}_{layers}_{nodes}_{i + 1}_{device_type}'
                model_weights_name = f'{save_dir}/models/exp_models/{exp_rnn_config}.pt'
                print(model_weights_name)

                if "CFG" in exp_name:
                    training_data, input_al, output_al = generate_data_from_pda(automaton,
                                                                                num_examples=num_training_samples)

                    validation_data, _, _ = generate_data_from_pda(automaton, num_examples=num_validation_samples)
                else:
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
                n_epochs = 200
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
                    opt.train(train, val, n_epochs=n_epochs, exp_name=model_weights_name, early_stop=True,
                              verbose=False,
                              save=False, load=False)

                    opt.save(model_weights_name)
                else:
                    load_status = opt.load(model_weights_name)
                    if not load_status:
                        print(f'Can not find weights file of: {model_weights_name}')
                        continue

                # conf_test_res = 1 - conformance_test(model, automaton)
                # accuracy_results[exp_rnn_config] = conf_test_res

                if 'CFG' in exp_name:
                    # print('Computing clustering functions and its ambiguities (stackless)')
                    # stackless = compare_clustering_methods(automaton, model, validation_data)
                    print('Computing clustering functions and its ambiguities (top of stack)')
                    top_of_stack = compare_clustering_methods(automaton, model, validation_data, pda_stack_limit=-1)

                    # clustering_results[exp_rnn_config + '_stackless'] = stackless
                    clustering_results[exp_rnn_config + '_top_of_stack_2'] = top_of_stack
                    print(top_of_stack)
                else:
                    print('Computing clustering functions and its ambiguities')
                    results = compare_clustering_methods(automaton, model, validation_data)
                    clustering_results[exp_rnn_config] = results

                with open(f'experiment_results/{accuracy_file}_top_of_stack.pickle', 'wb') as handle:
                    pickle.dump(accuracy_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(f'experiment_results/{ambiguity_file}_top_of_stack.pickle', 'wb') as handle:
                    pickle.dump(clustering_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
