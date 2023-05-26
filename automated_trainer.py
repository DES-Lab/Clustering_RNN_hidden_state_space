from os import listdir, path

import torch
import torch.optim as optim
from aalpy.utils import get_Angluin_dfa, load_automaton_from_file

from RNN import get_model, Optimization
from automata_data_generation import get_tomita, get_mqtt_mealy, generate_data_from_automaton, AutomatonDataset

save_dir = 'rnn_data/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_type = 'gpu' if device != 'cpu' else 'cpu'

experiments = [(get_tomita(1), 'tomita1'),
               (get_tomita(3), 'tomita3'),
               (get_tomita(5), 'tomita5'),
               (get_tomita(7), 'tomita7'),
               (get_Angluin_dfa(), 'angluin'),
               (get_mqtt_mealy(), 'mqtt'),
               (load_automaton_from_file('automata_models/regex_paper.dot', automaton_type='dfa'), 'regex')]

randomly_generated_dfa = [f for f in listdir('automata_models/') if f[:3] == 'dfa' ]
randomly_generated_moore_machines = [f for f in listdir('automata_models/') if f[:5] == 'moore' ]

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

# comment out if you want to see if loading of models works
perform_training = True

for automaton, exp_name in experiments:
    print(exp_name)
    for rnn in rnn_types:
        optimal_size = len(automaton.get_input_alphabet()) * automaton.size
        sizes = ((1, optimal_size), (1, int(optimal_size*1.5)), (2, optimal_size))
        for layers, nodes in sizes:
            for i in range(num_repeats_per_config):
                current_iteration += 1
                print(f'Automated driver progress: {round((current_iteration/total_exp_configs) * 100, 2)}%')

                exp_rnn_config = f'{exp_name}_{rnn}_{layers}_{nodes}_{i + 1}_{device_type}'
                model_weights_name = f'{save_dir}/models/exp_models/{exp_rnn_config}.pt'
                print(model_weights_name)

                if not perform_training and path.exists(model_weights_name):
                    print(f'Successfully loaded: {model_weights_name}')
                    continue

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
                                'data_handler': data_handler,}

                model = get_model(model_type, model_params)
                model.model_name = model_weights_name

                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                opt = Optimization(model=model, optimizer=optimizer)
                opt.train(train, val, n_epochs=n_epochs, exp_name=model_weights_name, early_stop=True, verbose=False,
                          save=False, load=False)

                opt.save(model_weights_name)

