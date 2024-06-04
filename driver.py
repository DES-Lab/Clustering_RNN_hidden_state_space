from random import shuffle

import torch
import torch.optim as optim
from aalpy.utils import load_automaton_from_file

from PushDownAutomaton import pda_for_L1
from RNN import get_model, Optimization
from automata_data_generation import generate_data_from_automaton, get_tomita, get_coffee_machine, \
    get_ssh, get_angluin, AutomatonDataset
from methods import extract_automaton_from_rnn, conformance_test, \
    extract_hidden_state_automaton_from_rnn, examine_clusters
from util import compute_ambiguity
from visualization_util import visualize_hidden_states, visualize_hs_over_training

experiments = {'CFG_L_1': pda_for_L1(),
               'tomita3': get_tomita(3),
               'tomita5': get_tomita(5),
               'tomita7': get_tomita(7),
               'coffee': get_coffee_machine(),
               'ssh': get_ssh(),
               'angluin': get_angluin(),
               'regex': load_automaton_from_file('automata_models/regex_paper.dot', automaton_type='dfa',
                                                 compute_prefixes=True),
               'tree': load_automaton_from_file('automata_models/tree.dot', automaton_type='dfa',
                                                compute_prefixes=True),
               'last_a': load_automaton_from_file('automata_models/last_a.dot', automaton_type='dfa',
                                                  compute_prefixes=True)
               }

device = None  # for auto detection

exp_name = 'coffee'
automaton = experiments[exp_name]

# Number of training and validation samples
num_training_samples = 50000
num_val_samples = 4000

# Do not learn the original automaton, but a mapping of sequance of inputs to reached state
# Each state is represented by unique state_id
classify_states = False
exp_name = exp_name if not classify_states else exp_name + '_states'

# Generate training and validation data
automaton_data, input_al, output_al = generate_data_from_automaton(automaton, num_training_samples + num_val_samples,
                                                                   classify_states=classify_states)

shuffle(automaton_data)

training_data, validation_data = automaton_data[:num_training_samples], automaton_data[num_training_samples:]

# Setup RNN parameters
model_type = 'rnn'
activation_fun = 'relu'  # note that activation_fun value is irrelevant for GRU and LSTM
input_dim = len(input_al)
output_dim = len(output_al)
hidden_dim = 64
layer_dim = 1
batch_size = 64
dropout = 0  # 0.1 if layer_dim > 1 else 0
n_epochs = 100
optimizer = optim.Adam
learning_rate = 0.0005
weight_decay = 1e-6
early_stop = True  # Stop training if loss is smaller than small threshold for few epochs

data_handler = AutomatonDataset(input_al, output_al, batch_size, device=device)

train, val = data_handler.create_dataset(training_data), data_handler.create_dataset(validation_data)

model_params = {'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'layer_dim': layer_dim,
                'output_dim': output_dim,
                'nonlinearity': activation_fun,
                'dropout_prob': dropout,
                'data_handler': data_handler,
                'device': device}

model = get_model(model_type, model_params)

optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(model=model, optimizer=optimizer, device=device)

model.get_model_name(exp_name)
process_hs_fun = 'flatten_lstm' if model_type == 'lstm' else 'flatten'

# This will train the RNN
# If trained model with same parametrization exists, it will be loaded unless load flag is set to False
opt.train(train, val, n_epochs=n_epochs, exp_name=exp_name, early_stop=early_stop, save=True, load=True)

# disable all gradient computations to speed up execution
model.eval()

# check the RNN for accuracy on newly generated data
conformance_test(model, automaton, n_tests=1000, max_test_len=30)

# function that maps states to clusters over executions of  random sequences
state_cluster_map = examine_clusters(model, automaton, validation_data, 'k_means',
                                     clustering_fun_args={'n_clusters': automaton.size * 8})

amb = compute_ambiguity(state_cluster_map)

for k, v in state_cluster_map.items():
    print(f'{k} : {v}')

print(f'Computed ambiguity: {amb[0]}, Weighted ambiguity: {amb[1]}')
exit()

# extracts input-output automaton from RNN
extract_automaton_from_rnn(model, input_al, automaton_type='moore')

# visualizes the hidden state space over the training process
visualize_hs_over_training(opt, automaton, (train, val), validation_data, epochs=10, save_intervals=1,
                           exp_name=exp_name, train=True)

# create clusters and try to learn a language of clusters
extract_hidden_state_automaton_from_rnn(model, input_al, data=validation_data,
                                        clustering_fun='mini_batch_k_means', process_hs_fun=process_hs_fun)

# check the RNN for accuracy on newly generated data
conformance_test(model, automaton, n_tests=10000, max_test_len=30)
