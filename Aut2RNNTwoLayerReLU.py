import copy

import numpy as np
import numpy.linalg
import torch.nn
from aalpy.automata import Dfa
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils import get_Angluin_dfa, save_automaton_to_file
from torch import optim
from torch.nn.functional import softmax

from Aut2RNNTwoLayer import get_2state_dfa
from AutomataDataGen import get_tomita, generate_data_from_automaton
from DataProcessing import AutomatonDataset
from RNN import Optimization
from methods import visualize_hidden_states
from util import RNNSul


def retrain_constructed_rnn(rnn, automaton: Dfa, num_training_samples, num_val_samples, exp_name, n_epochs,
                            dropout=None, learning_rate=0.0001, weight_decay=0.0001, device='cpu'):
    device = torch.device(device if device == 'cpu' else 'cuda:0')
    classify_states = False
    training_data, input_al, output_al = generate_data_from_automaton(automaton, num_training_samples,
                                                                      classify_states=classify_states)
    validation_data, _, _ = generate_data_from_automaton(automaton, num_val_samples, classify_states=classify_states)

    data_handler = AutomatonDataset(input_al, output_al, batch_size=1, device=device)

    train, val = data_handler.create_dataset(training_data), data_handler.create_dataset(validation_data)
    model_params = {'input_dim': len(input_al),
                    'hidden_dim': len(input_al) * len(automaton.states),
                    'layer_dim': 2,
                    'output_dim': 2,
                    'nonlinearity': 'relu',
                    'dropout_prob': dropout,
                    'data_handler': data_handler}

    model = rnn
    rnn.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    opt = Optimization(model=model, optimizer=optimizer, device=device)
    model.get_model_name(exp_name)
    process_hs_fun = 'flatten'
    opt.train(train, val, n_epochs=n_epochs, exp_name=exp_name, early_stop=True, load=False)
    return model


class AutomatonRNN(torch.nn.Module):
    def __init__(self, rnn_layer, output_layer, dfa: Dfa, state_encoding, input_encoding,device):
        super(AutomatonRNN, self).__init__()
        self.device = device
        self.rnn_layer = rnn_layer
        self.output_layer = output_layer
        self.dfa = dfa
        self.state_encoding = state_encoding
        self.hs = None
        self.input_encoding = input_encoding
        self.hidden_size = len(state_encoding) * len(input_encoding)

    def forward(self, x):
        hs_save = self.hs
        self.reset_hidden_state()
        for input_index in range(x.size(1)):
            step_input = x[0, input_index, :].squeeze()
            self.step(step_input, do_input_encoding=False)

        model_output = self.output_layer(self.hs[1, :, :])  # second layer state
        self.hs = hs_save
        return model_output

    def reset_hidden_state(self):
        init_h_state = self.state_encoding[self.dfa.initial_state]
        init_h_state = init_h_state.reshape([1, self.hidden_size])
        init_h_state = np.concatenate([init_h_state, init_h_state], axis=0)
        self.hs = torch.tensor(init_h_state, dtype=torch.float).unsqueeze(1).to(self.device)
        self.hs = torch.clone(self.hs, memory_format=torch.contiguous_format)

    def step(self, inp, do_input_encoding=True, return_hidden=False):
        if do_input_encoding:
            inp = torch.tensor(self.input_encoding[inp], dtype=torch.float)
        inp = inp.unsqueeze(0)
        inp = inp.unsqueeze(0)
        out, self.hs = self.rnn_layer(inp, self.hs)
        out_hs = self.hs[1].squeeze()

        self.hs = torch.cat([self.hs[1], self.hs[1]], dim=0).unsqueeze(1)
        out = self.output_layer(out)
        out = out.squeeze()
        p = softmax(out, dim=0).data
        ind = torch.argmax(p).item()
        output = "True" if ind == 1 else "False"
        if return_hidden:
            return output, out_hs
        else:
            return output

    def get_model_name(self, exp_name=None):
        if self.model_name:
            return self.model_name
        else:
            assert exp_name is not None
            self.model_name = f'{self.model_type}{self.activation_fun}_l{self.layer_dim}' \
                              f'_d{self.hidden_dim}_{exp_name}'
            return exp_name


class Dfa2RnnTransformer:
    def __init__(self, dfa: Dfa, saturation_factor_output, noise, device='cpu'):
        self.dfa = dfa
        self.output_size = 2
        self.nr_inputs = len(dfa.get_input_alphabet())
        self.nr_states = len(dfa.states)
        self.hidden_size = self.nr_inputs * self.nr_states
        self.noise_stddev = noise
        self.saturation_factor_output = saturation_factor_output
        self.device = torch.device(device if device == 'cpu' else 'cuda:0')

    def transform(self):
        input_encoding = self.create_input_encoding_map()
        state_encoding = self.create_state_encoding_map()
        first_layer_Whh = np.identity(self.hidden_size)
        first_layer_Wih = self.create_input_weights()
        first_layer_ih_bias = np.full(self.hidden_size, -0.25)
        first_layer_hh_bias = np.zeros(self.hidden_size)
        first_layer_mappings = self.create_first_layer_mappings(state_encoding, input_encoding, first_layer_Whh,
                                                                first_layer_ih_bias, first_layer_Wih)
        second_layer_Whh = np.zeros_like(first_layer_Whh)
        second_layer_Wih = self.create_second_layer_weights(first_layer_mappings, state_encoding)
        second_layer_ih_bias = np.zeros(self.hidden_size)
        second_layer_hh_bias = np.zeros(self.hidden_size)

        first_layer_Wih += np.random.normal(0, self.noise_stddev, size=first_layer_Wih.shape)
        first_layer_ih_bias += np.random.normal(0, self.noise_stddev, size=first_layer_ih_bias.shape)
        first_layer_Whh += np.random.normal(0, self.noise_stddev,
                                            size=(self.hidden_size, self.hidden_size))  #
        second_layer_Wih += np.random.normal(0, self.noise_stddev,
                                             size=(self.hidden_size, self.hidden_size))  # TODO used for experimentation

        rnn = self.create_rnn_layers(first_layer_Whh, first_layer_Wih,
                                     first_layer_hh_bias, first_layer_ih_bias,
                                     second_layer_Whh, second_layer_Wih,
                                     second_layer_hh_bias, second_layer_ih_bias)
        output_layer = self.create_output_layer(state_encoding)
        return AutomatonRNN(rnn, output_layer, self.dfa, state_encoding, input_encoding, self.device)

    def create_rnn_layers(self, first_layer_Whh, first_layer_Wih, first_layer_hh_bias, first_layer_ih_bias,
                          second_layer_Whh, second_layer_Wih, second_layer_hh_bias, second_layer_ih_bias):
        rnn = torch.nn.RNN(input_size=self.nr_inputs, hidden_size=self.hidden_size, dropout=0, bias=True,
                           num_layers=2, nonlinearity="relu").to(self.device)
        with torch.no_grad():
            first_layer_Wih_cont = torch.from_numpy(first_layer_Wih).type(torch.FloatTensor)
            first_layer_Wih_cont = first_layer_Wih_cont.contiguous()
            rnn.weight_ih_l0 = torch.nn.Parameter(data=first_layer_Wih_cont.to(self.device))

            rnn.weight_ih_l1 = torch.nn.Parameter(data=torch.tensor(second_layer_Wih,
                                                                    dtype=torch.float).to(self.device))
            rnn.weight_hh_l0 = torch.nn.Parameter(data=torch.tensor(first_layer_Whh,
                                                                    dtype=torch.float).to(self.device))
            rnn.weight_hh_l1 = torch.nn.Parameter(data=torch.tensor(second_layer_Whh,
                                                                    dtype=torch.float).to(self.device))
            rnn.bias_ih_l0 = torch.nn.Parameter(data=torch.tensor(first_layer_ih_bias,
                                                                  dtype=torch.float).to(self.device))
            rnn.bias_ih_l1 = torch.nn.Parameter(data=torch.tensor(second_layer_ih_bias,
                                                                  dtype=torch.float).to(self.device))
            rnn.bias_hh_l0 = torch.nn.Parameter(data=torch.tensor(first_layer_hh_bias,
                                                                  dtype=torch.float).to(self.device))
            rnn.bias_hh_l1 = torch.nn.Parameter(data=torch.tensor(second_layer_hh_bias,
                                                                  dtype=torch.float).to(self.device))

            rnn.flatten_parameters()
        return rnn

    def create_input_encoding_map(self):
        alphabet = self.dfa.get_input_alphabet()
        n_inputs = len(alphabet)
        empty_encoding = [0] * n_inputs
        encoding_map = dict()
        for i in range(n_inputs):
            symbol = alphabet[i]
            encoding = copy.deepcopy(empty_encoding)
            encoding[i] = 1
            encoding_map[symbol] = np.array(encoding).transpose()
        return encoding_map

    def create_state_encoding_map(self):
        n_states = len(self.dfa.states)
        n_inputs = len(self.dfa.get_input_alphabet())
        empty_encoding = [0.125] * n_states
        encoding_map = dict()
        for i in range(n_states):
            state = self.dfa.states[i]
            encoding = copy.deepcopy(empty_encoding)
            encoding[i] = 1
            encoding_map[state] = np.array(encoding * n_inputs).transpose()
        return encoding_map

    def create_model_params(self):
        input_dim = len(self.dfa.get_input_alphabet())
        nr_states = len(self.dfa.states)
        model_params = {'input_dim': input_dim,
                        'hidden_dim': nr_states * input_dim,
                        'layer_dim': 2,
                        'output_dim': self.output_size,
                        'nonlinearity': "relu",
                        'dropout_prob': 0}
        return model_params

    def create_input_weights(self):
        zeros = [0] * (self.nr_states)
        ones = [-1] * (self.nr_states)
        columns = []
        for index in range(self.nr_inputs):
            column = []
            for inner_index in range(self.nr_inputs):
                if index == inner_index:
                    column.extend(zeros)
                else:
                    column.extend(ones)
            columns.append(column)
        input_weights = np.array(columns, dtype=float)
        return input_weights.transpose()

    def create_first_layer_mappings(self, state_encoding, input_encoding, first_layer_Whh, first_layer_ih_bias,
                                    first_layer_Wih):
        mapping_list = []
        for q in range(self.nr_states):
            for i in range(self.nr_inputs):
                encoded_q = state_encoding[self.dfa.states[q]]
                encoded_i = input_encoding[self.dfa.get_input_alphabet()[i]]
                mapped_by_first_layer = first_layer_Whh @ encoded_q + first_layer_ih_bias + first_layer_Wih @ encoded_i
                mapping_list.append((q, i, np.maximum(mapped_by_first_layer, np.zeros_like(mapped_by_first_layer))))
        return mapping_list

    def create_second_layer_weights(self, first_layer_mappings, state_encoding):
        matrix_rows = []
        # create matrix A, s.t. A * flatten(Whh) <= np.ones(hidden_size) * (-saturation_factor)
        zero_coefficients = np.zeros(self.hidden_size)
        result_vector_content = []
        for (q_int, i_int, mapped) in first_layer_mappings:
            q_p = self.dfa.states[q_int].transitions[self.dfa.get_input_alphabet()[i_int]]
            q_p_enc = state_encoding[q_p]
            for q_index in range(self.hidden_size):  # hidden_size == len(q_p_enc)
                non_zero_coefficients = copy.deepcopy(mapped)
                result_vector_content.append(q_p_enc[q_index])

                # one matrix rows looks like:
                # q_index many zero_coefficients, non_zero_coefficients, n*m - q_index - 1 many zero_coefficients,
                # where n is the number of states and m is the number of inputs
                single_matrix_row_list = []
                for j in range(q_index):
                    single_matrix_row_list.append(zero_coefficients)
                single_matrix_row_list.append(non_zero_coefficients)
                for j in range(self.hidden_size - q_index - 1):
                    single_matrix_row_list.append(zero_coefficients)
                matrix_row_array = np.concatenate(tuple(single_matrix_row_list))
                matrix_rows.append(matrix_row_array.reshape([1, len(matrix_row_array)]))
        matrix = np.concatenate(matrix_rows, axis=0)
        inv_matrix = numpy.linalg.inv(matrix)
        result_vector = np.array(result_vector_content)
        weights = inv_matrix @ result_vector
        weights = weights.reshape([self.hidden_size, self.hidden_size])
        print(weights)
        return weights

    def create_output_layer(self, state_encoding):
        output_weights = self.compute_output_weights(state_encoding)
        output_weights += np.random.normal(0, self.noise_stddev,
                                           size=output_weights.shape)
        output_layer = torch.nn.Linear(self.hidden_size, self.output_size).to(self.device)
        with torch.no_grad():
            output_layer.weight = torch.nn.Parameter(data=torch.tensor(output_weights,
                                                                       dtype=torch.float).to(self.device))
            output_layer.bias = torch.nn.Parameter(data=torch.tensor(np.zeros(self.output_size), dtype=torch.float).to(self.device))
        return output_layer

    def compute_output_weights(self, state_encoding):
        result_vector_content = []
        output_coeff_matrix_rows = []
        zero_elems = np.zeros(self.hidden_size)
        for q, q_enc in state_encoding.items():
            if q.is_accepting:
                result_vector_content.append(np.array([-1, 1]) * self.saturation_factor_output)
            else:
                result_vector_content.append(np.array([1, -1]) * self.saturation_factor_output)
            first_row = np.concatenate([q_enc, zero_elems])
            second_row = np.concatenate([zero_elems, q_enc])
            output_coeff_matrix_rows.append(first_row.reshape([1, self.hidden_size * 2]))
            output_coeff_matrix_rows.append(second_row.reshape([1, self.hidden_size * 2]))
        result_vector = np.concatenate(result_vector_content)
        output_coeff_matrix = np.concatenate(output_coeff_matrix_rows, axis=0)
        output_weights, residuals, matrix_rank, singular_values = np.linalg.lstsq(output_coeff_matrix, result_vector,
                                                                                  rcond=None)
        return output_weights.reshape([2, self.hidden_size])


def main():
    torch.autograd.set_detect_anomaly(True)
    test_dfas = [("Angluin", get_Angluin_dfa()), ("twoQ", get_2state_dfa()), ("tomita3", get_tomita(3))]


    settings_sat_noise = [(2, 2, 0.02), (2, 2, 0.05),  (2, 1, 0.05), (3, 3, 0.02), (3, 3, 0.05),
                          (5, 2, 0.02), (5, 2, 0.05)]
    for (dfa_name, test_dfa) in test_dfas:
        print(f"Processing DFA {dfa_name}")
        for (saturation_output, noise) in settings_sat_noise:
            print(f"Processing settings: {saturation_output}, {noise}")
            transformer = Dfa2RnnTransformer(test_dfa, saturation_output, noise)
            aut_rnn = transformer.transform()
            rnn_sul = RNNSul(aut_rnn, clustering_fun=None)
            input_al = test_dfa.get_input_alphabet()
            eq_oracle = RandomWordEqOracle(input_al, sul=rnn_sul, num_walks=500, min_walk_len=3, max_walk_len=20)
            model = run_Lstar(sul=rnn_sul, alphabet=input_al, eq_oracle=eq_oracle, automaton_type='mealy',
                              max_learning_rounds=5,
                              print_level=2)
            # # visualize_automaton(model)
            if len(model.states) != len(test_dfa.states):
                print(f"Incorrectly constructed model for {(saturation_output, noise)}")
            else:
                print(f"Correctly constructed model for {(saturation_output, noise)}")
            #
            validation_data, _, _ = generate_data_from_automaton(test_dfa, 5000, classify_states=False)
            #
            aut_rnn.model_name = f'constructed_{dfa_name}_{saturation_output}_{noise}.model'
            figure_name = "relu_construction/" + aut_rnn.model_name.replace(".model", "")
            aut_file = "rnn_data/figures/relu_construction/" + aut_rnn.model_name.replace(".model", "")
            save_automaton_to_file(model, aut_file)
            #
            visualize_hidden_states(aut_rnn, test_dfa, validation_data, 'pca', process_hs_fun='copy',
                                    save_path=figure_name)
            retrain_constructed_rnn(aut_rnn, test_dfa, 1000, 1000, "generated_retraining", 50, dropout=0.0,
                                    learning_rate=0.0005, weight_decay=0)
            figure_name_retrained = figure_name.replace("constructed", "retrained")
            visualize_hidden_states(aut_rnn, test_dfa, validation_data, 'pca', process_hs_fun='copy',
                                    save_path=figure_name_retrained)

            rnn_sul = RNNSul(aut_rnn, clustering_fun=None)
            eq_oracle = RandomWordEqOracle(input_al, sul=rnn_sul, num_walks=2000, min_walk_len=3, max_walk_len=20)
            model = run_Lstar(sul=rnn_sul, alphabet=input_al, eq_oracle=eq_oracle, automaton_type='mealy',
                              max_learning_rounds=5,
                              print_level=2)
            aut_file = "rnn_data/figures/relu_construction/" + "retrained_" + aut_rnn.model_name.replace(".model", "")
            save_automaton_to_file(model, aut_file)

            if len(model.states) != len(test_dfa.states):
                print(f"Incorrectly retrained model for {(saturation_output, noise)}")

def example():
    example, dfa = 'angluin', get_Angluin_dfa()
    saturation_hidden, saturation_output, noise = 5, 5, 0.0
    noise = 0.2

    transformer = Dfa2RnnTransformer(dfa, saturation_hidden, saturation_output, device='gpu')
    aut_rnn = transformer.transform()
    aut_rnn.model_name = example

    retrain_constructed_rnn(aut_rnn, dfa, 5000, 1000, "generated_retraining", 50, dropout=0.0,
                            learning_rate=0.0005, weight_decay=0, device='gpu')


if __name__ == "__main__":
    example()
