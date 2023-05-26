import copy

import numpy as np
import numpy.linalg
import torch.nn
from aalpy.automata import Dfa, DfaState
from aalpy.utils import get_Angluin_dfa
from torch.nn.functional import softmax


class AutomatonRNN(torch.nn.Module):
    def __init__(self, rnn_layer, output_layer, dfa: Dfa, state_encoding, input_encoding, rnn_initial_state,
                 device=None):
        super(AutomatonRNN, self).__init__()

        # Defining the device
        if device is None:
            self.device = torch.device('cuda:0' if device != 'cpu' and torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model_type = 'tanh'
        self.rnn_layer = rnn_layer
        self.output_layer = output_layer.to(self.device)
        self.dfa = dfa
        self.state_encoding = state_encoding
        self.hs = None
        self.input_encoding = input_encoding
        self.hidden_size = len(state_encoding) * len(input_encoding)
        self.rnn_initial_state = rnn_initial_state.reshape([1, self.hidden_size])
        self.rnn_initial_state = torch.tensor(self.rnn_initial_state, dtype=torch.float).unsqueeze(1)
        self.rnn_initial_state = self.rnn_initial_state.to(self.device)

    def forward(self, x):
        self.reset_hidden_state(x.shape[0])
        out, self.hs = self.rnn_layer(x, self.hs)
        model_outputs = self.output_layer(out[:, -1, :])
        return model_outputs

    def reset_hidden_state(self, batch_size=1):
        self.hs = torch.clone(self.rnn_initial_state, memory_format=torch.contiguous_format).repeat(1, batch_size, 1)

    def step(self, inp, do_input_encoding=True, return_hidden=False):
        if do_input_encoding:
            inp = torch.tensor(self.input_encoding[inp], dtype=torch.float).to(self.device)
        inp = inp.unsqueeze(0)
        inp = inp.unsqueeze(0)
        out, self.hs = self.rnn_layer(inp, self.hs)

        out = self.output_layer(out)
        out = out.squeeze()
        # print(softmax(out, dim=0))
        p = softmax(out, dim=0).data
        ind = torch.argmax(p).item()
        output = True if ind == 1 else False
        if return_hidden:
            return output, self.hs.squeeze()
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


class Dfa2RnnTransformer1Layer:
    def __init__(self, dfa: Dfa, saturation_factor, saturation_output, noise, class_to_index_map=None, device=None):
        self.dfa, self.dummy_state = self.add_dummy_initial_state(dfa)
        self.output_size = 2
        self.saturation_factor = saturation_factor
        self.nr_inputs = len(self.dfa.get_input_alphabet())
        self.nr_states = len(self.dfa.states)
        self.hidden_size = self.nr_inputs * self.nr_states
        self.rnn_initial_state = None
        self.saturation_output = saturation_output
        self.noise = noise
        # Defining the device
        if device is None:
            self.device = torch.device('cuda:0' if device != 'cpu' and torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def transform(self):
        input_encoding = self.create_input_encoding_map()
        state_encoding = self.create_state_encoding_map()
        transition_encoding = self.create_transition_encoding_map()
        Wih = self.create_input_weights()  # np.zeros_like(self.create_input_weights())#self.create_input_weights()
        Whh = self.create_transition_mapping_weights(state_encoding, transition_encoding)
        ih_bias = np.full(self.hidden_size,
                          -0.5 * self.saturation_factor)  # np.zeros(self.hidden_size)#  np.full(self.hidden_size, -0.5 * self.saturation_factor)
        hh_bias = np.zeros(self.hidden_size)
        #
        Wih += np.random.normal(0, self.noise, size=Wih.shape)
        ih_bias += np.random.normal(0, self.noise, size=ih_bias.shape)
        hh_bias += np.random.normal(0, self.noise, size=hh_bias.shape)
        Whh += np.random.normal(0, self.noise,
                                size=(self.hidden_size, self.hidden_size))
        #
        any_input = self.dfa.get_input_alphabet()[0]
        self.rnn_initial_state = transition_encoding[(self.dummy_state, any_input)]
        rnn = self.create_rnn_layers(Whh, Wih,
                                     hh_bias, ih_bias)
        output_layer = self.create_output_layer(transition_encoding)
        return AutomatonRNN(rnn, output_layer, self.dfa, state_encoding, input_encoding, self.rnn_initial_state,
                            device=self.device)

    def create_rnn_layers(self, Whh, Wih, hh_bias, ih_bias):
        rnn = torch.nn.RNN(input_size=self.nr_inputs, hidden_size=self.hidden_size, dropout=0, bias=True,
                           batch_first=True,
                           num_layers=1, ).to(self.device)
        with torch.no_grad():
            Wih_tensor = torch.from_numpy(Wih).type(torch.FloatTensor)
            Wih_tensor = Wih_tensor.contiguous()
            rnn.weight_ih_l0 = torch.nn.Parameter(data=Wih_tensor.to(self.device))
            rnn.weight_hh_l0 = torch.nn.Parameter(data=torch.tensor(Whh,
                                                                    dtype=torch.float).to(self.device))
            rnn.bias_ih_l0 = torch.nn.Parameter(data=torch.tensor(ih_bias,
                                                                  dtype=torch.float).to(self.device))
            rnn.bias_hh_l0 = torch.nn.Parameter(data=torch.tensor(hh_bias,
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
        empty_encoding = [-1] * n_states
        encoding_map = dict()
        for i in range(n_states):
            state = self.dfa.states[i]
            encoding = copy.deepcopy(empty_encoding)
            encoding[i] = 1
            encoding_map[state] = np.array(encoding * n_inputs).transpose()
        return encoding_map

    def create_transition_encoding_map(self):
        alphabet = self.dfa.get_input_alphabet()
        n_states = len(self.dfa.states)
        n_inputs = len(self.dfa.get_input_alphabet())
        empty_encoding = [-1] * (n_states * n_inputs)
        encoding_map = dict()
        for q in range(n_states):
            for i in range(n_inputs):
                state = self.dfa.states[q]
                input = alphabet[i]
                encoding = copy.deepcopy(empty_encoding)
                one_hot_pos = q + (i * n_states)
                encoding[one_hot_pos] = 1
                encoding_map[(state, input)] = np.array(encoding).transpose()
        return encoding_map

    def create_model_params(self):
        input_dim = len(self.dfa.get_input_alphabet())
        nr_states = len(self.dfa.states)
        model_params = {'input_dim': input_dim,
                        'hidden_dim': nr_states * input_dim,
                        'layer_dim': 1,
                        'output_dim': self.output_size,
                        'nonlinearity': "tanh",
                        'dropout_prob': 0}
        return model_params

    def create_input_weights(self):
        zeros = [0] * (self.nr_states)
        ones = [-self.saturation_factor] * (self.nr_states)
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

    def create_transition_mapping_weights(self, state_encoding, transition_encoding):
        matrix_rows = []
        # create matrix A, s.t. A * flatten(Whh) <= np.ones(hidden_size) * (-saturation_factor)
        zero_coefficients = np.zeros(self.hidden_size)
        alphabet = self.dfa.get_input_alphabet()
        for i_int in range(self.nr_inputs):
            for q_int in range(self.nr_states):
                q = self.dfa.states[q_int]
                input = alphabet[i_int]
                q_p = q.transitions[input]
                trans_encoded = transition_encoding[(q, input)]
                q_p_enc = state_encoding[q_p]
                for q_index in range(self.hidden_size):  # hidden_size == len(q_p_enc)
                    non_zero_coefficients = copy.deepcopy(trans_encoded)
                    if q_p_enc[q_index] > 0:
                        non_zero_coefficients *= -1
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
        saturation_vector = np.ones(self.hidden_size ** 2) * (-self.saturation_factor)
        weights = inv_matrix @ saturation_vector
        weights = weights.reshape([self.hidden_size, self.hidden_size])
        return weights

    def create_output_layer(self, transition_encoding):
        output_weights = self.compute_output_weights(transition_encoding)
        output_weights += np.random.normal(0, self.noise,
                                           size=output_weights.shape)
        output_layer = torch.nn.Linear(self.hidden_size, self.output_size).to(self.device)
        with torch.no_grad():
            output_layer.weight = torch.nn.Parameter(data=torch.tensor(output_weights,
                                                                       dtype=torch.float))
            output_layer.bias = torch.nn.Parameter(data=torch.tensor(np.zeros(self.output_size), dtype=torch.float))
        return output_layer

    def compute_output_weights(self, transition_encoding):
        result_vector_content = []
        output_coeff_matrix_rows = []
        zero_elems = np.zeros(self.hidden_size)
        for (q, input), t_enc in transition_encoding.items():
            q_prime = q.transitions[input]
            if q_prime.is_accepting:
                result_vector_content.append(np.array([-1, 1]) * self.saturation_output)
            else:
                result_vector_content.append(np.array([1, -1]) * self.saturation_output)
            first_row = np.concatenate([t_enc, zero_elems])
            second_row = np.concatenate([zero_elems, t_enc])
            output_coeff_matrix_rows.append(first_row.reshape([1, self.hidden_size * 2]))
            output_coeff_matrix_rows.append(second_row.reshape([1, self.hidden_size * 2]))
        result_vector = np.concatenate(result_vector_content)
        output_coeff_matrix = np.concatenate(output_coeff_matrix_rows, axis=0)
        output_weights, residuals, matrix_rank, singular_values = np.linalg.lstsq(output_coeff_matrix, result_vector,
                                                                                  rcond=None)
        return output_weights.reshape([2, self.hidden_size])

    def add_dummy_initial_state(self, dfa):
        state_list = list(dfa.states)  # shallow copy, as we do not change existing state
        alphabet = dfa.get_input_alphabet()
        dummy_state = DfaState("dummy")
        old_initial_state = dfa.initial_state
        state_list.insert(0, dummy_state)
        for i in alphabet:
            dummy_state.transitions[i] = old_initial_state
        augmented_dfa = Dfa(initial_state=dummy_state, states=state_list)
        return augmented_dfa, dummy_state


def example():
    example, dfa = 'angluin', get_Angluin_dfa()
    saturation_hidden, saturation_output, noise = 5, 5, 0.05

    transformer = Dfa2RnnTransformer1Layer(dfa, saturation_hidden, saturation_output, noise, device=None)
    aut_rnn = transformer.transform()
    aut_rnn.model_name = example

    from methods import conformance_test
    conformance_test(aut_rnn, dfa)


if __name__ == "__main__":
    example()
