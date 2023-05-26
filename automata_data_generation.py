import random
from collections import defaultdict
from itertools import product

import torch
from aalpy.automata import MealyMachine, MooreMachine
from aalpy.utils import load_automaton_from_file
from aalpy.utils.HelperFunctions import all_prefixes


def get_mqtt_mealy():
    return load_automaton_from_file('automata_models/MQTT.dot', automaton_type='mealy',
                                    compute_prefixes=True)


def get_coffee_machine():
    return load_automaton_from_file('automata_models/Coffee_machine.dot', automaton_type='mealy',
                                    compute_prefixes=True)


def get_angluin():
    return load_automaton_from_file('automata_models/angluin.dot', automaton_type='dfa', compute_prefixes=True)


def get_tcp():
    return load_automaton_from_file('automata_models/TCP_Linux_Client.dot', automaton_type='mealy',
                                    compute_prefixes=True)


def get_bp():
    return load_automaton_from_file('automata_models/bp_depth_16.dot', automaton_type='dfa',
                                    compute_prefixes=True)


def get_ssh():
    return load_automaton_from_file('automata_models/OpenSSH.dot', automaton_type='mealy',
                                    compute_prefixes=True)


def get_tomita(tomita_num):
    return load_automaton_from_file(f'automata_models/tomita_{tomita_num}.dot', automaton_type='dfa',
                                    compute_prefixes=True)


def generate_data_from_automaton(automaton, num_examples, lens=None, classify_states=False):
    input_al = automaton.get_input_alphabet()
    if isinstance(automaton, MealyMachine):
        output_al = {output for state in automaton.states for output in state.output_fun.values()}
    elif isinstance(automaton, MooreMachine):
        output_al = {state.output for state in automaton.states}
    else:
        output_al = [False, True]
    if classify_states:
        output_al = [s.state_id for s in automaton.states]

    if lens is None:
        lens = list(range(1, 15))

    sum_lens = sum(lens)
    # key is length, value is number of examples for said length
    ex_per_len = dict()

    additional_seq = 0
    for l in lens:
        ex_per_len[l] = int(num_examples * (l / sum_lens)) + 1
        if ex_per_len[l] > pow(len(input_al), l):
            additional_seq += ex_per_len[l] - pow(len(input_al), l)
            ex_per_len[l] = 'comb'

    additional_seq = additional_seq // len([i for i in ex_per_len.values() if i != 'comb'])

    training_data = []
    for l in ex_per_len.keys():
        seqs = []
        if ex_per_len[l] == 'comb':
            seqs = list(product(input_al, repeat=l))
        else:
            for _ in range(ex_per_len[l] + additional_seq):
                seqs.append([random.choice(input_al) for _ in range(l)])

        for seq in seqs:
            automaton.reset_to_initial()
            out = None
            for inp in seq:
                out = automaton.step(inp)

            training_data.append((tuple(seq), out if not classify_states else automaton.current_state.state_id))

    input_al = automaton.get_input_alphabet()

    return training_data, input_al, output_al


def get_characterizing_set_training_data(automaton, include_extended_s_set=True, prefix_closed=True):
    input_al = automaton.get_input_alphabet()
    if isinstance(automaton, MealyMachine):
        output_al = {output for state in automaton.states for output in state.output_fun.values()}
    else:
        output_al = [False, True]

    observation_table_data = []
    prefixes = [state.prefix for state in automaton.states]

    e_set = automaton.compute_characterization_set()
    if include_extended_s_set:
        extended_prefixes = []
        for p in prefixes:
            for a in automaton.get_input_alphabet():
                extended_prefixes.append(p + tuple([a]))

        prefixes.extend(extended_prefixes)

    data_set = set()
    for prefix in prefixes:
        for suffix in e_set:
            cell = prefix + suffix
            if prefix_closed:
                data_set.update(all_prefixes(cell))
            else:
                data_set.add(cell)

    for seq in list(data_set):
        output = automaton.compute_output_seq(automaton.initial_state, seq)[-1]
        observation_table_data.append((seq, output))

    return observation_table_data, input_al, output_al


class AutomatonDataset:
    def __init__(self, input_al_str, classes, batch_size, device=None):
        self.inputs = input_al_str
        self.classes = [str(c) for c in classes]
        self.class_to_index = {k: ind for ind, k in enumerate(classes)}
        self.index_to_class = {ind: k for ind, k in enumerate(classes)}
        self.len_dict = defaultdict(list)
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device('cuda:0' if device != 'cpu' and torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def input_tensor(self, line, length):
        tensor = torch.zeros(length, len(self.inputs)).to(self.device)
        for i, el in enumerate(line):
            tensor[i][self.inputs.index(el)] = 1
        return tensor

    def classification_tensor(self, classification):
        return torch.LongTensor([self.class_to_index[classification]]).to(self.device)

    def create_dataset(self, data):
        batches = []
        for d in data:
            self.len_dict[len(d[0])].append(d)

        for _, seqs in self.len_dict.items():
            sequences = []
            labels = []
            for s in seqs:
                sequences.append(self.input_tensor(s[0], len(s[0])))
                labels.append(self.classification_tensor(s[1]))

            # experimental
            if len(sequences) < self.batch_size:
                while len(sequences) < self.batch_size:
                    sequences.extend(sequences)
                    labels.extend(labels)
                sequences = sequences[:self.batch_size]
                labels = labels[:self.batch_size]

            for i in range(0, len(sequences), self.batch_size):
                batch_seqs = torch.stack(sequences[i:i + self.batch_size])
                batch_labels = torch.stack(labels[i:i + self.batch_size])

                batches.append((batch_seqs, batch_labels))

        return batches
