import pickle
from collections import defaultdict
from math import log

import numpy as np
import torch
from aalpy.base import SUL
from sklearn.metrics import precision_score, recall_score

from PushDownAutomaton import Pda
from dim_reduction import reduce_dimensions


class RNNSul(SUL):
    def __init__(self, nn, clustering_fun=None):
        super().__init__()
        self.nn = nn
        self.clustering_fun = clustering_fun
        self.nn.eval()

    def pre(self):
        self.nn.reset_hidden_state()

    def post(self):
        pass

    def step(self, letter):
        if self.clustering_fun:
            _, hs = self.nn.step(letter, return_hidden=True)
            hs = copy_hs(hs).reshape(1, -1)
            hs = hs.astype(np.double)  # why does this happen
            return f'c{str(self.clustering_fun.predict(hs)[0])}'

        return self.nn.step(letter)


def copy_hs(hs):
    return hs.clone().squeeze_().detach().cpu().numpy()


def flatten_hs(hs):
    return torch.flatten(hs).clone().detach().cpu().numpy()


def flatten_lstm(hs):
    stack = torch.stack(hs)
    return torch.flatten(stack).detach().cpu().numpy()


def copy_lstm_hs(hs):
    return hs[0].clone().squeeze_().detach().cpu().numpy()


def copy_lstm_cs(hs):
    return hs[1].clone().squeeze_().detach().cpu().numpy()


process_hs_functions = {'copy': copy_hs,
                        'flatten': flatten_hs,
                        'flatten_lstm': flatten_lstm,
                        'copy_lstm_hs': copy_lstm_hs,
                        'copy_lstm_cs': copy_lstm_cs}


def filter_tests_leading_to_sink(data, automaton):
    # TODO introduces diff to PCA when comparing to examples where sink is filtered later
    pruned_data = []
    for seq_label in data:
        automaton.reset_to_initial()
        sink_found = False
        for i in seq_label[0]:
            automaton.step(i)
            if automaton.current_state.state_id == 'sink':
                sink_found = True
                break

        if not sink_found:
            pruned_data.append(seq_label)

    return pruned_data


def extract_hidden_states(model, data, process_hs_fun='copy', save=True, load=True):
    assert process_hs_fun in process_hs_functions.keys()

    if model.model_name:
        load_existing = load_from_file(f'rnn_data/hidden_states/{model.model_name}_hs')
        if load and load_existing is not None:
            return load_existing

    hidden_states = []

    for seq, _ in data:
        model.reset_hidden_state()
        for i in seq:
            _, hs = model.step(i, return_hidden=True)
            hidden_states.append(process_hs_functions[process_hs_fun](hs))

    if save:
        save_to_file(hidden_states, f'rnn_data/hidden_states/{model.model_name}_hs')
    return hidden_states


def map_tc_to_states_and_hs(tc, hs, automaton):
    ind = 0
    tc_hs_map = dict()
    for seq, _ in tc:
        automaton.reset_to_initial()
        states = []
        for i in seq:
            automaton.step(i)
            states.append(automaton.current_state.state_id)
        tc_hs_map[seq] = (states, hs[ind:ind + len(seq)])
        ind += len(seq)
    return tc_hs_map


def map_hidden_states_to_automaton(model, automaton, data, process_hs_fun='copy', map_hidden_to='state', save=True,
                                   load=True):
    assert map_hidden_to in {'state', 'state_input', 'input_new_state'}
    assert process_hs_fun in process_hs_functions.keys()

    data = list(set(data))
    load_existing = load_from_file(f'rnn_data/hidden_states/{model.model_name}_hs_state_map')
    if load and load_existing is not None:
        print('State to Hidden State map loaded.')
        return load_existing

    automaton_state_hidden_state_map = defaultdict(list)

    for seq, _ in data:
        automaton.reset_to_initial()
        model.reset_hidden_state()
        for i in seq:
            if map_hidden_to == 'state':
                _, hs = model.step(i, return_hidden=True)
                automaton.step(i)
                state_id = automaton.current_state.state_id
                automaton_state_hidden_state_map[state_id].append(process_hs_functions[process_hs_fun](hs))
            elif map_hidden_to == 'state_input':
                state_id = f'{automaton.current_state.state_id}_{i}'
                _, hs = model.step(i, return_hidden=True)
                automaton.step(i)
                automaton_state_hidden_state_map[state_id].append(process_hs_functions[process_hs_fun](hs))

    if save:
        save_to_file(automaton_state_hidden_state_map, f'rnn_data/hidden_states/{model.model_name}_hs_state_map')
    return automaton_state_hidden_state_map


def reduce_dim_of_state_hidden_state_map(state_hs_map, dim_reduction_fun='pca', dims=2, dim_reduction_args=None):
    if dim_reduction_args is None:
        dim_reduction_args = {}

    reduced_dim_dict = defaultdict(list)

    # get all hidden states in a list
    all_hs = []
    key_order = []
    for k, v in state_hs_map.items():
        key_order.append(k)
        all_hs.extend(v)

    # reduce dimensions
    reduced_data = reduce_dimensions(all_hs, dim_reduction_fun, target_dimensions=dims, **dim_reduction_args).tolist()

    # map reduced dimensions to states
    for k in key_order:
        for _ in state_hs_map[k]:
            reduced_dim_dict[k].append(reduced_data.pop(0))

    assert len(reduced_data) == 0
    return reduced_dim_dict


def map_hidden_states_to_clusters(rnn, data, clustering_fun, dim_reduction_fun='pca', process_hs_fun='copy',
                                  dim_reaction_args=None, clustering_fun_args=None):
    from clustering import compute_clusters

    dim_reaction_args = dim_reaction_args if dim_reaction_args is not None else {}
    clustering_fun_args = clustering_fun_args if clustering_fun_args is not None else {}

    if clustering_fun == 'k_means' and not clustering_fun_args:
        print('Number of states not defined for k_means. Setting it to 8')

    assert process_hs_fun in process_hs_functions.keys()

    data = list(set(data))

    cluster_hidden_state_map = defaultdict(list)

    hs = extract_hidden_states(rnn, data, process_hs_fun, load=False)
    clustering_fun = compute_clusters(hs, clustering_fun, **clustering_fun_args)
    test_seq = reduce_dimensions(hs, dim_reduction_fun, 2, **dim_reaction_args)

    test_index = 0
    for seq, _ in data[:100]:
        for _ in seq:
            data_points = hs[test_index]
            data_points = data_points.reshape(1, -1).astype(np.double)
            cluster = f'c{clustering_fun.predict(data_points)}'
            cluster_hidden_state_map[cluster].append(test_seq[test_index])

            test_index += 1

    return cluster_hidden_state_map


def save_to_file(obj, path):
    pickle.dump(obj, open(f'{path}.pk', "wb"))


def load_from_file(path):
    try:
        with open(f'{path}.pk', "rb") as f:
            return pickle.load(f)
    except IOError:
        return None


# analysis of clusters

def permute_cluster_ids(cluster_names, perm):
    cluster_id_map_permutation = dict()
    for i, name in enumerate(cluster_names):
        cluster_id_map_permutation[name] = perm[i]
    return cluster_id_map_permutation


def apply_permutation(state_cluster_map, state_id_map, cluster_id_map_permuted):
    state_cluster_id_map = dict()
    for (s, cluster_counter) in state_cluster_map.items():
        s_id = state_id_map[s]
        state_cluster_id_map[s_id] = dict()
        for (cluster, counter_value) in cluster_counter.items():
            cluster_id = cluster_id_map_permuted[cluster]
            state_cluster_id_map[s_id][cluster_id] = counter_value
    return state_cluster_id_map


def compute_ambiguity(state_cluster_map, injective=False, weighted = True):
    if injective:
        return compute_ambiguity_injective(state_cluster_map)
    else:
        # non-injective renaming of clusters
        states = sorted(list(state_cluster_map.keys()))
        nr_states = len(states)
        cluster_to_state_count = defaultdict(list)
        for state_name, cluster_counter in state_cluster_map.items():
            for cluster_name in cluster_counter.keys():
                cluster_to_state_count[cluster_name].append((state_name,cluster_counter[cluster_name]))
        cluster_ambiguities = dict()
        for cluster in cluster_to_state_count.keys():
            state_counts = cluster_to_state_count[cluster]
            count_sum = sum(map(lambda sc : sc[1],state_counts))
            entropy_normalized = -sum(map(lambda sc : sc[1]/count_sum * log(sc[1]/count_sum,nr_states), state_counts))
            ambiguity_for_cluster = entropy_normalized
            # print(f"Renaming {cluster} -> {max_s}")
            cluster_ambiguities[cluster] = ambiguity_for_cluster
        avg_ambiguity = sum(cluster_ambiguities.values()) / len(cluster_ambiguities)
        max_ambiguity = max(cluster_ambiguities.values())
        min_ambiguity = min(cluster_ambiguities.values())

        if weighted:
            weighted_average = 0
            states_in_all_clusters = 0
            for cluster in cluster_to_state_count.keys():
                state_counts = cluster_to_state_count[cluster]
                states_in_cluster = sum(map(lambda sc: sc[1], state_counts))
                states_in_all_clusters += states_in_cluster
                weighted_average += cluster_ambiguities[cluster] * states_in_cluster
            weighted_average /= states_in_all_clusters
            return avg_ambiguity, weighted_average, max_ambiguity, min_ambiguity
        return avg_ambiguity, max_ambiguity, min_ambiguity


def compute_ambiguity_injective(state_cluster_map):
    states = sorted(list(state_cluster_map.keys()))
    cluster_names = list()
    for cluster_counter in state_cluster_map.values():
        for cluster_name in cluster_counter.keys():
            if cluster_name not in cluster_names:
                cluster_names.append(cluster_name)
    cluster_names.sort()
    state_id_map = dict(map(lambda s_id: (s_id[1], s_id[0]), enumerate(states)))
    import itertools
    cluster_naming_permutations = list(itertools.permutations(range(len(states))))  # [list(range( len(states)))]
    lowest_ambiguity = (1,1)
    for perm in cluster_naming_permutations:
        cluster_id_map_permuted = permute_cluster_ids(cluster_names, perm)
        state_cluster_id_map_permuted = apply_permutation(state_cluster_map, state_id_map, cluster_id_map_permuted)
        ambiguity = compute_ambiguity_injective_single(state_cluster_id_map_permuted)
        if ambiguity[0] < lowest_ambiguity[0]:
            lowest_ambiguity = ambiguity
    # print(f"Lowest ambiguity: {lowest_ambiguity}")
    return lowest_ambiguity


def compute_ambiguity_injective_single(state_cluster_id_map):
    ambiguity_values = []
    for s_id in state_cluster_id_map.keys():
        cluster_counter_for_state = state_cluster_id_map[s_id]
        all_clusters = sum(cluster_counter_for_state.values())
        cluster_id_count = 0
        if s_id in cluster_counter_for_state:
            cluster_id_count = cluster_counter_for_state[s_id]
        ambiguity_values.append(1 - cluster_id_count / all_clusters)
    avg_ambiguity = sum(ambiguity_values) / len(ambiguity_values)
    max_ambiguity = max(ambiguity_values)
    return avg_ambiguity,max_ambiguity


if __name__ == '__main__':
    # TODO failing test case
    # x = {'s0': {'c1': 1709, 'c5': 804}, 's1': {'c2': 1707}, 's2': {'c6': 1392}, 's4': {'c3': 657, 'c2': 365, 'c5': 146},
    #      'sink': {'c0': 1922, 'c4': 788, 'c7': 767, 'c6': 1}}
    # a = compute_ambiguity(x)
    # print(a)
    #
    # x = {'s0': {'c1': 1709, 'c2': 804, 'c3': 900}, 's1': {'c1': 1709}, 's2': {'c1': 1709}}
    # a = compute_ambiguity(x)
    # print(a)
    #
    #
    # x = {'s0': {'c1': 1709, 'c4': 804}, 's1': {'c2': 1707}, 's2': {'c3': 1392, 'c2' : 231}, 's4': {'c3': 657, 'c2': 365},
    #      'sink': {'c0': 1922, 'c4': 788}}
    x = {'s0': {'c0': 502, 'c12': 446, 'c6': 427, 'c9': 425, 'c5': 260, 'c2': 246, 'c18': 216, 'c21': 204}, 's1': {'c1': 503, 'c13': 391, 'c7': 310, 'c3': 255, 'c19': 214, 'c28': 104}, 's2':{'c8': 412, 'c4': 242, 'c20': 211, 'c14': 208, 'c24': 119, 'c32': 51, 'c33': 49}, 's4': {'c10': 471, 'c15': 220, 'c22': 124, 'c25': 123, 'c30': 63, 'c31': 59, 'c21': 37}, 'sink': {'c16': 531, 'c23': 517, 'c11': 507, 'c17': 481, 'c24': 291, 'c27': 269, 'c26': 256, 'c31': 133, 'c32': 127, 'c22': 76, 'c29': 74, 'c36': 26, 'c-1': 19, 'c34': 18, 'c38': 18, 'c35': 11, 'c37': 6, 'c39': 6}}

    a = compute_ambiguity(x)
    a_inj = compute_ambiguity(x, injective=False)
    print(a, a_inj)


def compute_state_to_hidden_list(automaton, hidden_states, test_seqs, pda_stack_limit=None):
    state_to_hidden_state = defaultdict(list)
    hs_list = hidden_states.copy()
    if not isinstance(hs_list, list):
        hs_list = hs_list.tolist()
    for walk, _ in test_seqs:
        automaton.reset_to_initial()
        for i in walk:
            _ = automaton.step(i)
            state_id = automaton.current_state.state_id
            if pda_stack_limit is not None:
                if pda_stack_limit == -1 and isinstance(automaton, Pda):
                    state_id = f"{state_id}_{automaton.top()}"
                else:
                    state_id = f"{state_id}_{min(len(automaton.config),pda_stack_limit)}"
            state_to_hidden_state[state_id].append(hs_list.pop(0))
    state_to_hidden_state_list = list(state_to_hidden_state.items())
    return state_to_hidden_state_list


def get_accuracy_statistics(rnn, automaton, data):
    num_diff_sequances = 0

    actual_labels, predicted_labels = [], []

    for input_seq, _ in data:
        faulty_seq = False

        automaton.reset_to_initial()
        rnn.reset_hidden_state()

        for i in input_seq:
            rnn_output = rnn.step(i, return_hidden=False)
            automaton_output = automaton.step(i)

            actual_labels.append(automaton_output)
            predicted_labels.append(rnn_output)

            if rnn_output != automaton_output:
                faulty_seq = True

        if faulty_seq:
            num_diff_sequances += 1

    accuracy = 1 - (num_diff_sequances / len(data))
    precision = precision_score(actual_labels, predicted_labels)
    recall = recall_score(actual_labels, predicted_labels)
    print(f'Sequance-wise match  : {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall   : {recall}')

    return accuracy, precision, recall