from collections import defaultdict, Counter
from collections import defaultdict, Counter
from random import randint, choice, choices

import numpy as np
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWordEqOracle
from aalpy.utils import visualize_automaton

from clustering import compute_clusters
from util import RNNSul, extract_hidden_states, copy_hs, compute_state_to_hidden_list


def examine_clusters(nn_model, automaton, data, clustering_fun, process_hs_fun='flatten', clustering_fun_args=None):
    if clustering_fun_args is None:
        clustering_fun_args = {}

    if clustering_fun == 'k_means' and clustering_fun_args is None:
        print('Number of states not defined for k_means. Setting it to size of the automaton.')
        clustering_fun_args = {'n_clusters': automaton.size}

    hs = extract_hidden_states(nn_model, data, process_hs_fun)

    cf = compute_clusters(hs, clustering_fun, **clustering_fun_args)

    return cluster_buster(automaton, nn_model, cf, data)


def examine_clusters_with_increasing_seq_len(nn_model, automaton, data, clustering_fun, max_seq_len=30,
                                             process_hs_fun='flatten', clustering_fun_args=None):
    clustering_fun_args = clustering_fun_args if clustering_fun_args is not None else {}

    hs = extract_hidden_states(nn_model, data, process_hs_fun)

    cf = compute_clusters(hs, clustering_fun, **clustering_fun_args)

    input_al = automaton.get_input_alphabet()

    num_test_sequances = 1000
    seq_cluster_buster_dict = dict()
    for i in range(1, max_seq_len + 1):
        test_seqs = [(choices(input_al, k=i), None) for _ in range(num_test_sequances)]
        seq_cluster_buster_dict[i] = cluster_buster(automaton, nn_model, cf, test_seqs)

    return seq_cluster_buster_dict


def extract_automaton_from_rnn(nn_model, input_al, automaton_type='mealy'):
    sul = RNNSul(nn_model)
    eq_oracle = RandomWordEqOracle(input_al, sul=sul, num_walks=1000, min_walk_len=3, max_walk_len=20)

    model = run_Lstar(sul=sul, alphabet=input_al, eq_oracle=eq_oracle, automaton_type=automaton_type,
                      max_learning_rounds=25,
                      print_level=2)

    return model


def extract_hidden_state_automaton_from_rnn(nn_model, input_al, clustering_fun, data, process_hs_fun, n_clusters=None):
    if clustering_fun == 'k_means':
        assert n_clusters
    hs = extract_hidden_states(nn_model, data, process_hs_fun)

    # cf = create_k_means_clusters(hs, n_clusters) if clustering_fun == 'k_means' else create_mean_shift_clusters(hs)
    cf = compute_clusters(hs, 'k_means', n_clusters=n_clusters)

    sul = RNNSul(nn_model, cf)
    eq_oracle = RandomWordEqOracle(input_al, sul=sul, num_walks=1000, min_walk_len=3, max_walk_len=20)

    model = run_Lstar(sul=sul, alphabet=input_al, eq_oracle=eq_oracle, automaton_type='mealy', max_learning_rounds=25,
                      print_level=2)

    visualize_automaton(model)
    return model


def conformance_test(nn_model, automaton, n_tests=10000, min_test_len=16, max_test_len=30):
    sul = RNNSul(nn_model)
    input_al = automaton.get_input_alphabet()

    cex_counter = 0
    for _ in range(n_tests):
        tc = [choice(input_al) for _ in range(randint(min_test_len, max_test_len))]

        sul.pre()
        automaton.reset_to_initial()
        for i in tc:
            o_sul = sul.step(i)
            o_aut = automaton.step(i)
            if o_sul != o_aut:
                cex_counter += 1
                break
        sul.post()

    print(f'Conformance Testing with {n_tests} Random Strings Found {cex_counter} counterexamples.')
    return cex_counter / n_tests


def compute_between_class_cov(means, classes):
    mean_mean = np.mean(means)
    between_class_cov = np.zeros((means.shape[1], means.shape[1]))
    # print(means.shape)
    for i in range(classes):
        between_class_cov += (means[i, :] - mean_mean) @ np.transpose(means[i, :] - mean_mean)
    between_class_cov *= 1 / classes
    return between_class_cov


def cluster_buster(automaton, model, clustering_function, test_seqs):
    state_cluster_counter = defaultdict(Counter)

    for walk, _ in test_seqs:
        automaton.reset_to_initial()
        model.reset_hidden_state()
        for i in walk:
            _ = automaton.step(i)
            _, hs = model.step(i, return_hidden=True)

            state_id = automaton.current_state.state_id
            hs = copy_hs(hs).reshape(1, -1)
            hs = hs.astype(np.double)  # why does this happen
            cluster = f'c{str(clustering_function.predict(hs)[0])}'
            state_cluster_counter[state_id][cluster] += 1

    return state_cluster_counter


def compute_lda_separation(lda, automaton, hidden_states, test_seqs):
    state_to_hidden_state_list = compute_state_to_hidden_list(automaton, hidden_states, test_seqs)
    weight_vect = np.transpose(lda.coef_)
    # print(weight_vect.shape)
    separation_values = list()
    for j in range(len(range(weight_vect.shape[1]))):
        weight_vect_dir = weight_vect[:, j]
        within_class_cov = lda.covariance_
        between_class_cov = compute_between_class_cov(lda.means_, len(state_to_hidden_state_list))
        # print(weight_vect_dir.shape)
        # print(between_class_cov.shape)
        # print(within_class_cov.shape)
        separation = (np.transpose(weight_vect_dir) @ between_class_cov @ weight_vect_dir) / (
                np.transpose(weight_vect_dir) @ within_class_cov @ weight_vect_dir)
        # print(separation.shape)
        # print(separation)
        separation_values.append(separation)
    return separation_values


def examine_normalized_mutual_info_score(nn_model, automaton, data,
                                         clustering_fun,
                                         process_hs_fun='flatten',
                                         clustering_fun_args=None):
    from util import compute_ambiguity

    if clustering_fun_args is None:
        clustering_fun_args = {}

    if clustering_fun == 'k_means' and clustering_fun_args is None:
        print('Number of states not defined for k_means. Setting it to size of the automaton.')
        clustering_fun_args = {'n_clusters': automaton.size}

    hs = extract_hidden_states(nn_model, data, process_hs_fun)

    cf = compute_clusters(hs, clustering_fun, **clustering_fun_args)

    # our technique
    state_cluster_counter = defaultdict(Counter)
    # normalized mutual info
    automaton_states, cluster_labels = [], []

    for walk, _ in data:
        automaton.reset_to_initial()
        nn_model.reset_hidden_state()
        for i in walk:
            _ = automaton.step(i)
            _, hs = nn_model.step(i, return_hidden=True)

            state_id = automaton.current_state.state_id
            hs = copy_hs(hs).reshape(1, -1)
            hs = hs.astype(np.double)  # why does this happen
            cluster = f'c{str(cf.predict(hs)[0])}'

            state_cluster_counter[state_id][cluster] += 1

            automaton_states.append(state_id)
            cluster_labels.append(cluster)

    from sklearn.metrics import normalized_mutual_info_score

    nmi_score = normalized_mutual_info_score(automaton_states, cluster_labels)
    amb = compute_ambiguity(state_cluster_counter)

    print('NMI:', nmi_score)
    print('AMB:', amb)