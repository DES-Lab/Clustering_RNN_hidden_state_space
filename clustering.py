from collections import defaultdict, Counter

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering, MeanShift, \
    AgglomerativeClustering, FeatureAgglomeration, DBSCAN, OPTICS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from PushDownAutomaton import Pda
from util import compute_state_to_hidden_list

clustering_functions = {
    'k_means': KMeans,
    'mini_batch_k_means': MiniBatchKMeans,
    'spectral': SpectralClustering,
    'mean_shift': MeanShift,
    'agglo': AgglomerativeClustering,
    'feature_agglo': FeatureAgglomeration,
    'DBSCAN': DBSCAN,
    'OPTICS': OPTICS
}


def get_available_clustering_functions():
    return list(clustering_functions.keys())


def compute_clusters(data, clustering_fun, **kwargs):
    assert clustering_fun in clustering_functions.keys()

    if clustering_fun == 'k_means':
        kwargs['init'] = 'k-means++'

    if clustering_fun == 'OPTICS':
        kwargs['n_jobs'] = -1

    cf = clustering_functions[clustering_fun](**kwargs)
    cf.fit(data)

    return cf


def compute_linear_separability_classifier(automaton, hidden_states, test_seqs, method='lda', reduce_dims=False,
                                           pda_stack_limit=None):
    assert method in {'lda', 'lr'}
    state_to_hidden_state_list = compute_state_to_hidden_list(automaton, hidden_states, test_seqs,
                                                              pda_stack_limit=pda_stack_limit)

    x_list = []
    y_list = []
    for (s_id, hs) in state_to_hidden_state_list:
        for h in hs:
            x_list.append(h)
            y_list.append(s_id)
    x = np.array(x_list)

    y = np.array(y_list)

    if method == 'lda':
        lin_sep = LinearDiscriminantAnalysis(store_covariance=True, n_components=2 if reduce_dims else None,
                                             solver='svd')
    else:
        lin_sep = LogisticRegression(max_iter=1000, solver='saga')
    lin_sep.fit(x, y)
    return lin_sep


def compute_linear_separability_and_map_to_states(automaton, hidden_states, test_seq, method='lda',
                                                  pda_stack_limit=None):
    lda = compute_linear_separability_classifier(automaton, hidden_states, test_seq, method,
                                                 pda_stack_limit=pda_stack_limit)

    hs_list = hidden_states.copy()
    if not isinstance(hs_list, list):
        hs_list = hs_list.tolist()

    state_cluster_counter = defaultdict(Counter)
    for walk, _ in test_seq:
        automaton.reset_to_initial()
        for i in walk:
            _ = automaton.step(i)

            hs = hs_list.pop(0)
            hs = np.array(hs)
            hs = hs.reshape(1, -1)
            state_id = automaton.current_state.state_id
            cluster = f'c{str(lda.predict(hs)[0])}'
            state_cluster_counter[state_id][cluster] += 1

    return state_cluster_counter


def compute_clusters_and_map_to_states(automaton, test_seq, hidden_states, clustering_fun,
                                       mean_shift_default_bandwidth=None, return_fit_predict=False,
                                       pda_stack_limit=None, return_cf=False):
    n_cluster_override = None
    clustering_fun_saved = clustering_fun
    kwargs = dict()

    import time
    start = time.time()
    if clustering_fun.startswith('k_means'):
        if "double" in clustering_fun:
            n_cluster_override = automaton.size * 2
        if "4" in clustering_fun:
            n_cluster_override = automaton.size * 4
        if "8" in clustering_fun:
            n_cluster_override = automaton.size * 8
        if "6" in clustering_fun:
            n_cluster_override = automaton.size * 6
        if "inc" in clustering_fun:
            n_cluster_override = automaton.size + 1
        if "dec" in clustering_fun:
            n_cluster_override = automaton.size - 1

        if pda_stack_limit is not None:
            if n_cluster_override is None:
                n_cluster_override = automaton.size
            if pda_stack_limit != -1:
                n_cluster_override *= pda_stack_limit
            else:
                n_cluster_override *= len(automaton.get_input_alphabet())
        clustering_fun = 'k_means'

    if 'DBSCAN' in clustering_fun:
        if 'double' in clustering_fun:
            kwargs['eps'] = 1
        if 'quad' in clustering_fun:
            kwargs['eps'] = 2
        if 'triple' in clustering_fun:
            kwargs['eps'] = 1.5
        if 'half' in clustering_fun:
            kwargs['eps'] = 0.25
        clustering_fun = 'DBSCAN'

    if 'mean_shift' in clustering_fun:
        mean_shift_bandwidth = mean_shift_default_bandwidth
        if '2' in clustering_fun:
            mean_shift_bandwidth = mean_shift_default_bandwidth / 2
        if '4' in clustering_fun:
            mean_shift_bandwidth = mean_shift_default_bandwidth / 4
        if '8' in clustering_fun:
            mean_shift_bandwidth = mean_shift_default_bandwidth / 8

        kwargs['bandwidth'] = mean_shift_bandwidth
        clustering_fun = 'mean_shift'

    assert clustering_fun in clustering_functions.keys()

    if clustering_fun == 'k_means':
        kwargs['init'] = 'k-means++'
        if n_cluster_override:
            kwargs['n_clusters'] = n_cluster_override
        else:
            kwargs['n_clusters'] = automaton.size

    if clustering_fun in {'OPTICS', 'mean_shift', 'DBSCAN'}:
        kwargs['n_jobs'] = -1

    cf = clustering_functions[clustering_fun](**kwargs)

    effective_hidden_states = hidden_states
    if clustering_fun in {'spectral', 'mean_shift', 'OPTICS'}:
        effective_hidden_states = effective_hidden_states[0:int(len(hidden_states) / 4)]

    cluster_labels = cf.fit_predict(effective_hidden_states).tolist()
    end = time.time()
    print(f"Time for {clustering_fun_saved} : {end - start}")

    nr_clusters = len(set(cf.labels_))
    if return_fit_predict:
        return cluster_labels

    cl_copy = cluster_labels.copy()
    state_cluster_counter = defaultdict(Counter)

    for walk, _ in test_seq:
        automaton.reset_to_initial()
        for i in walk:
            if not cluster_labels:
                break
            _ = automaton.step(i)
            state_id = automaton.current_state.state_id
            if pda_stack_limit is not None and isinstance(automaton, Pda):
                state_id = f"{state_id}_{min(len(automaton.config), pda_stack_limit)}"
            if pda_stack_limit == -1 and isinstance(automaton, Pda):
                state_id = f"{state_id}_{automaton.top()}"
            cluster = f'c{str(cluster_labels.pop(0))}'
            state_cluster_counter[state_id][cluster] += 1

    if return_cf:
        return state_cluster_counter, nr_clusters, cf, cl_copy
    return state_cluster_counter, nr_clusters


def compute_clusters_and_map_to_states_prime(pda, reg, test_seq, hidden_states, clustering_fun,
                                             mean_shift_default_bandwidth=None, return_fit_predict=False,
                                             pda_stack_limit=None):
    n_cluster_override = None
    clustering_fun_saved = clustering_fun
    kwargs = dict()

    import time
    start = time.time()

    if 'DBSCAN' in clustering_fun:
        if 'double' in clustering_fun:
            kwargs['eps'] = 1
        if 'quad' in clustering_fun:
            kwargs['eps'] = 2
        if 'triple' in clustering_fun:
            kwargs['eps'] = 1.5
        if 'half' in clustering_fun:
            kwargs['eps'] = 0.25
        clustering_fun = 'DBSCAN'

    if 'mean_shift' in clustering_fun:
        mean_shift_bandwidth = mean_shift_default_bandwidth
        if '2' in clustering_fun:
            mean_shift_bandwidth = mean_shift_default_bandwidth / 2
        if '4' in clustering_fun:
            mean_shift_bandwidth = mean_shift_default_bandwidth / 4
        if '8' in clustering_fun:
            mean_shift_bandwidth = mean_shift_default_bandwidth / 8

        kwargs['bandwidth'] = mean_shift_bandwidth
        clustering_fun = 'mean_shift'

    assert clustering_fun in clustering_functions.keys()

    if clustering_fun in {'OPTICS', 'mean_shift', 'DBSCAN'}:
        kwargs['n_jobs'] = -1

    cf = clustering_functions[clustering_fun](**kwargs)

    effective_hidden_states = hidden_states
    if clustering_fun in {'spectral', 'mean_shift', 'OPTICS'}:
        effective_hidden_states = effective_hidden_states[0:int(len(hidden_states) / 4)]

    cluster_labels = cf.fit_predict(effective_hidden_states).tolist()
    end = time.time()
    print(f"Time for {clustering_fun_saved} : {end - start}")

    nr_clusters = len(set(cf.labels_))
    if return_fit_predict:
        return cluster_labels

    state_cluster_counter_pda = defaultdict(Counter)
    state_cluster_counter_reg = defaultdict(Counter)

    for automaton_type, automaton, cluster_counter in [('pda', pda, state_cluster_counter_pda),
                                                       ('reg', reg, state_cluster_counter_reg)]:
        cluster_index = 0
        for walk, _ in test_seq:
            automaton.reset_to_initial()
            for i in walk:
                if not cluster_labels or cluster_index >= len(cluster_labels):
                    break
                _ = automaton.step(i)
                state_id = automaton.current_state.state_id
                if pda_stack_limit is not None and automaton_type == 'pda':
                    state_id = f"{state_id}_{min(len(automaton.config), pda_stack_limit)}"
                cluster = f'c{str(cluster_labels[cluster_index])}'
                cluster_index += 1
                cluster_counter[state_id][cluster] += 1

    return state_cluster_counter_pda, state_cluster_counter_reg, nr_clusters
