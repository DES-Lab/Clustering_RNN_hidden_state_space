from collections import defaultdict, Counter

from sklearn.cluster import estimate_bandwidth

from clustering import compute_linear_separability_classifier, \
    compute_clusters_and_map_to_states, compute_linear_separability_and_map_to_states, \
    compute_clusters_and_map_to_states_prime
from methods import extract_hidden_states, compute_lda_separation
from util import compute_ambiguity


# Unused function
def compute_lda_and_separation(automaton, model, validation_data):
    hs_processing_fun = 'flatten' if model.model_type != 'lstm' else 'flatten_lstm'

    import time
    before_extract = time.time()
    hidden_states = extract_hidden_states(model, validation_data, process_hs_fun=hs_processing_fun,
                                          save=False, load=False)

    before_lda = time.time()
    lda = compute_linear_separability_classifier(automaton, hidden_states, validation_data)
    before_separation = time.time()
    separation_values = compute_lda_separation(lda, automaton, hidden_states, validation_data)
    end = time.time()

    print(f"Separation: {separation_values}")
    print(f"Times: {before_lda - before_extract}, {before_separation - before_lda}, {end - before_separation}")
    return separation_values


# Unused function
def map_clusters_to_lda(automaton, model, validation_data):
    clustering_functions = ['k_means', 'mean_shift', 'spectral', 'DBSCAN', 'OPTICS']

    hs_processing_fun = 'flatten' if model.model_type != 'lstm' else 'flatten_lstm'
    hidden_states = extract_hidden_states(model, validation_data, process_hs_fun=hs_processing_fun,
                                          save=False, load=False)

    cluster_to_lda_maps = []
    lda = compute_linear_separability_classifier(automaton, hidden_states, validation_data)
    lda_predictions = [lda.predict(x.reshape(1, -1)) for x in hidden_states.copy()]
    for cf in clustering_functions:
        cf_predictions = compute_clusters_and_map_to_states(automaton, validation_data, hidden_states, cf,
                                                            return_fit_predict=True)
        lda_to_cf_map = defaultdict(Counter)
        for l, c in zip(lda_predictions, cf_predictions):
            lda_to_cf_map[f'l{l}'][f'c{c}'] += 1
            cluster_to_lda_maps.append(('cf', lda_to_cf_map))

    ambiguity_results = []
    for cf, lda_cluster_map in cluster_to_lda_maps:
        ambiguity_results.append((cf, compute_ambiguity(lda_cluster_map)))

    print(ambiguity_results)


def compare_clustering_methods(automaton, model, validation_data, pda_stack_limit=None, reduced_cf=False):
    if not reduced_cf:
        clustering_functions = ['DBSCAN_half', 'DBSCAN', 'DBSCAN_double', 'DBSCAN_triple', 'DBSCAN_quad', 'k_means',
                                'k_means_dec', 'k_means_inc', 'k_means_double', 'k_means_4', 'k_means_6', 'k_means_8',
                                'OPTICS', 'mean_shift', 'mean_shift_2', 'mean_shift_4', 'mean_shift_8']
    if reduced_cf:
        clustering_functions = ['DBSCAN_half', 'DBSCAN',  'DBSCAN_quad', 'k_means',
                               'k_means_4', 'k_means_8', 'OPTICS', 'mean_shift', 'mean_shift_4',]

    model.eval()

    hs_processing_fun = 'flatten' if model.model_type != 'lstm' else 'flatten_lstm'
    hidden_states = extract_hidden_states(model, validation_data, process_hs_fun=hs_processing_fun,
                                          save=False, load=False)

    cluster_to_state_maps = []
    lda_state_map = compute_linear_separability_and_map_to_states(automaton, hidden_states, validation_data,
                                                                  method='lda', pda_stack_limit=pda_stack_limit)
    cluster_to_state_maps.append(('lda', lda_state_map, automaton.size))
    lr_state_map = compute_linear_separability_and_map_to_states(automaton, hidden_states, validation_data, method='lr',
                                                                 pda_stack_limit=pda_stack_limit)
    cluster_to_state_maps.append(('lr', lr_state_map, automaton.size))

    mean_shift_default_bandwidth = estimate_bandwidth(hidden_states)

    for cf in clustering_functions:
        cf_state_map, nr_clusters = compute_clusters_and_map_to_states(automaton, validation_data, hidden_states,
                                                                       cf, mean_shift_default_bandwidth,
                                                                       pda_stack_limit=pda_stack_limit)
        cluster_to_state_maps.append((cf, cf_state_map, nr_clusters))

    ambiguity_results = []
    for cf, state_cluster_map, nr_clusters in cluster_to_state_maps:
        ambiguity_results.append((cf, compute_ambiguity(state_cluster_map, weighted=True), nr_clusters))

    # print(ambiguity_results)
    return ambiguity_results


def compare_clustering_methods_pda_and_reg(pda, reg_approx, model, validation_data, pda_stack_limit=None):
    clustering_functions = ['DBSCAN_half', 'DBSCAN', 'DBSCAN_double', 'DBSCAN_triple', 'DBSCAN_quad', 'k_means',
                            'k_means_dec', 'k_means_inc', 'k_means_double', 'k_means_4', 'k_means_6', 'k_means_8',
                            'OPTICS', 'mean_shift', 'mean_shift_2', 'mean_shift_4', 'mean_shift_8']

    model.eval()

    hs_processing_fun = 'flatten' if model.model_type != 'lstm' else 'flatten_lstm'
    hidden_states = extract_hidden_states(model, validation_data, process_hs_fun=hs_processing_fun,
                                          save=False, load=False)

    cluster_to_state_maps = []

    # separability for PDAs
    lda_state_map = compute_linear_separability_and_map_to_states(pda, hidden_states, validation_data,
                                                                  method='lda', pda_stack_limit=pda_stack_limit)
    cluster_to_state_maps.append(('pda', 'lda', lda_state_map, pda.size))
    lr_state_map = compute_linear_separability_and_map_to_states(pda, hidden_states, validation_data, method='lr',
                                                                 pda_stack_limit=pda_stack_limit)
    cluster_to_state_maps.append(('pda', 'lr', lr_state_map, pda.size))

    # separability for regular
    lda_state_map = compute_linear_separability_and_map_to_states(reg_approx, hidden_states, validation_data,
                                                                  method='lda', pda_stack_limit=None)
    cluster_to_state_maps.append(('reg', 'lda', lda_state_map, reg_approx.size))
    lr_state_map = compute_linear_separability_and_map_to_states(reg_approx, hidden_states, validation_data,
                                                                 method='lr',
                                                                 pda_stack_limit=None)
    cluster_to_state_maps.append(('reg', 'lr', lr_state_map, reg_approx.size))

    mean_shift_default_bandwidth = estimate_bandwidth(hidden_states)

    for cf in clustering_functions:
        if cf.startswith('k_means'):
            for automaton, automaton_type in [(pda, 'pda'), (reg_approx, 'reg')]:
                cf_state_map, nr_clusters = compute_clusters_and_map_to_states(automaton, validation_data,
                                                                               hidden_states,
                                                                               cf, mean_shift_default_bandwidth,
                                                                               pda_stack_limit=pda_stack_limit
                                                                               if automaton_type == 'pda' else None)
                cluster_to_state_maps.append((automaton_type, cf, cf_state_map, nr_clusters))
        else:
            cf_state_map_pda, cf_state_map_reg, nr_clusters = compute_clusters_and_map_to_states_prime(pda, reg_approx,
                                                                                                       validation_data,
                                                                                                       hidden_states,
                                                                                                       cf,
                                                                                                       mean_shift_default_bandwidth,
                                                                                                       pda_stack_limit=pda_stack_limit)
            cluster_to_state_maps.append(('pda', cf, cf_state_map_pda, nr_clusters))
            cluster_to_state_maps.append(('reg', cf, cf_state_map_reg, nr_clusters))

    ambiguity_results = []
    for automaton_type, cf, state_cluster_map, nr_clusters in cluster_to_state_maps:
        ambiguity_results.append((automaton_type, cf, compute_ambiguity(state_cluster_map, weighted=True), nr_clusters))

    print(ambiguity_results)
    return ambiguity_results
