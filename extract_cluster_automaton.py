from aalpy.SULs import DfaSUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWMethodEqOracle
from aalpy.utils import compare_automata
from sklearn.cluster import estimate_bandwidth
from torch import optim

from Aut2RNNOneLayer import Dfa2RnnTransformer1Layer
from RNN import Optimization
from automata_data_generation import AutomatonDataset
from automata_data_generation import generate_data_from_automaton
from automata_data_generation import get_tomita
from clustering import compute_clusters_and_map_to_states
from clustering_comparison import compare_clustering_methods
from methods import conformance_test
from util import extract_hidden_states


def extract_cluster_automaton(automaton, model, test_seq, clustering_fun='mean_shift_8'):
    from aalpy.automata.Dfa import Dfa
    from aalpy.automata.Dfa import DfaState

    test_seq = test_seq.copy()
    hs_processing_fun = 'flatten' if model.model_type != 'lstm' else 'flatten_lstm'
    hidden_states = extract_hidden_states(model, test_seq, process_hs_fun=hs_processing_fun,
                                          save=False, load=False)

    mean_shift_default_bandwidth = None
    if 'mean_shift' in clustering_fun:
        mean_shift_default_bandwidth = estimate_bandwidth(hidden_states)

    cluster_labels = compute_clusters_and_map_to_states(automaton, test_seq, hidden_states, clustering_fun,
                                                        mean_shift_default_bandwidth, True)

    distinct_labels = set(cluster_labels)
    states = dict()
    for l in distinct_labels:
        states[l] = DfaState(l)

    init_state = DfaState('init')  # because initial state of RNN is not stored
    states['init'] = init_state

    i = 0

    max_label_exceeded = False
    for seq, out in test_seq:
        if max_label_exceeded:
            break

        current_state = init_state
        current_min_state = automaton.initial_state
        current_state.is_accepting = current_min_state.is_accepting
        for input in seq:
            next_cluster_label = cluster_labels[i]
            i += 1
            next_state = states[next_cluster_label]
            current_state.transitions[input] = next_state
            current_state = next_state
            current_min_state = current_min_state.transitions[input]
            current_state.is_accepting = current_min_state.is_accepting

            if i == len(cluster_labels):
                max_label_exceeded = True
                break

    return Dfa(init_state, list(states.values()))


automaton = get_tomita(5)

num_training_samples = 50 * 1000
num_val_samples = 2 * 1000

saturation_hidden, saturation_output, noise = 3, 3, 0.2

transformer = Dfa2RnnTransformer1Layer(automaton, saturation_hidden, saturation_output, noise, device=None)
rnn = transformer.transform()
rnn.model_name = f'noisy_tomita5'

training_data, input_al, output_al = generate_data_from_automaton(automaton, num_training_samples)
validation_data, _, _ = generate_data_from_automaton(automaton, num_val_samples)

data_handler = AutomatonDataset(input_al, output_al, batch_size=128, device=None)
train, val = data_handler.create_dataset(training_data), data_handler.create_dataset(validation_data)

optimizer = optim.Adam(rnn.parameters(), lr=0.0005, weight_decay=1e-6)
opt = Optimization(model=rnn, optimizer=optimizer, device=None)

opt.train(train, val, n_epochs=100, exp_name='visualization_retraining',
          verbose=True, early_stop=True, load=False, save=False)
rnn.eval()

conformance_test(rnn, automaton, min_test_len=6, max_test_len=15)
compare_clustering_methods(automaton, rnn, validation_data)

dbscan_dfa = extract_cluster_automaton(automaton, rnn, validation_data, 'DBSCAN')
mean_shift_4_dfa = extract_cluster_automaton(automaton, rnn, validation_data, 'mean_shift_4')
k_means_dfa_4 = extract_cluster_automaton(automaton, rnn, validation_data, 'k_means_4')

cluster_dfas = (('DBSCAN', dbscan_dfa), ('mean_shift_4', mean_shift_4_dfa), ('k_means_4', k_means_dfa_4))

for cf, cluster_dfa in cluster_dfas:
    print(f'----------------{cf}----------------')
    cluster_dfa.save(f'rnn_data/tomita5_cluster_dfa_{cf}')
    print(f'{cf} cluster DFA: {cluster_dfa.size} states.')

    input_al = automaton.get_input_alphabet()
    sul = DfaSUL(cluster_dfa)
    eq_oracle = RandomWMethodEqOracle(input_al, sul)
    minimized_cluster_dfa = run_Lstar(input_al, sul, eq_oracle, 'dfa', print_level=0)

    print(f'Minimized {cf} cluster automaton has {minimized_cluster_dfa.size} states.')
    cex = compare_automata(automaton, minimized_cluster_dfa)
    if cex:
        print(f'Extracted cluster automaton does not conform to the ground truth.')
        print(cex[0])
    else:
        print(f'Minimized {cf} cluster automaton conforms the ground-truth.')
