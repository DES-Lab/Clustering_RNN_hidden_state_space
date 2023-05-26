import pickle
from os import listdir

from aalpy.utils import load_automaton_from_file
from torch import optim

from Aut2RNNOneLayer import Dfa2RnnTransformer1Layer
from Aut2RNNTwoLayer import Dfa2RnnTransformer2Layers
from RNN import Optimization
from automata_data_generation import get_tomita, generate_data_from_automaton, AutomatonDataset
from clustering_comparison import compare_clustering_methods
from methods import conformance_test
from visualization_util import visualize_lda

experiments = [(get_tomita(3), 'tomita3'),
               (get_tomita(5), 'tomita5'),
               (get_tomita(7), 'tomita7'), ]

randomly_generated_dfa = [f for f in listdir('automata_models/') if f[:3] == 'dfa']
for aut in randomly_generated_dfa:
   experiments.append((load_automaton_from_file(f'automata_models/{aut}', automaton_type='dfa'), aut))

construction_methods = [('1_layer', Dfa2RnnTransformer1Layer), ]  # ('2_layers', Dfa2RnnTransformer2Layers),]
construction_methods.reverse()

device = None  # automatically determines

num_training_samples = 50 * 1000
num_val_samples = 2 * 1000

clustering_results = []

saturation_noise_values = ((3, 3, 0.2),(1.5, 1.5, 0.05))

for automaton, exp_name in experiments:
    print(exp_name)

    for saturation_hidden, saturation_output, noise in saturation_noise_values:
        print(saturation_hidden, saturation_output, noise)

        for construction_name, construction_function in construction_methods:
            transformer = construction_function(automaton, saturation_hidden, saturation_output, noise, device=device)
            rnn = transformer.transform()
            rnn.model_name = f'{exp_name}_{construction_name}_sh{saturation_hidden}_so{saturation_output}_n{noise}'

            training_data, input_al, output_al = generate_data_from_automaton(automaton, num_training_samples)
            validation_data, _, _ = generate_data_from_automaton(automaton, num_val_samples)

            data_handler = AutomatonDataset(input_al, output_al, batch_size=128, device=device)
            train, val = data_handler.create_dataset(training_data), data_handler.create_dataset(validation_data)

            optimizer = optim.Adam(rnn.parameters(), lr=0.0005, weight_decay=1e-6)
            opt = Optimization(model=rnn, optimizer=optimizer, device=device)

            opt.save(f'rnn_data/models/exp_models/{rnn.model_name}')

            results = compare_clustering_methods(automaton, rnn, validation_data)
            rnn.eval()
            clustering_results.append((f'{rnn.model_name}_noisy', results))

            rnn.train()
            # visualize_lda(rnn, automaton, validation_data, process_hs_fun='flatten')
            opt.train(train, val, n_epochs=250, exp_name=exp_name, verbose=True, early_stop=True, load=False, save=False)
            # visualize_lda(rnn, automaton, validation_data, process_hs_fun='flatten')
            conftest_res = conformance_test(rnn, automaton, min_test_len=6, max_test_len=14)

            opt.save(f'rnn_data/models/exp_models/{rnn.model_name}_retrained')

            rnn.eval()
            results = compare_clustering_methods(automaton, rnn, validation_data)
            results.append(("conf_test",conftest_res))
            clustering_results.append((f'{rnn.model_name}_retrained', results))


    with open(f'retraining_clustering_2.pickle', 'wb') as handle:
        pickle.dump(clustering_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
