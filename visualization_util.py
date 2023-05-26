import os
import re
import shutil
from collections import defaultdict
from os import makedirs

import imageio
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np

from dim_reduction import reduce_dimensions
from util import map_hidden_states_to_automaton, reduce_dim_of_state_hidden_state_map, map_hidden_states_to_clusters, \
    extract_hidden_states, map_tc_to_states_and_hs


def visualize_test_cases(test_cases, tc_hs_map, save_to_path=None, fig_name=None):
    if save_to_path:
        matplotlib.use('Agg')

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    if fig_name:
        ax.set_title(fig_name)

    state_points_map = defaultdict(list)
    for tc in test_cases:
        assert tc in tc_hs_map.keys()
        states, points = tc_hs_map[tc]
        for s, p in zip(states, points):
            state_points_map[s].append(p)

    for state, points in state_points_map.items():
        x, y = [i[0] for i in points], [i[1] for i in points]
        ax.scatter(x, y, label=state)

    for tc in test_cases:
        if len(tc) == 1:
            continue
        cor = tc_hs_map[tc][1]
        x, y = [i[0] for i in cor], [i[1] for i in cor]

        for i in range(0, len(x) - 1, 1):
            ax.arrow(x[i], y[i], x[i + 1] - x[i], y[i + 1] - y[i],
                     width=0.02, head_width=0.2, length_includes_head=True, overhang=1., )

    ax.legend()
    if not save_to_path:
        plt.show()
    else:
        makedirs(f'rnn_data/figures/{save_to_path}.png', exist_ok=True)
        plt.savefig(f'rnn_data/figures/{save_to_path}.png', format='png')

    plt.close('all')


def visualize_state_hs_map(data, save_to_path=None, fig_name='', save_tikz=False):
    if save_to_path:
        matplotlib.use('Agg')
    dims = len(list(data.values())[0][0])
    assert dims == 2 or dims == 3

    fig = plt.figure()
    if dims == 2:
        ax = fig.add_subplot()
    else:
        ax = fig.add_subplot(projection='3d')
        ax.set_zlabel('Z', fontsize=12)

    ax.set_xlabel('First LDA Component', fontsize=12)
    ax.set_ylabel('Second LDA Component', fontsize=12)
    if fig_name:
        ax.set_title(fig_name)

    for state_id, points in data.items():
        # if state_id == 'sink':
        #    continue
        coordinates = [[] for _ in range(dims)]
        for p in points:
            for i in range(dims):
                coordinates[i].append(p[i])
        ax.scatter(*coordinates, label=state_id)

    ax.legend()
    fig.tight_layout()
    if not save_to_path:
        plt.show()
    else:
        # makedirs(f'rnn_data/figures/{save_to_path}.png', exist_ok=True)
        makedirs(f'rnn_data/figures/', exist_ok=True)
        plt.savefig(f'rnn_data/figures/{save_to_path}.png', format='png', )
        if save_tikz:
            tikzplotlib.save(f'rnn_data/figures/{save_to_path}.tex')

    plt.close('all')


def visualize_hidden_states(nn_model, automaton, data, dim_red_fun, process_hs_fun='copy', map_hidden_to='state',
                            save_path=None, fig_name=None):
    assert map_hidden_to in {'state', 'state_input', 'input_new_state'}
    hss = map_hidden_states_to_automaton(nn_model, automaton, data, process_hs_fun=process_hs_fun,
                                         map_hidden_to=map_hidden_to, save=False, load=False)

    hss = reduce_dim_of_state_hidden_state_map(hss, dim_red_fun, dims=2, )

    visualize_state_hs_map(hss, save_path, fig_name)


def visualize_lda(nn_model, automaton, data, use_pca=False, process_hs_fun='copy', save_path=None, fig_name=None,
                  use_tikz=False):
    from clustering import compute_linear_separability_classifier

    hss = extract_hidden_states(nn_model, data, process_hs_fun, load=False, save=False)

    reduced_dim_pca = None
    if use_pca:
        reduced_dim_pca = reduce_dimensions(hss, 'pca', target_dimensions=2, )

    lda = compute_linear_separability_classifier(automaton, hss, data, reduce_dims=True)
    state_lda_map = defaultdict(list)

    hs_list = hss.copy()
    for i in range(len(hss)):
        hs = hs_list.pop(0)
        hs = np.array(hs)
        hs = hs.reshape(1, -1)

        state = lda.predict(hs)[0]
        coordinates = lda.transform(hs)[0] if not use_pca else reduced_dim_pca[i]

        state_lda_map[state].append(coordinates)

    visualize_state_hs_map(state_lda_map, save_path, fig_name, use_tikz)


def visualize_clusters(nn_model, data, clustering_fun, process_hs_fun='flatten', dim_reduction_fun='pca'):
    cluster_hs_map = map_hidden_states_to_clusters(nn_model, data, clustering_fun, dim_reduction_fun, process_hs_fun)

    visualize_state_hs_map(cluster_hs_map)


def visualize_hs_over_training(opt, automaton, train_val_data, data, epochs, save_intervals, exp_name,
                               dim_red_method='pca', train=True,
                               delete_aux_files=True):
    if train:
        opt.train(train_val_data[0], train_val_data[1], n_epochs=epochs, exp_name=exp_name, early_stop=False,
                  save_location='rnn_data/models/', save_interval=save_intervals, load=False)

    tmp_dir = 'tmpVisualizationHelper'
    os.makedirs(f'rnn_data/figures/{tmp_dir}', exist_ok=True)

    for i in range(save_intervals, epochs, save_intervals):
        weights_name = f'rnn_data/models/{opt.model.get_model_name()}_epoch_{i}'

        load_status = opt.load(weights_name)
        assert load_status

        if dim_red_method == 'pca':
            hs_processing_fun = 'flatten' if opt.model.model_type != 'lstm' else 'flatten_lstm'

            visualize_hidden_states(nn_model=opt.model, automaton=automaton, data=data,
                                    dim_red_fun='pca', process_hs_fun=hs_processing_fun,
                                    save_path=f'{tmp_dir}/{exp_name}_{opt.model.model_type}_{i}',
                                    fig_name=f'{exp_name}_{dim_red_method}_{opt.model.model_type}_{i}')
        else:
            visualize_lda(opt.model, automaton, data, process_hs_fun=hs_processing_fun,
                          save_path=f'{tmp_dir}/{exp_name}_{opt.model.model_type}_{i}',
                          fig_name=f'{exp_name}_{dim_red_method}_{opt.model.model_type}_{i}')
        if delete_aux_files:
            os.remove(weights_name)

    # sort images according to epoch
    images = os.listdir(f'rnn_data/figures/{tmp_dir}')
    sorted_images = []
    for i in images:
        image_num = int(re.search(f'{exp_name}_{opt.model.model_type}_(.*).png', i).group(1))
        sorted_images.append((image_num, i))
    sorted_images.sort(key=lambda x: x[0])
    images = [i[1] for i in sorted_images]

    gif_images = []
    for filename in images:
        gif_images.append(imageio.imread(f'rnn_data/figures/{tmp_dir}/{filename}'))

    imageio.mimsave(f'training_process_gifs/{exp_name}_{opt.model.model_type}.gif', gif_images, duration=0.15)
    print(f'Hidden states of the training process saved to training_process_gifs/{exp_name}_{opt.model.model_type}.gif')

    if delete_aux_files:
        shutil.rmtree(f'rnn_data/figures/{tmp_dir}')


def visualize_test_cases_over_time(test_cases, nn_model, automaton, data):
    hs = extract_hidden_states(nn_model, data)

    hs = reduce_dimensions(hs, 'pca', target_dimensions=2)

    tc_hs_map = map_tc_to_states_and_hs(data, hs, automaton)

    visualize_test_cases(test_cases, tc_hs_map)


def visualize_constructed_rnn_noise_and_retrained():
    from automata_data_generation import get_tomita
    from automata_data_generation import generate_data_from_automaton
    from torch import optim
    from RNN import Optimization
    from Aut2RNNOneLayer import Dfa2RnnTransformer1Layer
    from methods import conformance_test
    from automata_data_generation import AutomatonDataset
    from clustering_comparison import compare_clustering_methods

    automaton = get_tomita(5)

    visualization_method = 'lda'
    save_path = 'paper_tomita5'
    save_to_tikz = False

    num_training_samples = 50 * 1000
    num_val_samples = 2 * 1000

    saturation_hidden, saturation_output, noise = 3, 3, 0.2

    transformer = Dfa2RnnTransformer1Layer(automaton, saturation_hidden, saturation_output, 0, device=None)
    no_noise_rnn = transformer.transform()

    no_noise_rnn.model_name = 'no_noise'

    transformer = Dfa2RnnTransformer1Layer(automaton, saturation_hidden, saturation_output, noise, device=None)
    rnn = transformer.transform()
    rnn.model_name = f'visualization_of_computed'

    training_data, input_al, output_al = generate_data_from_automaton(automaton, num_training_samples)
    validation_data, _, _ = generate_data_from_automaton(automaton, num_val_samples)

    data_handler = AutomatonDataset(input_al, output_al, batch_size=128, device=None)
    train, val = data_handler.create_dataset(training_data), data_handler.create_dataset(validation_data)

    optimizer = optim.Adam(rnn.parameters(), lr=0.0005, weight_decay=1e-6)
    opt = Optimization(model=rnn, optimizer=optimizer, device=None)

    opt.save(f'rnn_data/models/exp_models/{rnn.model_name}')

    rnn.eval()

    if visualization_method == 'lda':
        visualize_lda(no_noise_rnn, automaton, validation_data, process_hs_fun='flatten',
                      fig_name='Correct-by-construction RNN',
                      save_path=f'{save_path}_constructed', use_tikz=save_to_tikz)
        visualize_lda(rnn, automaton, validation_data, process_hs_fun='flatten',
                      fig_name='RNN with Gaussian Noise',
                      save_path=f'{save_path}_noisy', use_tikz=save_to_tikz)
    else:
        visualize_hidden_states(no_noise_rnn, automaton, validation_data, 'pca')
        visualize_hidden_states(rnn, automaton, validation_data, 'pca')

    rnn.train()
    opt.train(train, val, n_epochs=100, exp_name='visualization_retraining',
              verbose=True, early_stop=True, load=False, save=False)

    rnn.eval()

    if visualization_method == 'lda':
        visualize_lda(rnn, automaton, validation_data, process_hs_fun='flatten',
                      fig_name='Retrained RNN',
                      save_path=f'{save_path}_retrained',
                      use_tikz=save_to_tikz)
    else:
        visualize_hidden_states(rnn, automaton, validation_data, 'pca')

    conformance_test(rnn, automaton, min_test_len=6, max_test_len=15)

    print('No noise')
    compare_clustering_methods(automaton, no_noise_rnn, validation_data)
    print('Retrained')
    compare_clustering_methods(automaton, rnn, validation_data)


if __name__ == '__main__':
    visualize_constructed_rnn_noise_and_retrained()
