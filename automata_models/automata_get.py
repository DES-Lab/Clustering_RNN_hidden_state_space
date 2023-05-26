from os import listdir, remove

from aalpy.utils import generate_random_dfa, generate_random_moore_machine, load_automaton_from_file
from aalpy.oracles import StatePrefixEqOracle
from aalpy.learning_algs import run_Lstar
from aalpy.SULs import DfaSUL, MooreSUL


def random_dfas():
    sizes = (5, 10)
    input_sizes = (2, 4, 6)
    num_automata_per_config = 5

    for model_size in sizes:
        for input_size in input_sizes:
            num_generated_models = 0
            while num_generated_models < num_automata_per_config:

                input_al = list(range(1, input_size + 1))
                num_accepting_states = 2 if model_size == 5 else 4
                model = generate_random_dfa(model_size, input_al, num_accepting_states)

                try:
                    char_set = model.compute_characterization_set()
                    num_generated_models += 1
                    model.save(f'dfa_size_{model_size}_inputs_{input_size}_{num_generated_models + 1}')
                except SystemExit:
                    continue


def minimize_dfa():
    dfas = []
    randomly_generated_dfa = [f for f in listdir('.') if f[:3] == 'dfa']
    for aut in randomly_generated_dfa:
        dfas.append((load_automaton_from_file(f'{aut}', automaton_type='dfa'), aut))

    num_generated_models = 0
    for dfa, path in dfas:
        num_generated_models += 1
        input_al = dfa.get_input_alphabet()
        sul = DfaSUL(dfa)
        eq_oracle = StatePrefixEqOracle(input_al, sul)
        minimal_model = run_Lstar(input_al, sul, eq_oracle, 'dfa', print_level=0)
        if minimal_model.size > 2:
            minimal_model.save(f'dfa_size_{minimal_model.size}_inputs_{len(input_al)}_{num_generated_models}')
            remove(path)


def random_moore():
    sizes = (8, 12)
    input_sizes = (2, 4,)
    output_sizes = (3, 5)
    num_automata_per_config = 3

    for model_size in sizes:
        for input_size in input_sizes:
            for output_size in output_sizes:
                num_generated_models = 0
                while num_generated_models < num_automata_per_config:

                    input_al = list(range(1, input_size + 1))
                    output_al = list(range(1, output_size + 1))
                    model = generate_random_moore_machine(model_size, input_al, output_al)

                    try:
                        char_set = model.compute_characterization_set()
                        num_generated_models += 1
                        model.save(
                            f'moore_size_{model_size}_inputs_{input_size}_output{output_size}_{num_generated_models + 1}')
                    except SystemExit:
                        continue


minimize_dfa()
