from collections import defaultdict
from itertools import product
import random

from aalpy.base import Automaton, AutomatonState, SUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWMethodEqOracle

from automata_data_generation import generate_data_from_automaton


class PdaState(AutomatonState):
    """
    Single state of a deterministic finite automaton.
    """

    def __init__(self, state_id, is_accepting=False):
        super().__init__(state_id)
        self.transitions = defaultdict(list)
        self.is_accepting = is_accepting


from enum import Enum


# class syntax
class Action(Enum):
    PUSH = 1
    POP = 2
    NOTHING = 3
    PUSHTWICE = 4
    POPTWICE = 5


class Transition:
    def __init__(self, start: PdaState, target: PdaState, symbol, action, stack_guard=None):
        self.start = start
        self.target = target
        self.symbol = symbol
        self.action = action
        self.stack_guard = stack_guard


class Pda(Automaton):
    empty = "$"
    error_state = PdaState("err", False)

    def __init__(self, initial_state: PdaState, states):
        super().__init__(initial_state, states)
        self.initial_state = initial_state
        self.states = states
        self.current_state = None
        self.config = []

    def reset_to_initial(self):
        super().reset_to_initial()
        self.reset()

    def reset(self):
        self.current_state = self.initial_state
        self.config = [self.empty]
        return self.current_state.is_accepting and self.top() == self.empty

    def top(self):
        return self.config[-1]

    def possible(self, letter):
        if self.current_state == Pda.error_state:
            return True
        if letter is not None:
            transitions = self.current_state.transitions[letter]
            trans = [t for t in transitions if t.stack_guard is None or self.top() == t.stack_guard]
            assert len(trans) < 2
            if len(trans) == 0:
                return False
            else:
                return True
        return False

    def step(self, letter):
        """
        Args:

            letter: single input that is looked up in the transition table of the DfaState

        Returns:

            True if the reached state is an accepting state, False otherwise
        """
        if self.current_state == Pda.error_state:
            return False
        if not self.possible(letter):
            self.current_state = Pda.error_state
            return False
        if letter is not None:
            transitions = self.current_state.transitions[letter]
            trans = [t for t in transitions if t.stack_guard is None or self.top() == t.stack_guard][0]
            self.current_state = trans.target
            if trans.action == Action.PUSH:
                self.config.append(letter)
            if trans.action == Action.PUSHTWICE:
                self.config.append(letter)
                self.config.append(letter)
            elif trans.action == Action.POP:
                if len(self.config) <= 1:  # empty stack elem should always be there
                    self.current_state = Pda.error_state
                    return False
                self.config.pop()
            elif trans.action == Action.POPTWICE:
                if len(self.config) <= 2:
                    self.current_state = Pda.error_state
                    return False
                self.config.pop()
                self.config.pop()

        return self.current_state.is_accepting and self.top() == self.empty

    # def compute_output_seq(self, state, sequence):
    #     if not sequence:
    #         return [state.is_accepting]
    #     return super(Dfa, self).compute_output_seq(state, sequence)

    def to_state_setup(self):
        state_setup_dict = {}

        # ensure prefixes are computed
        # self.compute_prefixes()

        sorted_states = sorted(self.states, key=lambda x: len(x.prefix))
        for s in sorted_states:
            state_setup_dict[s.state_id] = (
            s.is_accepting, {k: (v.target.state_id, v.action) for k, v in s.transitions.items()})

        return state_setup_dict

    @staticmethod
    def from_state_setup(state_setup: dict, init_state_id):
        """
            First state in the state setup is the initial state.
            Example state setup:
            state_setup = {
                    "a": (True, {"x": ("b1",PUSH), "y": ("a", NONE)}),
                    "b1": (False, {"x": ("b2", PUSH), "y": "a"}),
                    "b2": (True, {"x": "b3", "y": "a"}),
                    "b3": (False, {"x": "b4", "y": "a"}),
                    "b4": (False, {"x": "c", "y": "a"}),
                    "c": (True, {"x": "a", "y": "a"}),
                }

            Args:

                state_setup: map from state_id to tuple(output and transitions_dict)

            Returns:

                PDA
            """
        # state_setup should map from state_id to tuple(is_accepting and transitions_dict)

        # build states with state_id and output
        states = {key: PdaState(key, val[0]) for key, val in state_setup.items()}
        states[Pda.error_state.state_id] = Pda.error_state  # PdaState(Pda.error_state,False)
        # add transitions to states
        for state_id, state in states.items():
            if state_id == Pda.error_state.state_id:
                continue
            for _input, trans_spec in state_setup[state_id][1].items():
                for (target_state_id, action, stack_guard) in trans_spec:
                    # action = Action[action_string]
                    trans = Transition(start=state, target=states[target_state_id], symbol=_input, action=action,
                                       stack_guard=stack_guard)
                    state.transitions[_input].append(trans)

        init_state = states[init_state_id]
        # states to list
        states = [state for state in states.values()]

        pda = Pda(init_state, states)
        return pda


class PdaSUL(SUL):
    """
    System under learning for DFAs.
    """

    def __init__(self, pda: Pda):
        super().__init__()
        self.pda = pda

    def pre(self):
        """
        Resets the dfa to the initial state.
        """
        self.pda.reset_to_initial()

    def post(self):
        pass

    def step(self, letter):
        """
        If the letter is empty/None check is preform to see if the empty string is accepted by the DFA.

        Args:

            letter: single input or None representing the empty string

        Returns:

            output of the dfa.step method (whether the next state is accepted or not)

        """
        return self.pda.step(letter)


def generate_data_from_pda(automaton, num_examples, lens=None, classify_states=False, stack_limit=None,
                           break_on_impossible=False, possible_prob=0.75):
    input_al = automaton.get_input_alphabet()
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
            for seq in seqs:

                out = automaton.reset()
                nr_steps = 0
                for inp in seq:
                    if automaton.possible(inp) or not break_on_impossible:
                        nr_steps += 1
                    if stack_limit and len(automaton.config) > stack_limit:
                        break
                    if break_on_impossible and not automaton.possible(inp):
                        break
                    out = automaton.step(inp)
                seq = seq[:nr_steps]
                training_data.append((tuple(seq), out if not classify_states else automaton.current_state.state_id))

        else:
            for _ in range(ex_per_len[l] + additional_seq):
                # seq = [random.choice(input_al) for _ in range(l)]
                out = automaton.reset()
                nr_steps = 0
                seq = []
                for i in range(l):
                    possible_inp = [inp for inp in input_al if automaton.possible(inp)]
                    if len(possible_inp) == 0:
                        inp = random.choice(input_al)
                    else:
                        if random.random() <= possible_prob:
                            inp = random.choice(possible_inp)
                        else:
                            inp = random.choice(input_al)
                    seq.append(inp)
                    if automaton.possible(inp) or not break_on_impossible:
                        nr_steps += 1
                    if stack_limit and len(automaton.config) > stack_limit:
                        break
                    if break_on_impossible and not automaton.possible(inp):
                        break
                    out = automaton.step(inp)
                seq = seq[:nr_steps]
                training_data.append((tuple(seq), out if not classify_states else automaton.current_state.state_id))

    return training_data, input_al, output_al


def sample_pda(pda: Pda, n_samples):
    sample_data = generate_data_from_pda(pda, n_samples, stack_limit=3)
    return sample_data


def pda_for_L1():
    # we always ensure that n >= 1
    state_setup = {
        "q0": (False, {"a": [("q1", Action.PUSH, None)], "b": [(Pda.error_state.state_id, Action.NOTHING, None)]}),
        "q1": (False, {"a": [("q1", Action.PUSH, None)], "b": [("q2", Action.POP, "a")]}),
        "q2": (True, {"a": [(Pda.error_state.state_id, Action.NOTHING, None)], "b": [("q2", Action.POP, "a")]}),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L2():
    state_setup = {
        "q0": (False, {"a": [("q1", Action.PUSH, None)], "b": [("q1", Action.PUSH, None)],
                       "c": [(Pda.error_state.state_id, Action.NOTHING, None)],
                       "d": [(Pda.error_state.state_id, Action.NOTHING, None)]}),
        "q1": (False, {"a": [("q1", Action.PUSH, None)], "b": [("q1", Action.PUSH, None)],
                       "c": [("q2", Action.POP, "a"), ("q2", Action.POP, "b")],
                       "d": [("q2", Action.POP, "a"), ("q2", Action.POP, "b")]}),
        "q2": (True, {"a": [(Pda.error_state.state_id, Action.NOTHING, None)],
                      "b": [(Pda.error_state.state_id, Action.NOTHING, None)],
                      "c": [("q2", Action.POP, "a"), ("q2", Action.POP, "b")],
                      "d": [("q2", Action.POP, "a"), ("q2", Action.POP, "b")]}),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L3():
    state_setup = {
        "q0": (False, {"a": [("q0a", Action.PUSH, None)],
                       "c": [("q0c", Action.PUSH, None)],
                       }),
        "q0a": (False, {"b": [("q1", Action.PUSH, None)]}),
        "q0c": (False, {"d": [("q1", Action.PUSH, None)]}),
        "q1": (False, {"a": [("q1a", Action.PUSH, None)],
                       "c": [("q1c", Action.PUSH, None)],
                       "e": [("q2e", Action.POP, "b"), ("q2e", Action.POP, "d")],
                       "g": [("q2g", Action.POP, "b"), ("q2g", Action.POP, "d")],  # stack should actually be redundant
                       }),
        "q1a": (False, {"b": [("q1", Action.PUSH, None)]}),
        "q1c": (False, {"d": [("q1", Action.PUSH, None)]}),
        "q2e": (False, {"f": [("q2", Action.POP, "a"), ("q2", Action.POP, "c")]}),
        "q2g": (False, {"h": [("q2", Action.POP, "a"), ("q2", Action.POP, "c")]}),
        "q2": (True, {"e": [("q2e", Action.POP, "b"), ("q2e", Action.POP, "d")],
                      "g": [("q2g", Action.POP, "b"), ("q2g", Action.POP, "d")]})
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L4():
    state_setup = {
        "q0": (False, {"a": [("q01", Action.PUSH, None)], "b": [(Pda.error_state.state_id, Action.NOTHING, None)]}),
        "q01": (False, {"b": [("q1", Action.PUSH, None)], "a": [(Pda.error_state.state_id, Action.NOTHING, None)]}),

        "q1": (False, {"a": [("q11", Action.PUSH, None)], "b": [(Pda.error_state.state_id, Action.NOTHING, None)],
                       "c": [("q21", Action.POP, "b")]}),
        "q11": (False, {"b": [("q1", Action.PUSH, None)], "a": [(Pda.error_state.state_id, Action.NOTHING, None)]}),
        "q21": (False, {"d": [("q2", Action.POP, "a")]}),
        "q2": (True, {"c": [("q21", Action.POP, "b")]}),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L5():
    state_setup = {
        "q0": (False, {"a": [("q01", Action.PUSH, None)]}),
        "q01": (False, {"b": [("q02", Action.PUSH, None)]}),
        "q02": (False, {"c": [("q1", Action.PUSH, None)]}),
        "q1": (False, {"a": [("q11", Action.PUSH, None)],
                       "d": [("q21", Action.POP, "c")]}),
        "q11": (False, {"b": [("q12", Action.PUSH, None)]}),
        "q12": (False, {"c": [("q1", Action.PUSH, None)]}),
        "q21": (False, {"e": [("q22", Action.POP, "b")]}),
        "q22": (False, {"f": [("q2", Action.POP, "a")]}),
        "q2": (True, {"d": [("q21", Action.POP, "c")]}),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L6():
    state_setup = {
        "q0": (False, {"a": [("q0a", Action.PUSH, None)],
                       "c": [("q1", Action.PUSHTWICE, None)],
                       }),
        "q0a": (False, {"b": [("q1", Action.PUSH, None)]}),
        "q1": (False, {"a": [("q1a", Action.PUSH, None)],
                       "c": [("q1", Action.PUSHTWICE, None)],
                       "d": [("q2d", Action.POP, None)],
                       "f": [("q2", Action.POPTWICE, None)],
                       }),
        "q1a": (False, {"b": [("q1", Action.PUSH, None)]}),

        "q2d": (False, {"e": [("q2", Action.POP, None)]}),
        "q2": (True, {"d": [("q2d", Action.POP, None)],
                      "f": [("q2", Action.POPTWICE, None)]})
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L7():
    # Dyck order 2
    state_setup = {
        "q0": (False, {"(": [("q1", Action.PUSH, None)],
                       "[": [("q1", Action.PUSH, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", Action.PUSH, None)],
                      "[": [("q1", Action.PUSH, None)],
                      ")": [("q1", Action.POP, "(")],
                      "]": [("q1", Action.POP, "[")]
                      }),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L8():
    # Dyck order 3
    state_setup = {
        "q0": (False, {"(": [("q1", Action.PUSH, None)],
                       "[": [("q1", Action.PUSH, None)],
                       "{": [("q1", Action.PUSH, None)],
                       }),
        "q1": (True, {"(": [("q1", Action.PUSH, None)],
                      "[": [("q1", Action.PUSH, None)],
                      "{": [("q1", Action.PUSH, None)],
                      ")": [("q1", Action.POP, "(")],
                      "]": [("q1", Action.POP, "[")],
                      "}": [("q1", Action.POP, "{")],
                      }),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L9():
    # Dyck order 4
    state_setup = {
        "q0": (False, {"(": [("q1", Action.PUSH, None)],
                       "[": [("q1", Action.PUSH, None)],
                       "{": [("q1", Action.PUSH, None)],
                       "<": [("q1", Action.PUSH, None)],
                       }),
        "q1": (True, {"(": [("q1", Action.PUSH, None)],
                      "[": [("q1", Action.PUSH, None)],
                      "{": [("q1", Action.PUSH, None)],
                      "<": [("q1", Action.PUSH, None)],
                      ")": [("q1", Action.POP, "(")],
                      "]": [("q1", Action.POP, "[")],
                      "}": [("q1", Action.POP, "{")],
                      ">": [("q1", Action.POP, "{")],
                      }),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L10():
    # RE Dyck order 1
    state_setup = {
        "q0": (False, {"a": [("qa", Action.PUSH, None)],
                       }),
        "qa": (False, {"b": [("qb", Action.NOTHING, None)],
                       }),
        "qb": (False, {"c": [("qc", Action.NOTHING, None)],
                       }),
        "qc": (False, {"d": [("qd", Action.NOTHING, None)],
                       }),
        "qd": (False, {"e": [("q1", Action.NOTHING, None)],
                       }),
        "q1": (True, {"a": [("qa", Action.PUSH, None)],
                      "v": [("qv", Action.POP, "a")]}),
        "qv": (False, {"w": [("qw", Action.NOTHING, None)]}),
        "qw": (False, {"x": [("qx", Action.NOTHING, None)]}),
        "qx": (False, {"y": [("qy", Action.NOTHING, None)]}),
        "qy": (False, {"z": [("q1", Action.NOTHING, None)]})
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L11():
    # RE Dyck order 1
    state_setup = {
        "q0": (False, {"a": [("qa", Action.PUSH, None)],
                       "c": [("q1", Action.PUSH, None)],
                       }),
        "qa": (False, {"b": [("q1", Action.NOTHING, None)],
                       }),
        "q1": (True, {"a": [("qa", Action.PUSH, None)],
                      "c": [("q1", Action.PUSH, None)],
                      "d": [("qd", Action.POP, "a"), ("qd", Action.POP, "c")],
                      "f": [("q1", Action.POP, "a"), ("q1", Action.POP, "c")]}),
        "qd": (False, {"e": [("q1", Action.NOTHING, None)]})
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L12():
    # Dyck order 2 (single-nested)
    state_setup = {
        "q0": (False, {"(": [("q1", Action.PUSH, None)],
                       "[": [("q1", Action.PUSH, None)],  # exclude empty seq
                       }),
        "q1": (False, {"(": [("q1", Action.PUSH, None)],
                       "[": [("q1", Action.PUSH, None)],
                       ")": [("q2", Action.POP, "(")],
                       "]": [("q2", Action.POP, "[")]}),
        "q2": (True, {
            ")": [("q2", Action.POP, "(")],
            "]": [("q2", Action.POP, "[")]
        }),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L13():
    # Dyck order 1
    state_setup = {
        "q0": (False, {"(": [("q1", Action.PUSH, None)],
                       "a": [("q1", Action.NOTHING, None)],
                       "b": [("q1", Action.NOTHING, None)],
                       "c": [("q1", Action.NOTHING, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", Action.PUSH, None)],
                      ")": [("q1", Action.POP, "(")],
                      "a": [("q1", Action.NOTHING, None)],
                      "b": [("q1", Action.NOTHING, None)],
                      "c": [("q1", Action.NOTHING, None)]
                      }),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L14():
    # Dyck order 2
    state_setup = {
        "q0": (False, {"(": [("q1", Action.PUSH, None)],
                       "[": [("q1", Action.PUSH, None)],
                       "a": [("q1", Action.NOTHING, None)],
                       "b": [("q1", Action.NOTHING, None)],
                       "c": [("q1", Action.NOTHING, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", Action.PUSH, None)],
                      "[": [("q1", Action.PUSH, None)],
                      ")": [("q1", Action.POP, "(")],
                      "]": [("q1", Action.POP, "[")],
                      "a": [("q1", Action.NOTHING, None)],
                      "b": [("q1", Action.NOTHING, None)],
                      "c": [("q1", Action.NOTHING, None)]
                      }),
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda


def pda_for_L15():
    # Dyck order 1
    state_setup = {
        "q0": (False, {"(": [("q1", Action.PUSH, None)],
                       "a": [("qa", Action.NOTHING, None)],
                       "d": [("q1", Action.NOTHING, None)],  # exclude empty seq
                       }),
        "q1": (True, {"(": [("q1", Action.PUSH, None)],
                      ")": [("q1", Action.POP, "(")],
                      "a": [("qa", Action.NOTHING, None)],
                      "d": [("q1", Action.NOTHING, None)],
                      }),
        "qa": (False, {"b": [("qb", Action.NOTHING, None)],
                       }),
        "qb": (False, {"c": [("q1", Action.NOTHING, None)],
                       })
    }
    pda = Pda.from_state_setup(state_setup, "q0")
    return pda

