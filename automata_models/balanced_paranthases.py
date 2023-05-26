from aalpy.base import SUL
from aalpy.learning_algs import run_Lstar
from aalpy.oracles import RandomWordEqOracle, StatePrefixEqOracle

input_al = ['(', ')']


def is_accepting(test_str):
    brackets = ['()', '{}', '[]']
    while any(x in test_str for x in brackets):
        for br in brackets:
            test_str = test_str.replace(br, '')
    return not test_str


class BpSUL(SUL):
    def __init__(self):
        super().__init__()
        self.curr_string = ''

    def pre(self):
        self.curr_string = ''

    def post(self):
        pass

    def step(self, letter):
        if letter is None:
            return True
        self.curr_string += letter

        return is_accepting(self.curr_string)


sul = BpSUL()
eq_oracle = RandomWordEqOracle(input_al, sul, num_walks=100000, min_walk_len=5, max_walk_len=32)
eq_oracle = StatePrefixEqOracle(input_al, sul, walks_per_state=100)

model = run_Lstar(input_al, sul, eq_oracle, automaton_type='dfa')
model.save('bp_depth_16')
