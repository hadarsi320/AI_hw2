import pysat

ids = ['318792827', '111111111']

sick = 'S'
sick_0 = 'S0'
sick_1 = 'S1'
sick_2 = 'S2'

healthy = 'H'

quarantined = 'Q'
quarantined_0 = 'Q_0'
quarantined_1 = 'Q_1'

immune_0 = 'I_0'
immune = 'I'

unpopulated = 'U'

unknown = '?'

sat_states = [sick_0, sick_1, sick_2, healthy, quarantined_0, quarantined_1, immune_0, immune, unpopulated]
query_states = [sick, healthy, immune, quarantined, unpopulated]


def get_neighbors(i, j, height, width):
    return [val for val in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] if in_board(*val, height, width)]


def in_board(i, j, height, width):
    return 0 <= i < height and 0 <= j < width




def actions_precondition_clauses(height, width, turns, num_police, num_medics):
    clauses = []
    for row in range(height):
        for col in range(width):
            clauses.extend(generate_clauses(row, col, height, width, num_police, num_medics))


def sick_clauses(turns, row, col, height, width, num_police, num_medics):
    clauses = []
    
    # In initial state
    for turn in range(1, turns):
        clauses.append([])


def generate_clauses(turns, row, col, height, width, num_police, num_medics):
    neighbors = get_neighbors(row, col, height, width)

    clauses = []
    clauses.extend(sick_clauses(turns, row, col, height, width, num_police, num_medics))

    return clauses


def solve_problem(input):
    pass
    # put your solution here, remember the format needed


class Solver:
    def __init__(self, input):
        self.num_police = input['police']
        self.num_medics = input['medics']
        observations = input['observations']
        self.num_turns = len(observations)  # TODO maybe max on queries
        self.height = len(observations[0])
        self.width = len(observations[0][0])
        self.prop2index, self.index2prop = self.initialize_propositions()

    def initialize_propositions(self):
        index_to_proposition = []
        for turn in range(self.num_turns):
            for row in range(self.height):
                for col in range(self.width):
                    for state in sat_states:
                        index_to_proposition.append((turn, row, col, state))

        proposition_to_index = {state: i for i, state in enumerate(index_to_proposition)}
        return proposition_to_index, index_to_proposition


