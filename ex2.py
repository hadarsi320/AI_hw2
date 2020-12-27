from pysat.card import CardEnc, EncType
from pysat.formula import IDPool

ids = ['318792827', '111111111']

UNPOPULATED = 'U'

SICK = 'S'
SICK_0 = 'S0'
SICK_1 = 'S1'
SICK_2 = 'S2'

HEALTHY = 'H'

QUARANTINED = 'Q'
QUARANTINED_0 = 'Q0'
QUARANTINED_1 = 'Q1'

IMMUNE_RECENTLY = 'IX'
IMMUNE = 'I'

UNK = '?'

STATES = [SICK_0, SICK_1, SICK_2, HEALTHY, QUARANTINED_0, QUARANTINED_1, IMMUNE_RECENTLY, IMMUNE, UNPOPULATED]
# FIRST_TURN_IMPOSSIBLE_STATES = [SICK_0, SICK_1, QUARANTINED_0, QUARANTINED_1, IMMUNE_RECENTLY, IMMUNE]
FIRST_TURN_POSSIBLE_STATES = [SICK_2, HEALTHY, UNPOPULATED]
SECOND_TURN_POSSIBLE_STATES = [SICK_1, SICK_2, HEALTHY, QUARANTINED_1, IMMUNE_RECENTLY, UNPOPULATED]
QUERY_STATES = [SICK, HEALTHY, IMMUNE, QUARANTINED, UNPOPULATED]

INPUT2PROPS = {
    'S': {'obs': SICK_2, 'not': (HEALTHY, UNPOPULATED)},
    'H': {'obs': HEALTHY, 'not': (SICK_2, UNPOPULATED)},
    'U': {'obs': UNPOPULATED, 'not': (HEALTHY, SICK_2)}
}


def solve_problem(input):
    solver = Solver(input)
    # put your solution here, remember the format needed


class Solver:
    def __init__(self, solver_input):
        self.num_police = solver_input['police']
        self.num_medics = solver_input['medics']
        self.observations = solver_input['observations']
        self.num_turns = len(self.observations)  # TODO maybe max on queries
        self.height = len(self.observations[0])
        self.width = len(self.observations[0][0])
        self.prop2index, self.index2prop = self.initialize_propositions()
        # self.clauses = self.generate_clauses()

    def initialize_propositions(self):
        index_to_proposition = [0]
        for turn in range(self.num_turns):
            for row in range(self.height):
                for col in range(self.width):
                    for state in STATES:
                        index_to_proposition.append((turn, row, col, state))

        proposition_to_index = {state: i for i, state in enumerate(index_to_proposition)}
        return proposition_to_index, index_to_proposition

    def generate_clauses(self):
        clauses = []
        clauses.extend(self.generate_observations_clauses())
        clauses.extend(self.generate_validity_clauses())  # TODO
        clauses.extend(self.generate_dynamics_clauses())
        clauses.extend(self.generate_valid_actions_clauses())  # TODO
        return clauses

    def generate_observations_clauses(self):
        clauses = []

        for turn, observation in enumerate(self.observations):
            for row in range(self.height):
                for col in range(self.width):
                    state = observation[row][col]
                    if state == SICK:
                        clauses.append([self.prop2index[(turn, row, col, SICK_0)],
                                        self.prop2index[(turn, row, col, SICK_1)],
                                        self.prop2index[(turn, row, col, SICK_2)]])
                    elif state == QUARANTINED:
                        clauses.append([self.prop2index[(turn, row, col, QUARANTINED_0)],
                                        self.prop2index[(turn, row, col, QUARANTINED_1)]])
                    elif state == IMMUNE:
                        clauses.append([self.prop2index[(turn, row, col, IMMUNE_RECENTLY)],
                                        self.prop2index[(turn, row, col, IMMUNE)]])
                    elif state == UNK:
                        continue
                    else:
                        clauses.append([self.prop2index[(turn, row, col, state)]])

        return clauses

    def generate_validity_clauses(self):
        clauses = []
        for row in range(self.height):
            for col in range(self.width):
                clauses.extend(self.first_turn_clauses(row, col))
                clauses.extend(self.second_turn_clauses(row, col))
                clauses.extend(self.uniqueness_clauses(row, col))

        return clauses

    def first_turn_clauses(self, row, col):
        lits = [self.prop2index[(0, row, col, state)] for state in FIRST_TURN_POSSIBLE_STATES]
        clauses = CardEnc.equals(lits, bound=1, encoding=EncType.pairwise).clauses
        return clauses

    def second_turn_clauses(self, row, col):
        lits = [self.prop2index[(1, row, col, state)] for state in SECOND_TURN_POSSIBLE_STATES]
        clauses = CardEnc.equals(lits, bound=1, encoding=EncType.pairwise).clauses
        return clauses

    def uniqueness_clauses(self, row, col):
        clauses = []
        for turn in range(2, self.num_turns):
            lits = [self.prop2index[(1, row, col, state)] for state in STATES]
            clauses.extend(CardEnc.equals(lits, bound=1, encoding=EncType.pairwise).clauses)
        return clauses

    def generate_dynamics_clauses(self):
        clauses = []
        for turn in range(self.num_turns):
            for row in range(self.height):
                for col in range(self.width):
                    clauses.extend(self.unpopulated_clauses(turn, row, col))
                    clauses.extend(self.sick_clauses(turn, row, col))
                    clauses.extend(self.healthy_clauses(turn, row, col))
                    clauses.extend(self.immune_clauses(turn, row, col))
                    clauses.extend(self.quarantine_clauses(turn, row, col))

        return clauses

    def unpopulated_clauses(self, turn, row, col):
        clauses = []

        # Previous Turn
        if 0 < turn:
            # U_t = > U_t-1
            clauses.append([-self.prop2index[(turn, row, col, UNPOPULATED)],
                            self.prop2index[(turn - 1, row, col, UNPOPULATED)]])

        # Next Turn
        if turn < self.num_turns - 1:
            # U_t => U_t+1
            clauses.append([-self.prop2index[(turn, row, col, UNPOPULATED)],
                            self.prop2index[(turn + 1, row, col, UNPOPULATED)]])

        return clauses

    def sick_clauses(self, turn, row, col):
        clauses = []
        neighbors = self.get_neighbors(row, col)

        # Is sick
        # Previous Turn
        if 0 < turn:
            # S2_t => H_t-1
            clauses.append([-self.prop2index[(turn, row, col, SICK_2)],
                            self.prop2index[(turn - 1, row, col, HEALTHY)]])
            # S1_t => S2_t-1
            clauses.append([-self.prop2index[(turn, row, col, SICK_1)],
                            self.prop2index[(turn - 1, row, col, SICK_2)]])
            # S0_t => S1_t-1
            clauses.append([-self.prop2index[(turn, row, col, SICK_0)],
                            self.prop2index[(turn - 1, row, col, SICK_1)]])

        # Next Turn
        if turn < self.num_turns - 1:
            # S2_t => S1_t+1 v Q1_t+1
            clauses.append([-self.prop2index[(turn, row, col, SICK_2)],
                            self.prop2index[(turn + 1, row, col, SICK_1)],
                            self.prop2index[(turn + 1, row, col, QUARANTINED_1)]])
            # S1_t => S0_t+1 v Q1_t+1
            clauses.append([-self.prop2index[(turn, row, col, SICK_1)],
                            self.prop2index[(turn + 1, row, col, SICK_0)],
                            self.prop2index[(turn + 1, row, col, QUARANTINED_1)]])
            # S0_t => H_t+1 v Q1_t+1
            clauses.append([-self.prop2index[(turn, row, col, SICK_0)],
                            self.prop2index[(turn + 1, row, col, HEALTHY)],
                            self.prop2index[(turn + 1, row, col, QUARANTINED_1)]])

        # Infected By Someone
        if 0 < turn:
            # S2_t => V (S2n_t-1 v S1n_t-1 v S0n_t-1) for n in neighbors
            clause = [-self.prop2index[(turn, row, col, SICK_2)]]
            for (n_row, n_col) in neighbors:
                clause.extend([self.prop2index[(turn - 1, n_row, n_col, SICK_2)],
                               self.prop2index[(turn - 1, n_row, n_col, SICK_1)],
                               self.prop2index[(turn - 1, n_row, n_col, SICK_0)]])
            clauses.append(clause)

        # Infecting Others
        if turn < self.num_turns - 1:
            for (n_row, n_col) in neighbors:
                for sick_i in [SICK_0, SICK_1, SICK_2]:
                    # Si_t /\ Hn_t /\ -Q1_t+1 /\ -I_recent_t+1 => S2n_t+1 (Sn, Hn stand for neighbor)
                    clauses.append([-self.prop2index[(turn, row, col, sick_i)],
                                    -self.prop2index[(turn, n_row, n_col, HEALTHY)],
                                    self.prop2index[(turn + 1, row, col, QUARANTINED_1)],
                                    self.prop2index[(turn + 1, n_row, n_col, IMMUNE_RECENTLY)],
                                    self.prop2index[(turn + 1, n_row, n_col, SICK_2)]])

        return clauses

    def healthy_clauses(self, turn, row, col):
        clauses = []

        # Previous Turn
        if 0 < turn:
            # H_t => H_t-1 v Q0_t-1 v S0_t-1
            clauses.append([-self.prop2index[(turn, row, col, HEALTHY)],
                            self.prop2index[(turn - 1, row, col, HEALTHY)],
                            self.prop2index[(turn - 1, row, col, QUARANTINED_0)],
                            self.prop2index[(turn - 1, row, col, SICK_0)]])

        # Next Turn
        if turn < self.num_turns - 1:
            # H_t => H_t+1 \/ S2_t+1 \/ I_recent_t+1
            clauses.append([-self.prop2index[(turn, row, col, HEALTHY)],
                            self.prop2index[(turn + 1, row, col, HEALTHY)],
                            self.prop2index[(turn + 1, row, col, SICK_2)],
                            self.prop2index[(turn + 1, row, col, IMMUNE_RECENTLY)]])
        return clauses

    def immune_clauses(self, turn, row, col):
        clauses = []

        # Previous Turn
        if 0 < turn:
            # I_t => I_t-1 v I_recent_t-1
            clauses.append([-self.prop2index[(turn, row, col, IMMUNE)],
                            self.prop2index[(turn - 1, row, col, IMMUNE)],
                            self.prop2index[(turn - 1, row, col, IMMUNE_RECENTLY)]])

            # I_recent_t => H_t-1
            clauses.append([-self.prop2index[(turn, row, col, IMMUNE_RECENTLY)],
                            self.prop2index[(turn - 1, row, col, HEALTHY)]])

        # Next Turn
        if turn < self.num_turns - 1:
            # I_t => I_t+1
            clauses.append([-self.prop2index[(turn, row, col, IMMUNE)],
                            self.prop2index[(turn + 1, row, col, IMMUNE)]])

            # I_recent_t => I_t+1
            clauses.append([-self.prop2index[(turn, row, col, IMMUNE_RECENTLY)],
                            self.prop2index[(turn + 1, row, col, IMMUNE)]])

        return clauses

    def quarantine_clauses(self, turn, row, col):
        clauses = []

        # Previous Turn
        if 0 < turn:
            # Q1_t => S2_t-1 v S1_t-1 v S0_t-1
            clauses.append([-self.prop2index[(turn, row, col, QUARANTINED_1)],
                            self.prop2index[(turn - 1, row, col, SICK_2)],
                            self.prop2index[(turn - 1, row, col, SICK_1)],
                            self.prop2index[(turn - 1, row, col, SICK_0)]])

            # Q0_t => Q1_t-1
            clauses.append([-self.prop2index[(turn, row, col, QUARANTINED_0)],
                            self.prop2index[(turn - 1, row, col, QUARANTINED_1)]])

        # Next Turn
        if turn < self.num_turns - 1:
            # Q1_t => Q0_t+1
            clauses.append([-self.prop2index[(turn, row, col, QUARANTINED_1)],
                            self.prop2index[(turn + 1, row, col, QUARANTINED_0)]])

            # Q0_t => H_t+1
            clauses.append([-self.prop2index[(turn, row, col, QUARANTINED_0)],
                            self.prop2index[(turn + 1, row, col, HEALTHY)]])

        return clauses

    def generate_valid_actions_clauses(self):
        return []

    def get_neighbors(self, i, j):
        return [val for val in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] if self.in_board(*val)]

    def in_board(self, i, j):
        return 0 <= i < self.height and 0 <= j < self.width

    def generate_query_clause(self, query):
        (q_row, q_col), turn, state = query

        if state == SICK:
            clause = [self.prop2index[(turn, q_row, q_col, SICK_0)],
                      self.prop2index[(turn, q_row, q_col, SICK_1)],
                      self.prop2index[(turn, q_row, q_col, SICK_2)]]

        elif state == QUARANTINED:
            clause = [self.prop2index[(turn, q_row, q_col, QUARANTINED_0)],
                      self.prop2index[(turn, q_row, q_col, QUARANTINED_1)]]

        elif state == IMMUNE:
            clause = [self.prop2index[(turn, q_row, q_col, IMMUNE)],
                      self.prop2index[(turn, q_row, q_col, IMMUNE_RECENTLY)]]

        else:
            clause = [self.prop2index[(turn, q_row, q_col, state)]]

        return clause

    def repr_clauses(self, clauses):
        return [self.clause2str(clause) for clause in clauses]

    def clause2str(self, clause):
        out = ''
        for ind in clause[:-1]:
            if ind < 0:
                out += f'-{self.prop2str(self.index2prop[-ind])} v '
            else:
                out += f'{self.prop2str(self.index2prop[ind])} v '
        if clause[-1] < 0:
            out += '-' + self.prop2str(self.index2prop[-clause[-1]])
        else:
            out += self.prop2str(self.index2prop[clause[-1]])
        return out

    def prop2str(self, prop):
        turn, row, col, state = prop
        return f'{state}_{turn}({row}, {col})'


def main():
    inp = {
        "police": 1,
        "medics": 0,
        "observations": [
            (
                ('H', 'S'),
                ('?', 'H')
            ),

            (
                ('S', '?'),
                ('?', 'S')
            ),
        ],

        "queries": [
            [((0, 1), 1, "H"), ((1, 0), 1, "S")]
        ]

    }

    s = Solver(inp)
    clauses = s.generate_clauses()


if __name__ == '__main__':
    main()
