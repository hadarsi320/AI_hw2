import itertools
from copy import deepcopy
from time import time

from pysat.card import CardEnc
from pysat.formula import IDPool
from pysat.solvers import Glucose4

ids = ['318792827', '321659187']

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
FIRST_TURN_STATES = [SICK_2, HEALTHY, UNPOPULATED]
SECOND_TURN_STATES = [SICK_1, SICK_2, HEALTHY, QUARANTINED_1, IMMUNE_RECENTLY, UNPOPULATED]
SICK_STATES = [SICK_0, SICK_1, SICK_2]
QUERY_STATES = [SICK, HEALTHY, QUARANTINED, IMMUNE, UNPOPULATED]


class Solver:
    def __init__(self, solver_input):
        self.num_police = solver_input['police']
        self.num_medics = solver_input['medics']
        self.observations = solver_input['observations']
        self.num_turns = len(self.observations)
        self.height = len(self.observations[0])
        self.width = len(self.observations[0][0])
        self.vpool = IDPool()
        self.tiles = [(i, j) for i in range(self.height) for j in range(self.width)]

        self.clauses = self.generate_clauses()

    def generate_clauses(self):
        clauses = []
        clauses.extend(self.generate_observations_clauses())
        clauses.extend(self.generate_validity_clauses())
        clauses.extend(self.generate_dynamics_clauses())
        clauses.extend(self.generate_valid_actions_clauses())
        return clauses

    def generate_observations_clauses(self):
        clauses = []

        for turn, observation in enumerate(self.observations):
            for row in range(self.height):
                for col in range(self.width):
                    state = observation[row][col]
                    if state == SICK:
                        clauses.append([self.vpool.id((turn, row, col, SICK_0)),
                                        self.vpool.id((turn, row, col, SICK_1)),
                                        self.vpool.id((turn, row, col, SICK_2))])
                    elif state == QUARANTINED:
                        clauses.append([self.vpool.id((turn, row, col, QUARANTINED_0)),
                                        self.vpool.id((turn, row, col, QUARANTINED_1))])
                    elif state == IMMUNE:
                        clauses.append([self.vpool.id((turn, row, col, IMMUNE_RECENTLY)),
                                        self.vpool.id((turn, row, col, IMMUNE))])
                    elif state == UNK:
                        continue
                    else:
                        clauses.append([self.vpool.id((turn, row, col, state))])

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
        lits = [self.vpool.id((0, row, col, state)) for state in FIRST_TURN_STATES]
        clauses = CardEnc.equals(lits, bound=1, vpool=self.vpool).clauses
        for state in STATES:
            if state not in FIRST_TURN_STATES:
                clauses.append([-self.vpool.id((0, row, col, state))])
        return clauses

    def second_turn_clauses(self, row, col):
        lits = [self.vpool.id((1, row, col, state)) for state in SECOND_TURN_STATES]
        clauses = CardEnc.equals(lits, bound=1, vpool=self.vpool).clauses
        for state in STATES:
            if state not in SECOND_TURN_STATES:
                clauses.append([-self.vpool.id((0, row, col, state))])
        return clauses

    def uniqueness_clauses(self, row, col):
        clauses = []
        for turn in range(self.num_turns):
            lits = [self.vpool.id((turn, row, col, state)) for state in STATES]
            clauses.extend(CardEnc.equals(lits, bound=1, vpool=self.vpool).clauses)
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
            clauses.append([-self.vpool.id((turn, row, col, UNPOPULATED)),
                            self.vpool.id((turn - 1, row, col, UNPOPULATED))])

        # Next Turn
        if turn < self.num_turns - 1:
            # U_t => U_t+1
            clauses.append([-self.vpool.id((turn, row, col, UNPOPULATED)),
                            self.vpool.id((turn + 1, row, col, UNPOPULATED))])

        return clauses

    def sick_clauses(self, turn, row, col):
        clauses = []
        neighbors = self.get_neighbors(row, col)

        # Is sick
        # Previous Turn
        if 0 < turn:
            # S2_t => H_t-1
            clauses.append([-self.vpool.id((turn, row, col, SICK_2)),
                            self.vpool.id((turn - 1, row, col, HEALTHY))])
            # S1_t => S2_t-1
            clauses.append([-self.vpool.id((turn, row, col, SICK_1)),
                            self.vpool.id((turn - 1, row, col, SICK_2))])
            # S0_t => S1_t-1
            clauses.append([-self.vpool.id((turn, row, col, SICK_0)),
                            self.vpool.id((turn - 1, row, col, SICK_1))])

        # Next Turn
        if turn < self.num_turns - 1:
            # S2_t => S1_t+1 v Q1_t+1
            clauses.append([-self.vpool.id((turn, row, col, SICK_2)),
                            self.vpool.id((turn + 1, row, col, SICK_1)),
                            self.vpool.id((turn + 1, row, col, QUARANTINED_1))])
            # S1_t => S0_t+1 v Q1_t+1
            clauses.append([-self.vpool.id((turn, row, col, SICK_1)),
                            self.vpool.id((turn + 1, row, col, SICK_0)),
                            self.vpool.id((turn + 1, row, col, QUARANTINED_1))])
            # S0_t => H_t+1 v Q1_t+1
            clauses.append([-self.vpool.id((turn, row, col, SICK_0)),
                            self.vpool.id((turn + 1, row, col, HEALTHY)),
                            self.vpool.id((turn + 1, row, col, QUARANTINED_1))])

        # Infected By Someone
        if 0 < turn:
            # S2_t => V (S2n_t-1 v S1n_t-1 v S0n_t-1) for n in neighbors
            clause = [-self.vpool.id((turn, row, col, SICK_2))]
            for (n_row, n_col) in neighbors:
                clause.extend([self.vpool.id((turn - 1, n_row, n_col, SICK_2)),
                               self.vpool.id((turn - 1, n_row, n_col, SICK_1)),
                               self.vpool.id((turn - 1, n_row, n_col, SICK_0))])
            clauses.append(clause)

        # Infecting Others
        if turn < self.num_turns - 1:
            for (n_row, n_col) in neighbors:
                for sick_i in [SICK_0, SICK_1, SICK_2]:
                    # Si_t /\ Hn_t /\ -Q1_t+1 /\ -I_recent_t+1 => S2n_t+1 (Sn, Hn stand for neighbor)
                    clauses.append([-self.vpool.id((turn, row, col, sick_i)),
                                    -self.vpool.id((turn, n_row, n_col, HEALTHY)),
                                    self.vpool.id((turn + 1, row, col, QUARANTINED_1)),
                                    self.vpool.id((turn + 1, n_row, n_col, IMMUNE_RECENTLY)),
                                    self.vpool.id((turn + 1, n_row, n_col, SICK_2))])

        return clauses

    def healthy_clauses(self, turn, row, col):
        clauses = []

        # Previous Turn
        if 0 < turn:
            # H_t => H_t-1 v Q0_t-1 v S0_t-1
            clauses.append([-self.vpool.id((turn, row, col, HEALTHY)),
                            self.vpool.id((turn - 1, row, col, HEALTHY)),
                            self.vpool.id((turn - 1, row, col, QUARANTINED_0)),
                            self.vpool.id((turn - 1, row, col, SICK_0))])

        # Next Turn
        if turn < self.num_turns - 1:
            # H_t => H_t+1 \/ S2_t+1 \/ I_recent_t+1
            clauses.append([-self.vpool.id((turn, row, col, HEALTHY)),
                            self.vpool.id((turn + 1, row, col, HEALTHY)),
                            self.vpool.id((turn + 1, row, col, SICK_2)),
                            self.vpool.id((turn + 1, row, col, IMMUNE_RECENTLY))])
        return clauses

    def immune_clauses(self, turn, row, col):
        clauses = []

        # Previous Turn
        if 0 < turn:
            # I_t => I_t-1 v I_recent_t-1
            clauses.append([-self.vpool.id((turn, row, col, IMMUNE)),
                            self.vpool.id((turn - 1, row, col, IMMUNE)),
                            self.vpool.id((turn - 1, row, col, IMMUNE_RECENTLY))])

            # I_recent_t => H_t-1
            clauses.append([-self.vpool.id((turn, row, col, IMMUNE_RECENTLY)),
                            self.vpool.id((turn - 1, row, col, HEALTHY))])

        # Next Turn
        if turn < self.num_turns - 1:
            # I_t => I_t+1
            clauses.append([-self.vpool.id((turn, row, col, IMMUNE)),
                            self.vpool.id((turn + 1, row, col, IMMUNE))])

            # I_recent_t => I_t+1
            clauses.append([-self.vpool.id((turn, row, col, IMMUNE_RECENTLY)),
                            self.vpool.id((turn + 1, row, col, IMMUNE))])

        return clauses

    def quarantine_clauses(self, turn, row, col):
        clauses = []

        # Previous Turn
        if 0 < turn:
            # Q1_t => S2_t-1 v S1_t-1 v S0_t-1
            clauses.append([-self.vpool.id((turn, row, col, QUARANTINED_1)),
                            self.vpool.id((turn - 1, row, col, SICK_2)),
                            self.vpool.id((turn - 1, row, col, SICK_1)),
                            self.vpool.id((turn - 1, row, col, SICK_0))])

            # Q0_t => Q1_t-1
            clauses.append([-self.vpool.id((turn, row, col, QUARANTINED_0)),
                            self.vpool.id((turn - 1, row, col, QUARANTINED_1))])

        # Next Turn
        if turn < self.num_turns - 1:
            # Q1_t => Q0_t+1
            clauses.append([-self.vpool.id((turn, row, col, QUARANTINED_1)),
                            self.vpool.id((turn + 1, row, col, QUARANTINED_0))])

            # Q0_t => H_t+1
            clauses.append([-self.vpool.id((turn, row, col, QUARANTINED_0)),
                            self.vpool.id((turn + 1, row, col, HEALTHY))])

        return clauses

    def generate_valid_actions_clauses(self):
        clauses = []
        clauses.extend(self.generate_police_clauses())
        clauses.extend(self.generate_medic_clauses())
        return clauses

    def generate_police_clauses(self):
        clauses = []

        for turn in range(1, self.num_turns):
            lits = [self.vpool.id((turn, row, col, QUARANTINED_1))
                    for row in range(self.height)
                    for col in range(self.width)]
            clauses.extend(CardEnc.atmost(lits, bound=self.num_police, vpool=self.vpool).clauses)

        if self.num_police == 0:
            return clauses

        for turn in range(self.num_turns - 1):
            for num_sick in range(self.width * self.height):
                for sick_tiles in itertools.combinations(self.tiles, num_sick):
                    healthy_tiles = [tile for tile in self.tiles if tile not in sick_tiles]
                    for sick_states in itertools.combinations_with_replacement(self.possible_sick_states(turn),
                                                                               num_sick):
                        clause = []

                        for (row, col), state in zip(sick_tiles, sick_states):
                            clause.append(-self.vpool.id((turn, row, col, state)))
                        for row, col in healthy_tiles:
                            for state in self.possible_sick_states(turn):
                                clause.append(self.vpool.id((turn, row, col, state)))

                        lits = [self.vpool.id((turn + 1, row, col, QUARANTINED_1)) for row, col in sick_tiles]
                        equals_clauses = CardEnc.equals(
                            lits, bound=min(self.num_police, num_sick), vpool=self.vpool).clauses
                        for sub_clause in equals_clauses:
                            temp_clause = deepcopy(clause)
                            temp_clause += sub_clause
                            clauses.append(temp_clause)

        return clauses

    def generate_medic_clauses(self):
        clauses = []

        for turn in range(self.num_turns):
            lits = [self.vpool.id((turn, row, col, IMMUNE_RECENTLY))
                    for row in range(self.height)
                    for col in range(self.width)]
            clauses.extend(CardEnc.atmost(lits, bound=self.num_medics, vpool=self.vpool).clauses)
        if self.num_medics == 0:
            return clauses

        for turn in range(self.num_turns - 1):
            for num_healthy in range(self.width * self.height):
                for healthy_tiles in itertools.combinations(self.tiles, num_healthy):
                    sick_tiles = [tile for tile in self.tiles if tile not in healthy_tiles]
                    clause = []

                    for row, col in healthy_tiles:
                        clause.append(-self.vpool.id((turn, row, col, HEALTHY)))

                    for row, col in sick_tiles:
                        clause.append(self.vpool.id((turn, row, col, HEALTHY)))

                    lits = [self.vpool.id((turn + 1, row, col, IMMUNE_RECENTLY)) for row, col in healthy_tiles]
                    equals_clauses = CardEnc.equals(
                        lits, bound=min(self.num_medics, num_healthy), vpool=self.vpool).clauses
                    for sub_clause in equals_clauses:
                        temp_clause = deepcopy(clause)
                        temp_clause += sub_clause
                        clauses.append(temp_clause)

        return clauses

    def get_neighbors(self, i, j):
        return [val for val in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] if self.in_board(*val)]

    def in_board(self, i, j):
        return 0 <= i < self.height and 0 <= j < self.width

    def generate_query_clause(self, query):
        (q_row, q_col), turn, state = query

        if state == SICK:
            clause = [self.vpool.id((turn, q_row, q_col, SICK_0)),
                      self.vpool.id((turn, q_row, q_col, SICK_1)),
                      self.vpool.id((turn, q_row, q_col, SICK_2))]

        elif state == QUARANTINED:
            clause = [self.vpool.id((turn, q_row, q_col, QUARANTINED_0)),
                      self.vpool.id((turn, q_row, q_col, QUARANTINED_1))]

        elif state == IMMUNE:
            clause = [self.vpool.id((turn, q_row, q_col, IMMUNE)),
                      self.vpool.id((turn, q_row, q_col, IMMUNE_RECENTLY))]

        else:
            clause = [self.vpool.id((turn, q_row, q_col, state))]

        return clause

    def __str__(self):
        return '\n'.join(self.repr_clauses())

    def repr_clauses(self):
        return [self.clause2str(clause) for clause in self.clauses]

    def clause2str(self, clause):
        # out = ''
        # for ind in clause[:-1]:
        #     out += f'{self.prop2str(self.vpool.obj(abs(ind)))} v '
        # out += self.prop2str(self.vpool.obj(abs(clause[-1])))

        out = ' \\/ '.join(['-' * (ind < 0) + self.prop2str(self.vpool.obj(abs(ind)))
                            for ind in clause])
        return out

    @staticmethod
    def prop2str(prop):
        if prop is None:
            return 'Fictive'
        turn, row, col, state = prop
        return f'{state}_{turn}_({row},{col})'

    @staticmethod
    def possible_sick_states(turn):
        if turn == 0:
            return [SICK_2]
        if turn == 1:
            return [SICK_1, SICK_2]
        return SICK_STATES

    def get_state_tiles(self, state, turn):
        tiles = []
        for i, row in enumerate(self.observations[turn]):
            for j, tile in enumerate(row):
                if tile == state:
                    tiles.append((i, j))
        return tiles


def solve_problem(problem):
    solver = Solver(problem)
    results = {}
    for query in problem['queries']:
        results[query] = answer_query(solver, query)
    return results


def answer_query(solver: Solver, query):
    formula = solver.clauses + [solver.generate_query_clause(query)]
    if solve_formula(formula):
        alternative_queries = get_alternative_queries(query)
        for alternative_query in alternative_queries:
            formula = solver.clauses + [solver.generate_query_clause(alternative_query)]
            if solve_formula(formula):
                return '?'
        return 'T'
    return 'F'


def solve_formula(formula):
    g = Glucose4(with_proof=True)
    g.append_formula(formula)
    return g.solve()


def get_alternative_queries(query):
    loc, turn, q_state = query

    cf_queries = []
    for state in QUERY_STATES:
        if state == q_state:
            continue
        cf_queries.append((loc, turn, state))
    return cf_queries


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
    print(solve_problem(inp))


if __name__ == '__main__':
    start_time = time()
    main()
    print(f'Total run time: {time() - start_time}')
