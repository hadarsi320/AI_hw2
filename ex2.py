from pysat.card import CardEnc

ids = ['318792827', '111111111']

unpopulated = 'U'

sick = 'S'
sick_0 = 'S0'
sick_1 = 'S1'
sick_2 = 'S2'

healthy = 'H'

quarantined = 'Q'
quarantined_0 = 'Q_0'
quarantined_1 = 'Q_1'

immune_recently = 'I_X'
immune = 'I'


unknown = '?'

sat_states = [sick_0, sick_1, sick_2, healthy, quarantined_0, quarantined_1, immune_recently, immune, unpopulated]
query_states = [sick, healthy, immune, quarantined, unpopulated]





def solve_problem(input):
    solver = Solver(input)
    # put your solution here, remember the format needed


class Solver:
    def __init__(self, input):
        self.num_police = input['police']
        self.num_medics = input['medics']
        self.observations = input['observations']
        self.num_turns = len(self.observations)  # TODO maybe max on queries
        self.height = len(self.observations[0])
        self.width = len(self.observations [0][0])
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

    def actions_precondition_clauses(self):
        clauses = []
        for row in range(self.height):
            for col in range(self.width):
                clauses.extend(self.generate_clauses(row, col))

    def unpopulated_clauses(self, row, col):
        clauses = []

        for turn in range(self.num_turns):
            # Previous Turn
            if 0 < turn:
                # U_t = > U_t-1
                clauses.append([-self.prop2index[(turn, row, col, unpopulated)],
                               self.prop2index[(turn - 1, row, col, unpopulated)]])

            # Next Turn
            if turn < self.num_turns - 1:
                # U_t => U_t+1
                clauses.append([-self.prop2index[(turn, row, col, unpopulated)],
                               self.prop2index[(turn + 1, row, col, unpopulated)]])

        return clauses

    def sick_clauses(self, row, col):
        clauses = []
        neighbors = self.get_neighbors(row, col)

        for turn in range(1, self.num_turns - 1):
            # Previous Turn
            if 0 < turn:
                # S2_t => H_t-1
                clauses.append([-self.prop2index[(turn, row, col, sick_2)],
                                self.prop2index[(turn - 1, row, col, healthy)]])
                # S1_t => S2_t-1
                clauses.append([-self.prop2index[(turn, row, col, sick_1)],
                                self.prop2index[(turn - 1, row, col, sick_2)]])
                # S0_t => S1_t-1
                clauses.append([-self.prop2index[(turn, row, col, sick_0)],
                                self.prop2index[(turn - 1, row, col, sick_1)]])

            # Next Turn
            if turn < self.num_turns - 1:
                # S2_t => S1_t+1 v Q1_t+1
                clauses.append([-self.prop2index[(turn, row, col, sick_2)],
                                self.prop2index[(turn + 1, row, col, sick_1)],
                                self.prop2index[(turn + 1, row, col, quarantined_1)]])
                # S1_t => S0_t+1 v Q1_t+1
                clauses.append([-self.prop2index[(turn, row, col, sick_1)],
                                self.prop2index[(turn + 1, row, col, sick_0)],
                                self.prop2index[(turn + 1, row, col, quarantined_1)]])
                # S0_t => H_t+1 v Q1_t+1
                clauses.append([-self.prop2index[(turn, row, col, sick_0)],
                                self.prop2index[(turn + 1, row, col, healthy)],
                                self.prop2index[(turn + 1, row, col, quarantined_1)]])

            # Infected By Someone
            if 0 < turn:
                # S2_t => V (S2n_t-1 v S2n_t-1 v S2n_t-1) for n in neighbors
                clause = [-self.prop2index[(turn, row, col, sick_2)]]
                for (n_row, n_col) in neighbors:
                    clause.extend([self.prop2index[(turn - 1, n_row, n_col, sick_2)],
                                   self.prop2index[(turn - 1, n_row, n_col, sick_1)],
                                   self.prop2index[(turn - 1, n_row, n_col, sick_0)]])
                clauses.append(clause)

            # Infecting Others
            if turn < self.num_turns - 1:
                for (n_row, n_col) in neighbors:
                    # S2_t, Hn_t, -Q1_t+1, -Irecentn_t+1 => S2n_t+1 (Sn, Hn stand for neighbor)
                    clauses.append([-self.prop2index[(turn, row, col, sick_2)],
                                    -self.prop2index[(turn, n_row, n_col, healthy)],
                                    self.prop2index[(turn + 1, row, col, quarantined_1)],
                                    self.prop2index[(turn + 1, n_row, n_col, immune_recently)],
                                    self.prop2index[(turn + 1, n_row, n_col, sick_2)]])

                    # S1_t, Hn_t, -Q1_t+1, -Irecentn_t+1 => S2n_t+1
                    clauses.append([-self.prop2index[(turn, row, col, sick_1)],
                                    -self.prop2index[(turn, n_row, n_col, healthy)],
                                    self.prop2index[(turn + 1, row, col, quarantined_1)],
                                    self.prop2index[(turn + 1, n_row, n_col, immune_recently)],
                                    self.prop2index[(turn + 1, n_row, n_col, sick_2)]])

                    # S0_t, Hn_t, -Q1_t+1, -Irecentn_t+1 => S2n_t+1
                    clauses.append([-self.prop2index[(turn, row, col, sick_0)],
                                    -self.prop2index[(turn, n_row, n_col, healthy)],
                                    self.prop2index[(turn + 1, row, col, quarantined_1)],
                                    self.prop2index[(turn + 1, n_row, n_col, immune_recently)],
                                    self.prop2index[(turn + 1, n_row, n_col, sick_2)]])

        return clauses

    def healthy_clauses(self, row, col):
        clauses = []

        for turn in range(self.num_turns):
            # Previous Turn
            if 0 < turn:
                # H_t => H_t-1 v Q0_t-1 v S0_t-1
                clauses.append([-self.prop2index[(turn, row, col, healthy)],
                                self.prop2index[(turn - 1, row, col, healthy)],
                                self.prop2index[(turn - 1, row, col, quarantined_0)],
                                self.prop2index[(turn - 1, row, col, sick_0)]])

            # Next Turn
            if turn < self.num_turns - 1:
                # H_t => H_t+1 v S2_t+1 v Irecently_t+1
                clauses.append([-self.prop2index[(turn, row, col, healthy)],
                                self.prop2index[(turn + 1, row, col, healthy)],
                                self.prop2index[(turn + 1, row, col, sick_2)],
                                self.prop2index[(turn + 1, row, col, immune_recently)]])
        return clauses

    def immune_clauses(self, row, col):
        clauses = []

        for turn in range(self.num_turns):
            # Previous Turn
            if 0 < turn:
                # I_t => I_t-1 v Irecent_t-1
                clauses.append([-self.prop2index[(turn, row, col, immune)],
                                self.prop2index[(turn - 1, row, col, immune)],
                                self.prop2index[(turn - 1, row, col, immune_recently)]])

                # Irecent_t => H_t-1
                clauses.append([-self.prop2index[(turn, row, col, immune_recently)],
                                self.prop2index[(turn - 1, row, col, healthy)]])

            # Next Turn
            if turn < self.num_turns - 1:
                # I_t => I_t+1
                clauses.append([-self.prop2index[(turn, row, col, immune)],
                                self.prop2index[(turn + 1, row, col, immune)]])

                # Irecent_t => I_t+1
                clauses.append([-self.prop2index[(turn, row, col, immune_recently)],
                                self.prop2index[(turn + 1, row, col, immune)]])

        return clauses

    def quarantine_clauses(self, row, col):
        clauses = []

        for turn in range(self.num_turns):
            # Previous Turn
            if 0 < turn:
                # Q1_t => S2_t-1 v S1_t-1 v S0_t-1
                clauses.append([-self.prop2index[(turn, row, col, quarantined_1)],
                                self.prop2index[(turn - 1, row, col, sick_2)],
                                self.prop2index[(turn - 1, row, col, sick_1)],
                                self.prop2index[(turn - 1, row, col, sick_0)]])

                # Q0_t => Q1_t-1
                clauses.append([-self.prop2index[(turn, row, col, quarantined_0)],
                                self.prop2index[(turn - 1, row, col, quarantined_1)]])

            # Next Turn
            if turn < self.num_turns - 1:
                # Q1_t => Q0_t+1
                clauses.append([-self.prop2index[(turn, row, col, quarantined_1)],
                                self.prop2index[(turn + 1, row, col, quarantined_0)]])

                # Q0_t => H_t+1
                clauses.append([-self.prop2index[(turn, row, col, quarantined_0)],
                                self.prop2index[(turn + 1, row, col, healthy)]])

        return clauses

    def generate_observations_clauses(self):


    def generate_clauses(self, turns, row, col, height, width, num_police, num_medics):
        neighbors = self.get_neighbors(row, col)

        clauses = []
        clauses.extend(self.sick_clauses(row, col))

        return clauses

    def get_neighbors(self, i, j):
        return [val for val in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] if self.in_board(*val)]

    def in_board(self, i, j):
        return 0 <= i < self.height and 0 <= j < self.width

