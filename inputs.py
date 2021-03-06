# disabled illegal problems
problems = [
    {
        "police": 0,
        "medics": 0,
        "observations": [
            (
                ('H', '?'),
                ('H', 'H')
            ),

            (
                ('S', '?'),
                ('?', 'S')
            ),
        ],

        "queries": [
            ((0, 1), 0, "H"), ((1, 0), 1, "S")
        ]
    },

    {
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
            ((0, 1), 1, "H"), ((1, 0), 1, "S")
        ]
    },

    {
        "police": 0,
        "medics": 0,
        "observations": [
            (
                ('H', '?', 'H'),
                ('H', 'H', 'H'),
                ('H', 'H', 'S'),
            ),

            (
                ('H', 'H', 'H'),
                ('?', 'H', 'S'),
                ('H', 'S', 'S'),
            ),

            (
                ('H', 'H', 'S'),
                ('H', '?', 'S'),
                ('S', 'S', 'S'),
            ),

            (
                ('?', 'S', 'S'),
                ('S', 'S', 'S'),
                ('S', 'S', 'H'),
            ),
        ],

        "queries": [
            ((0, 1), 0, 'H'), ((1, 0), 1, 'S'), ((1, 1), 2, 'H'), ((0, 0), 3, 'S')
        ],
    },

    # {
    #     "police": 1,
    #     "medics": 0,
    #     "observations": [
    #         (
    #             ('S', 'H', 'U'),
    #             ('H', 'H', 'H'),
    #             ('U', 'H', 'S'),
    #         ),
    #
    #         (
    #             ('S', 'S', 'U'),
    #             ('S', 'H', 'S'),
    #             ('U', 'S', 'Q'),
    #         ),
    #
    #         (
    #             ('?', 'S', 'U'),
    #             ('S', 'S', 'S'),
    #             ('U', 'S', 'Q'),
    #         ),
    #     ],
    #
    #     "queries": [
    #         ((0, 0), 2, 'H')
    #     ],
    #
    # },

    {
        "police": 0,
        "medics": 0,
        "observations": [
            (
                ('H', 'H', 'H', 'H'),
                ('H', 'S', 'U', 'H'),
                ('H', 'H', 'H', '?'),
                ('H', 'H', 'S', 'S'),
            ),

            (
                ('H', 'S', 'H', 'H'),
                ('S', 'S', 'U', 'H'),
                ('H', 'S', 'S', 'U'),
                ('?', 'S', 'S', 'S'),
            ),
        ],

        "queries": [
            ((2, 3), 0, 'H'), ((3, 0), 1, 'H')
        ],
    },

    # {
    #     "police": 0,
    #     "medics": 1,
    #     "observations": [
    #         (
    #             ('H', 'H', 'H', 'H'),
    #             ('H', 'S', 'U', 'H'),
    #             ('U', 'S', 'S', 'U'),
    #             ('H', 'S', 'H', 'H'),
    #         ),
    #
    #         (
    #             ('H', 'S', 'H', 'I'),
    #             ('S', 'S', 'U', 'H'),
    #             ('U', '?', 'S', 'U'),
    #             ('S', 'S', 'S', 'H'),
    #         ),
    #
    #         (
    #             ('H', 'S', 'I', 'I'),
    #             ('S', 'S', 'U', '?'),
    #             ('U', 'S', 'S', 'U'),
    #             ('?', 'S', 'S', 'S')
    #         ),
    #
    #     ],
    #
    #     "queries": [
    #         ((2, 1), 1, 'U'), ((1, 3), 2, 'I'), ((3, 0), 2, 'S')
    #     ],
    # },

    {
        "police": 0,
        "medics": 0,
        "observations": [
            (
                ('H', 'S'),
                ('H', 'H'),
            ),

            (
                ('S', 'S'),
                ('H', 'S'),
            ),

            (
                ('S', 'S'),
                ('S', 'S'),
            ),

            (
                ('S', 'H'),
                ('S', 'S'),
            ),

            (
                ('?', '?'),
                ('?', '?'),
            ),

        ],

        "queries": [
            ((0, 0), 4, 'H'), ((1, 0), 4, 'S')
        ],
    },
    ########################################
    {
        "police": 0,
        "medics": 0,
        "observations": [

            (
                ('?', 'H'),
                ('H', 'H'),
            ),

            (
                ('H', 'H'),
                ('H', 'H'),
            ),

        ],
        "queries": [((0, 0), 0, 'S'), ],
    },

    {
        "police": 0,
        "medics": 0,
        "observations": [
            (
                ('S', '?', 'S'),
                ('H', 'S', 'H'),
                ('H', 'H', 'H'),
            ),

            (
                ('S', 'S', 'S'),
                ('S', 'S', 'S'),
                ('H', 'S', 'H'),
            ),
        ],
        "queries": [((0, 1), 0, 'S'), ],
    },

    {
        "police": 1,
        "medics": 0,
        "observations": [
            (
                ('H', 'H', 'S'),
                ('S', 'H', 'S'),
                ('H', 'H', 'H'),
            ),

            (
                ('S', 'S', 'S'),
                ('S', 'S', '?'),
                ('S', 'H', 'H'),
            ),

            (
                ('S', 'Q', 'S'),
                ('S', 'S', '?'),
                ('?', 'S', 'H'),
            ),
        ],
        "queries": [((1, 2), 1, 'S'), ((2, 0), 2, 'S')],
    },

    {
        "police": 0,
        "medics": 1,
        "observations": [

            (
                ('?', '?'),
                ('?', '?'),
            ),

            (
                ('H', 'H'),
                ('I', 'H'),
            ),

        ],
        "queries": [((0, 0), 0, 'H'), ((0, 1), 0, 'H'), ((1, 0), 0, 'H'), ((1, 1), 0, 'H')],
    },

    {
        "police": 0,
        "medics": 1,
        "observations": [
            (
                ('S', 'H', 'S'),
                ('H', 'H', 'H'),
                ('?', 'H', 'S'),
            ),

            (
                ('S', 'S', 'S'),
                ('S', 'I', 'S'),
                ('?', 'S', 'S'),
            ),
        ],
        "queries": [((2, 0), 1, 'S'), ],
    },

    # {
    #     "police": 1,
    #     "medics": 1,
    #     "observations": [
    #         (
    #             ('S', 'S', 'H', 'H'),
    #             ('H', 'H', 'H', '?'),
    #             ('H', 'S', 'H', 'H'),
    #         ),
    #
    #         (
    #             ('S', 'S', 'S', 'H'),
    #             ('S', 'S', 'H', '?'),
    #             ('S', 'Q', 'H', 'H'),
    #         ),
    #     ],
    #     "queries": [((1, 3), 0, 'H'), ((1, 3), 1, 'H')],
    # },

    {
        "police": 4,
        "medics": 4,
        "observations": [
            (
                ('H', 'H', 'H', 'H'),
                ('H', 'H', 'H', 'H'),
                ('H', 'H', 'H', 'H'),
            ),

            (
                ('H', 'I', '?', 'H'),
                ('I', 'H', 'H', '?'),
                ('H', 'I', 'H', 'H'),
            ),
        ],
        "queries": [((0, 2), 1, 'H'), ((1, 3), 1, 'H')],
    },

    # {
    #     "police": 2,
    #     "medics": 2,
    #     "observations": [
    #         (
    #             ('H', 'H', 'H', 'H'),
    #             ('H', 'S', 'U', 'H'),
    #             ('U', 'S', 'S', 'U'),
    #             ('H', 'S', 'H', 'H'),
    #         ),
    #
    #         (
    #             ('H', '?', 'H', 'H'),
    #             ('H', 'S', 'U', 'I'),
    #             ('U', 'Q', 'S', 'U'),
    #             ('H', 'Q', 'S', 'I'),
    #         ),
    #
    #         (
    #             ('I', '?', '?', 'H'),
    #             ('Q', 'S', 'U', 'I'),
    #             ('U', 'Q', 'S', 'U'),
    #             ('I', 'Q', 'S', 'I'),
    #         ),
    #
    #     ],
    #
    #     "queries": [
    #         ((0, 1), 1, 'H'), ((0, 1), 2, 'S'), ((0, 2), 2, 'H')
    #     ],
    # },

]

from check import solve_problems

solve_problems(problems)
