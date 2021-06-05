import unittest

from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration

from game_generator.generate_monte_carlo_game import GenerateMonteCarloGame


class GenerateMonteCarloGameTest(unittest.TestCase):
    configuration = Configuration({"columns": 11,
                                   "rows": 7,
                                   "hunger_rate": 40,
                                   "min_food": 2,
                                   "max_length": 99})

    def test_run(self):
        generate_game = GenerateMonteCarloGame(self.configuration, 4)
        boards, moves = generate_game.generate_game(250, 5)

        # print(boards)
        # print(moves)
