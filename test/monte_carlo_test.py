import collections
import unittest

from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration

from game_state import GameState
from goose import Goose
from monte_carlo import MonteCarlo


class MonteCarloTest(unittest.TestCase):
    configuration = Configuration({"columns": 11,
                                   "rows": 7,
                                   "hunger_rate": 40,
                                   "min_food": 2,
                                   "max_length": 99})

    def test_run(self):
        best_moves = []

        for i in range(20):
            goose_white = Goose(0, [72], Action.NORTH)
            goose_blue = Goose(1, [49, 60], Action.NORTH)
            goose_green = Goose(2, [18, 7, 8], Action.SOUTH)
            goose_red = Goose(3, [11, 22], Action.NORTH)

            game_state = GameState([goose_white, goose_blue, goose_green, goose_red],
                                   [10, 73],
                                   self.configuration,
                                   11)

            monte_carlo = MonteCarlo(1000, 6)

            best_move = monte_carlo.select_best_move(game_state, 0)

            best_moves.append(best_move)

        print(collections.Counter(best_moves))

    def test_run_2(self):
        best_moves = []

        for i in range(20):
            goose_white = Goose(0, [42, 41], Action.EAST)
            goose_blue = Goose(1, [44, 55], Action.NORTH)
            goose_green = Goose(2, [25, 26, 15, 4], Action.WEST)
            goose_red = Goose(3, [71, 60, 61, 50, 51], Action.SOUTH)

            game_state = GameState([goose_white, goose_blue, goose_green, goose_red],
                                   [64, 46],
                                   self.configuration,
                                   11)

            monte_carlo = MonteCarlo(1000, 6)

            best_move = monte_carlo.select_best_move(game_state, 0)

            best_moves.append(best_move)

        print(collections.Counter(best_moves))

    def test_run_3(self):
        best_moves = []

        for i in range(20):
            goose_white = Goose(0, [49, 48], Action.EAST)
            goose_blue = Goose(1, [75, 74, 8, 19, 18, 17, 6, 7, 73, 62, 63, 52, 41, 30, 29, 28], Action.EAST)
            goose_green = Goose(2, [11, 21, 10, 76, 66, 55, 56, 57, 58, 59, 60, 61, 72], Action.EAST)
            goose_red = Goose(3, [], Action.SOUTH)

            game_state = GameState([goose_white, goose_blue, goose_green, goose_red],
                                   [38],
                                   self.configuration,
                                   148)

            monte_carlo = MonteCarlo(1000, 6)

            best_move = monte_carlo.select_best_move(game_state, 0)

            best_moves.append(best_move)

        print(collections.Counter(best_moves))
