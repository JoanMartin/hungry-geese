import unittest

import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration

from encoders.three_plane_encoder import ThreePlaneEncoder
from game_state import GameState
from goose import Goose


class ThreePlaneEncoderTest(unittest.TestCase):
    configuration = Configuration({"columns": 11,
                                   "rows": 7,
                                   "hunger_rate": 40,
                                   "min_food": 2,
                                   "max_length": 99})

    def test_run(self):
        goose_white = Goose(0, [72], Action.NORTH)
        goose_blue = Goose(1, [49, 60], Action.NORTH)
        goose_green = Goose(2, [18, 7, 8], Action.SOUTH)
        goose_red = Goose(3, [11, 22], Action.NORTH)

        game_state = GameState([goose_white, goose_blue, goose_green, goose_red],
                               [10, 73],
                               self.configuration,
                               11)

        three_plane_encoder = ThreePlaneEncoder(self.configuration.columns, self.configuration.rows)
        board_tensor = three_plane_encoder.encode(game_state, 0)

        expected_board_tensor = np.array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]],
                                          [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 5., 0., 0., 0., 0.]],
                                          [[0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
                                           [5., 0., 0., 0., 0., 0., 0., 5., 0., 0., 0.],
                                           [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 5., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]], dtype=float)

        assert np.array_equal(board_tensor, expected_board_tensor)

    def test_run_2(self):
        goose_white = Goose(0, [49, 48], Action.EAST)
        goose_blue = Goose(1, [75, 74, 8, 19, 18, 17, 6, 7, 73, 62, 63, 52, 41, 30, 29, 28], Action.EAST)
        goose_green = Goose(2, [11, 21, 10, 76, 66, 55, 56, 57, 58, 59, 60, 61, 72], Action.EAST)
        goose_red = Goose(3, [], Action.SOUTH)

        game_state = GameState([goose_white, goose_blue, goose_green, goose_red],
                               [38],
                               self.configuration,
                               148)

        three_plane_encoder = ThreePlaneEncoder(self.configuration.columns, self.configuration.rows)
        board_tensor = three_plane_encoder.encode(game_state, 0)

        expected_board_tensor = np.array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
                                          [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 1., 5., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
                                          [[0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1.],
                                           [5., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1.],
                                           [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                           [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                           [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                                           [1., 0., 0., 0., 0., 0., 1., 1., 1., 5., 1.]]], dtype=float)

        assert np.array_equal(board_tensor, expected_board_tensor)
