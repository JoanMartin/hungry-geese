import itertools

import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import row_col

from encoders.base_encoder import BaseEncoder
from game_state import GameState


class ThreePlaneEncoder(BaseEncoder):

    def __init__(self, columns: int, rows: int):
        self.columns = columns
        self.rows = rows
        self.num_planes = 3

    def name(self):
        return 'three_plane_encoder'

    def num_points(self):
        return self.columns * self.rows

    def encode(self, game_state: GameState, goose_index: int):
        board_tensor = np.zeros(self.shape())

        bodies = [i.positions for i in game_state.geese]

        my_body = bodies.pop(goose_index)
        if not my_body:
            raise Exception("Can't encode.")

        my_head = my_body[0]

        enemy_bodies = list(itertools.chain.from_iterable(bodies))
        enemy_heads = [positions[0] if len(positions) > 0 else None for positions in bodies]

        foods = [i for i in game_state.food]

        for i in range(self.num_points()):
            row, column = row_col(i, self.columns)

            if i in foods:
                board_tensor[0][row][column] = 1

            if i == my_head:
                board_tensor[1][row][column] = 5
            elif i in my_body:
                board_tensor[1][row][column] = 1

            if i in enemy_heads:
                board_tensor[2][row][column] = 5
            elif i in enemy_bodies:
                board_tensor[2][row][column] = 1

        return board_tensor

    def shape(self):
        return self.num_planes, self.rows, self.columns
