import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, translate

from encoders.base_encoder import BaseEncoder
from game_state import GameState


class SeventeenPlaneEncoder(BaseEncoder):

    def __init__(self, columns: int, rows: int):
        self.columns = columns
        self.rows = rows
        self.num_planes = 17
        self.actions = [action for action in Action]

    def name(self):
        return 'seventeen_plane_encoder'

    def encode(self, game_state: GameState, goose_index: int):
        board_tensor = np.zeros((self.num_planes, self.num_points()), dtype=np.float32)

        for goose in game_state.geese:
            # Head position
            for pos in goose.positions[:1]:
                # Current head position
                board_tensor[0 + (goose.index - goose_index) % 4, pos] = 1

                # Previous head position
                if goose.last_action:
                    prev_head = translate(pos, goose.last_action.opposite(), self.columns, self.rows)
                    board_tensor[12 + (goose.index - goose_index) % 4, prev_head] = 1

            # Tip position
            for pos in goose.positions[-1:]:
                board_tensor[4 + (goose.index - goose_index) % 4, pos] = 1

            # Whole position
            for pos in goose.positions:
                board_tensor[8 + (goose.index - goose_index) % 4, pos] = 1

        # Food position
        for pos in game_state.food:
            board_tensor[16, pos] = 1

        board_tensor = board_tensor.reshape((-1, self.rows, self.columns))

        return board_tensor

    def encode_action(self, action: Action):
        return self.actions.index(action)

    def decode_action_index(self, index: int):
        return self.actions[index]

    def num_points(self):
        return self.columns * self.rows

    def num_actions(self):
        return len(self.actions)

    def shape(self):
        return self.num_planes, self.rows, self.columns
