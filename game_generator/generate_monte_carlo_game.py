import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration

from encoders.three_plane_encoder import ThreePlaneEncoder
from game_state import GameState
from monte_carlo import MonteCarlo


class GenerateMonteCarloGame:

    def __init__(self, configuration: Configuration, number_geese: int):
        self.configuration = configuration
        self.number_geese = number_geese

    def generate_game(self, num_rounds: int, deep: int):
        boards, actions = [], []

        encoder = ThreePlaneEncoder(self.configuration.columns, self.configuration.rows)
        monte_carlo = MonteCarlo(num_rounds, deep)
        game_state = GameState.new_game(self.number_geese, self.configuration)

        while not game_state.is_over():
            round_actions = []

            for index, goose in enumerate(game_state.geese):
                action = monte_carlo.select_best_action(game_state, index)
                round_actions.append(action)

                if action:
                    # The encoded board situation is appended to boards
                    boards.append(encoder.encode(game_state, index))

                    # The one-hot-encoded action is appended to actions
                    action_one_hot = np.zeros(encoder.num_actions())
                    action_one_hot[encoder.encode_action(action)] = 1
                    actions.append(action_one_hot)

            print(f"{game_state.steps} - {round_actions}")
            game_state = game_state.apply_move(round_actions)

        return np.array(boards), np.array(actions)
