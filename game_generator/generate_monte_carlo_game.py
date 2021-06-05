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
        boards, moves = [], []

        encoder = ThreePlaneEncoder(self.configuration.columns, self.configuration.rows)
        monte_carlo = MonteCarlo(num_rounds, deep)
        game_state = GameState.new_game(self.number_geese, self.configuration)

        while not game_state.is_over():
            round_actions = []

            for index, goose in enumerate(game_state.geese):
                move = monte_carlo.select_best_move(game_state, index)
                round_actions.append(move)

                if move:
                    boards.append(encoder.encode(game_state, index))
                    moves.append(move)

            print(f"{game_state.steps} - {round_actions}")
            game_state = game_state.apply_move(round_actions)

        return np.array(boards), np.array(moves)
