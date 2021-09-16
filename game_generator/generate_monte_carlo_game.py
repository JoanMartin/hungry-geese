import copy

from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration

from game_state import GameState
from monte_carlo import MonteCarlo


class GenerateMonteCarloGame:

    def __init__(self, configuration: Configuration, number_geese: int):
        self.configuration = configuration
        self.number_geese = number_geese

    def generate_game(self, num_rounds: int, deep: int):
        game_states, actions = [], []

        monte_carlo = MonteCarlo(num_rounds, deep)
        game_state = GameState.new_game(self.number_geese, self.configuration)

        while not game_state.is_over():
            step_actions = []

            for index, goose in enumerate(game_state.geese):
                action = monte_carlo.select_best_action(game_state, index)
                step_actions.append(action)

            game_states.append(copy.deepcopy(game_state))
            actions.append(copy.deepcopy(step_actions))

            game_state = game_state.apply_move(step_actions)

        return game_states, actions
