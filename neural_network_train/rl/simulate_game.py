from typing import List

from kaggle_environments.helpers import Configuration

from game_state import GameState
from neural_network_train.rl.ac_agent import ACAgent


class SimulateGame:

    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def simulate(self, agents: List[ACAgent]):
        game_state = GameState.new_game(len(agents), self.configuration)

        while not game_state.is_over():
            step_actions = []

            for index, goose in enumerate(game_state.geese):
                action = agents[index].select_move(game_state, index)
                step_actions.append(action)

            game_state = game_state.apply_move(step_actions)

        return game_state
