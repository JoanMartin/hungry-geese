import copy
from random import choice

from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from numpy import mean

from game_state import GameState
from random_goose import RandomGoose


class MonteCarlo:

    def __init__(self, num_rounds: int, deep: int):
        self.num_rounds = num_rounds
        self.deep = deep

    def select_best_action(self, game_state: GameState, goose_index: int):
        if not len(game_state.geese[goose_index].positions) > 0:
            return None

        action_rewards = {action: [] for action in Action}

        for i in range(self.num_rounds):
            root_game_state = copy.deepcopy(game_state)

            moves = RandomGoose.select_moves(root_game_state)
            new_game_state = root_game_state.apply_move(moves)
            reward = self._simulate_random_game(new_game_state, self.deep, goose_index)
            action_rewards[moves[goose_index]].append(reward)

        x = {k: mean(v) for k, v in action_rewards.items() if v}
        max_value = max(x.values())

        return choice([k for k, v in x.items() if v == max_value])

    def _simulate_random_game(self, game_state: GameState, deep: int, goose_index: int):
        if deep == 0 or game_state.is_over():
            return game_state.geese[goose_index].reward

        moves = RandomGoose.select_moves(game_state)
        new_game_state = game_state.apply_move(moves)

        return self._simulate_random_game(new_game_state, deep - 1, goose_index)
