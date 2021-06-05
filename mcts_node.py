import copy
import itertools
import random

from kaggle_environments.envs.hungry_geese.hungry_geese import translate, Action

from game_state import GameState


class MCTSNode:

    def __init__(self, game_state: GameState, parent=None, moves=None):
        self.game_state = game_state
        self.parent = parent
        self.moves = moves
        self.moves_history = []

        self.rewards = []

        self.num_rollouts = 0
        self.children = []
        self.unvisited_moves = self._legal_moves()

    def _legal_moves(self):
        all_actions = [action for action in Action]
        moves = []

        for goose in self.game_state.geese:
            candidates = []

            if goose.status == "ACTIVE":
                for action in all_actions:
                    if not goose.is_last_action(action):
                        new_position = translate(goose.positions[0],
                                                 action,
                                                 self.game_state.columns,
                                                 self.game_state.rows)
                        if self.game_state.is_valid_position(new_position):
                            candidates.append(action)

                moves.append(candidates)
            else:
                moves.append([None])

        return list(itertools.product(*moves))

    def add_random_child(self):
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_moves = self.unvisited_moves.pop(index)
        self.moves_history.append(new_moves)

        new_game_state = copy.deepcopy(self.game_state)
        new_game_state = new_game_state.apply_move(new_moves)
        new_node = MCTSNode(new_game_state, self, new_moves)

        self.children.append(new_node)
        return new_node

    def can_add_child(self):
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        return self.game_state.is_over()

    def record_reward(self, reward: int):
        self.rewards.append(reward)
        self.num_rollouts += 1

    def reward_average(self):
        try:
            a = [float(i) / max(self.rewards) for i in self.rewards]
        except ZeroDivisionError:
            a = [0]
        return sum(a) / float(self.num_rollouts)
