import math
from kaggle_environments.envs.hungry_geese.hungry_geese import Action

from game_state import GameState
from mcts_node import MCTSNode
from random_goose import RandomGoose


class MCTS:

    def __init__(self, num_rounds: int, deep: int, temperature: float):
        self.num_rounds = num_rounds
        self.deep = deep
        self.action_rewards = {action: [] for action in Action}

        # Temperature is the parameter 'c'. With a larger value of 'c', more time
        # is spent visiting the least-explored nodes. With a smaller value of 'c',
        # more time is spent gathering a better evaluation of the most promising node.
        self.temperature = temperature

    def select_move(self, game_state: GameState, my_goose_index: int):
        root = MCTSNode(game_state)

        for i in range(self.num_rounds):
            node = root

            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self._select_child(node)

            # Adds a new child node into the tree
            if node.can_add_child():
                node = node.add_random_child()

            # Simulated a random game from this node
            reward = self._simulate_random_game(node.game_state, self.deep, my_goose_index)

            # Propagates the score back up the tree
            while node is not None:
                node.record_reward(reward)
                node = node.parent

        best_move = self._select_best_move(root, my_goose_index)
        return best_move

    def _select_child(self, node: MCTSNode):
        total_rollouts = sum(child.num_rollouts for child in node.children)

        best_score = -1
        best_child = None
        for child in node.children:
            score = self._uct_score(total_rollouts,
                                    child.num_rollouts,
                                    child.reward_average(),
                                    self.temperature)

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    @staticmethod
    def _select_best_move(node: MCTSNode, my_goose_index: int):
        best_move = None
        best_pct = -1.0
        for child in node.children:
            child_pct = child.reward_average()
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.moves[my_goose_index]

        return best_move

    @staticmethod
    def _uct_score(parent_rollouts, child_rollouts, win_pct, temperature):
        exploration = math.sqrt(math.log(parent_rollouts) / child_rollouts)
        return win_pct + temperature * exploration

    def _simulate_random_game(self, game_state: GameState, deep: int, my_goose_index: int):
        if deep == 0 or game_state.is_over():
            return game_state.geese[my_goose_index].reward

        moves = RandomGoose.select_moves(game_state)
        new_game_state = game_state.apply_move(moves)

        return self._simulate_random_game(new_game_state, deep - 1, my_goose_index)
