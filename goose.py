from typing import List

from kaggle_environments.envs.hungry_geese.hungry_geese import Action


class Goose:

    def __init__(self, index: int, positions: List[int], last_action: Action = None):
        self.index = index
        self.positions = positions
        self.status = "ACTIVE" if len(positions) > 0 else "DONE"
        self.reward = 0
        self.last_action = last_action

    def update_reward(self, step: int, max_length: int):
        self.reward = (step + 1) * (max_length + 1) + (len(self.positions) ** 5)
        # self.reward = (step + 1) * (max_length + 1) \
        #               + (100 * len(self.positions) * 3 * (1 / (len(self.positions) ** 0.2)))

    def update_status(self, new_status):
        self.status = new_status

    def is_last_action(self, action: Action):
        return self.last_action is not None and action == self.last_action.opposite()

    def __repr__(self):
        return f"{self.index} - {self.reward} - {self.status} - {self.positions} - {self.last_action}"
