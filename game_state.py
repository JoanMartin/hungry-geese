import copy
import random
from random import sample
from typing import List

from kaggle_environments.envs.hungry_geese.hungry_geese import translate, Configuration, adjacent_positions, Action
from kaggle_environments.helpers import histogram

from goose import Goose

adjacent_position = None


def _calculate_adjacent_positions(columns: int, rows: int):
    global adjacent_position

    adjacent_position = {
        i: adjacent_positions(i, columns, rows)
        for i in range(rows * columns)
    }


class GameState:

    def __init__(self,
                 geese: List[Goose],
                 food: List[int],
                 configuration: Configuration,
                 steps: int):
        self.geese = geese
        self.food = food

        self.steps = steps

        self.configuration = configuration
        self.columns = configuration.columns
        self.rows = configuration.rows
        self.min_food = configuration.min_food
        self.max_length = configuration.max_length
        self.hunger_rate = configuration.hunger_rate

        if not adjacent_position:
            _calculate_adjacent_positions(self.columns, self.rows)

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}

        a = copy.deepcopy(self.geese)
        b = copy.deepcopy(self.food)

        return GameState(a, b, self.configuration, self.steps)

    def is_head_adjacent_position(self, goose_index: int, position: int):
        opponents_head = [
            goose.positions[0]
            for goose in self.geese
            if goose.index != goose_index and len(goose.positions) > 0
        ]
        a = adjacent_position[position]

        return any(item in a for item in opponents_head)

    def is_valid_position(self, position: int):
        bodies = {position
                  for goose in self.geese
                  for position in goose.positions
                  if position != goose.positions[-1]}
        return position not in bodies

    def alive_geese(self):
        return len([goose for goose in self.geese if len(goose.positions) > 0])

    def is_over(self):
        return self.alive_geese() <= 1 or self.steps == 200

    def apply_move(self, moves: List[Action]):
        next_geese = self.geese
        next_food = self.food

        # Apply the actions from active agents.
        for index, goose in enumerate(next_geese):
            if goose.status != "ACTIVE":
                if goose.status != "INACTIVE" and goose.status != "DONE":
                    # ERROR, INVALID, or TIMEOUT, remove the goose.
                    goose.positions = []
                continue

            action = moves[index]

            # Check action direction
            if goose.is_last_action(action):
                goose.status = "DONE"
                goose.positions = []
                continue

            head = translate(goose.positions[0], action, self.columns, self.rows)

            # Consume food or drop a tail piece.
            if head in next_food:
                next_food.remove(head)
            else:
                goose.positions.pop()

            # Self collision.
            if head in goose.positions:
                goose.status = "DONE"
                goose.positions = []
                continue

            while len(goose.positions) >= self.max_length:
                goose.positions.pop()  # Free a spot for the new head if needed

            goose.positions.insert(0, head)  # Add New Head to the Goose.

            # If hunger strikes remove from the tail.
            if self.steps % self.hunger_rate == 0:
                if len(goose.positions) > 0:
                    goose.positions.pop()
                if len(goose.positions) == 0:
                    goose.status = "DONE"
                    continue

        goose_positions = histogram(
            position
            for goose in next_geese
            for position in goose.positions
        )

        # Check for collisions.
        for goose in next_geese:
            if len(goose.positions) > 0:
                head = goose.positions[0]
                if goose_positions[head] > 1:
                    goose.status = "DONE"
                    goose.positions = []

        # Add food if min_food threshold reached.
        needed_food = self.min_food - len(next_food)
        if needed_food > 0:
            collisions = {
                position
                for goose in next_geese
                for position in goose.positions
            }
            available_positions = set(range(self.rows * self.columns)) \
                .difference(collisions) \
                .difference(next_food)
            # Ensure we don't sample more food than available positions.
            needed_food = min(needed_food, len(available_positions))
            next_food.extend(sample(available_positions, needed_food))

        # Set rewards and update last action after deleting all geese
        # to ensure that geese don't receive a reward on the turn they perish.
        for index, goose in enumerate(next_geese):
            goose.last_action = moves[index]
            if goose.status == "ACTIVE":
                goose.update_reward(self.steps, self.max_length)
            else:
                goose.reward = 0

        # If only one ACTIVE agent left, set it to DONE.
        active_geese = [goose for goose in next_geese if goose.status == "ACTIVE"]
        if len(active_geese) == 1:
            goose = active_geese[0]
            goose.status = "DONE"

        return GameState(next_geese, next_food, self.configuration, self.steps + 1)

    @staticmethod
    def new_game(number_geese: int, configuration: Configuration):
        geese = []
        initial_positions = random.sample(range(configuration.rows * configuration.columns),
                                          number_geese + configuration.min_food)

        for i in range(number_geese):
            geese.append(Goose(i, [initial_positions[i]]))

        return GameState(geese, initial_positions[-2:], configuration, 1)
