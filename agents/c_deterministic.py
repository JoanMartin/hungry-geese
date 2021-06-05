from random import choice
from typing import List

from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration, Observation, translate, \
    min_distance, adjacent_positions, row_col

last_action = None


def distance_to_new_position(current_position: int, new_position: int, columns: int) -> int:
    current_row, current_column = row_col(current_position, columns)
    new_row, new_column = row_col(new_position, columns)
    return abs(current_row - new_row) + abs(current_column - new_column)


def get_closest_food(my_goose_index: int, columns: int, food_list: List[int], geese: List[List[int]]) -> List[int]:
    closest_food: List[int] = []

    for food in food_list:
        min_goose_food_distance: int = None
        min_food_distance: int = None

        for index, goose in enumerate(geese):
            if len(goose) > 0:
                goose_head = goose[0]
                food_distance = distance_to_new_position(goose_head, food, columns)

                if min_food_distance is None or food_distance <= min_food_distance:
                    min_goose_food_distance = index
                    min_food_distance = food_distance

        if min_goose_food_distance is not None and min_goose_food_distance == my_goose_index:
            closest_food.append(food)

    return closest_food


def goose_adjacent_to_food(food_list: List[int], goose_head: int, columns: int, rows: int) -> bool:
    return any(adjacent_position in food_list for adjacent_position in adjacent_positions(goose_head, columns, rows))


def deepen_in_path(head_position: int,
                   depth: int,
                   columns: int,
                   rows: int,
                   bodies: set,
                   future_positions: List[int]):
    valid_paths = 0
    if depth != 0:
        for action in Action:
            new_position = translate(head_position, action, columns, rows)
            if new_position not in bodies and new_position not in future_positions:
                new_future_positions = future_positions[:]
                new_future_positions.append(new_position)
                valid_paths += deepen_in_path(new_position, depth - 1, columns, rows, bodies, new_future_positions)
    else:
        valid_paths += 1

    return valid_paths


def agent(obs, config):
    global last_action

    observation = Observation(obs)
    configuration = Configuration(config)

    rows, columns = configuration.rows, configuration.columns

    food_list = observation.food
    geese = observation.geese
    alive_opponents = [
        goose
        for index, goose in enumerate(geese)
        if index != observation.index and len(goose) > 0
    ]

    # Don't move adjacent to any heads
    head_adjacent_positions = {
        opponent_head_adjacent
        for opponent in alive_opponents
        for opponent_head in [opponent[0]]
        for opponent_head_adjacent in adjacent_positions(opponent_head, columns, rows)
    }
    # Don't move into any bodies except last position if it's not adjacent to food
    bodies = {
        position
        for goose in geese
        for position in goose
        if position != goose[len(goose) - 1] or goose_adjacent_to_food(food_list, goose[0], columns, rows)
    }

    closest_food = get_closest_food(observation.index, columns, food_list, geese)

    # Select possible actions
    position = geese[observation.index][0]

    all_actions = [action for action in Action]
    processed_actions = {}

    for action in all_actions:
        new_position = translate(position, action, columns, rows)
        possibilities = deepen_in_path(new_position,
                                       len(geese[observation.index]) // 2
                                       if len(geese[observation.index]) <= 1
                                       else len(geese[observation.index]) // 2 - 1,
                                       columns,
                                       rows,
                                       bodies,
                                       [new_position])

        possibilities_few_deep = deepen_in_path(new_position,
                                                2,
                                                columns,
                                                rows,
                                                bodies,
                                                [new_position])

        valid = new_position not in bodies and (last_action is None or action != last_action.opposite())
        adjacent_valid = valid and new_position not in head_adjacent_positions

        if ((possibilities_few_deep <= 1 or possibilities <= 1) and
                new_position == geese[observation.index][len(geese[observation.index]) - 1]):
            follow_queue = True
        else:
            follow_queue = False

        food_distance = float('inf')
        if closest_food and (valid or adjacent_valid):
            food_distance = min_distance(new_position, closest_food, columns)

        if valid:
            processed_actions[action] = (
                adjacent_valid, food_distance,
                possibilities, possibilities > 1, possibilities_few_deep > 1,
                follow_queue
            )

    action = sorted(processed_actions, key=lambda x: (-processed_actions[x][4],
                                                      -processed_actions[x][3],
                                                      -processed_actions[x][5],
                                                      -processed_actions[x][0],
                                                      processed_actions[x][1],
                                                      -processed_actions[x][2]))[0] \
        if any(processed_actions) else choice(all_actions)

    last_action = action
    return action.name
