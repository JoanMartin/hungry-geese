# This is a copy of the notebook "Hungry Geese - Simple Monte Carlo (baseline)"
# https://www.kaggle.com/johntermaat/hungry-geese-simple-monte-carlo-baseline


import copy
import random

from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col

frame = 0
opposites = {Action.EAST: Action.WEST, Action.WEST: Action.EAST, Action.NORTH: Action.SOUTH, Action.SOUTH: Action.NORTH}
action_meanings = {Action.EAST: (1, 0), Action.WEST: (-1, 0), Action.NORTH: (0, -1), Action.SOUTH: (0, 1)}
action_names = {(1, 0): Action.EAST, (-10, 0): Action.EAST, (-1, 0): Action.WEST, (10, 0): Action.WEST,
                (0, -1): Action.NORTH, (0, 6): Action.NORTH, (0, -6): Action.SOUTH, (0, 1): Action.SOUTH}
strValue = {Action.EAST: 'EAST', Action.WEST: 'WEST', Action.NORTH: 'NORTH', Action.SOUTH: 'SOUTH'}
all_last_actions = [None, None, None, None]
revert_last_actions_list = [None, None, None, None]
last_observation = None


class Obs:
    pass


def set_last_actions(observation, configuration):
    global frame, revert_last_actions_list, all_last_actions
    if not frame == 0:
        for i in range(4):
            set_last_action(observation, configuration, i)
    revert_last_actions_list = copy.deepcopy(all_last_actions)


def revert_last_actions():
    global revert_last_actions_list, all_last_actions
    all_last_actions = copy.deepcopy(revert_last_actions_list)


def set_last_action(observation, configuration, goose_index):
    global last_observation, all_last_actions, action_names
    if len(observation.geese[goose_index]) > 0:
        old_goose_row, old_goose_col = row_col(last_observation.geese[goose_index][0], configuration.columns)
        new_goose_row, new_goose_col = row_col(observation.geese[goose_index][0], configuration.columns)
        all_last_actions[goose_index] = action_names[
            ((new_goose_col - old_goose_col) % configuration.columns,
             (new_goose_row - old_goose_row) % configuration.rows)]


def get_valid_directions(observation, configuration, goose_index):
    global all_last_actions, opposites
    directions = [Action.EAST, Action.WEST, Action.NORTH, Action.SOUTH]
    return_directions = []
    for direction in directions:
        row, col = get_row_col_for_action(observation, configuration, goose_index, direction)
        if not will_goose_be_there(observation, configuration, row, col) and \
                not all_last_actions[goose_index] == opposites[direction]:
            return_directions.append(direction)
    if len(return_directions) == 0:
        return directions
    return return_directions


def random_turn(observation, configuration, action_overrides, rewards, fr):
    new_observation = clone_observation(observation)
    for i in range(4):
        if len(observation.geese[i]) > 0:
            if i in action_overrides.keys():
                new_observation = perform_action_for_goose(observation, configuration, i, new_observation,
                                                           action_overrides[i])
            else:
                new_observation = random_action_for_goose(observation, configuration, i, new_observation)

    check_for_collisions(new_observation)
    update_rewards(new_observation, rewards, fr)
    hunger(new_observation, fr)
    return new_observation


def hunger(observation, fr):
    if fr % 40 == 0:
        for g, goose in enumerate(observation.geese):
            goose = goose[0:len(goose) - 1]


def update_rewards(observation, rewards, fr):
    for g, goose in enumerate(observation.geese):
        if len(goose) > 0:
            rewards[g] = 2 * fr + len(goose)


def check_for_collisions(observation):
    killed = []
    for g, goose in enumerate(observation.geese):
        if len(goose) > 0:
            for o, otherGoose in enumerate(observation.geese):
                for p, part in enumerate(otherGoose):
                    if not (o == g and p == 0):
                        if goose[0] == part:
                            killed.append(g)

    for kill in killed:
        observation.geese[kill] = []


def clone_observation(observation):
    new_observation = Obs()
    new_observation.index = observation.index
    new_observation.geese = copy.deepcopy(observation.geese)
    new_observation.food = copy.deepcopy(observation.food)
    return new_observation


def random_action_for_goose(observation, configuration, goose_index, new_observation):
    valid_actions = get_valid_directions(observation, configuration, goose_index)
    action = random.choice(valid_actions)
    row, col = get_row_col_for_action(observation, configuration, goose_index, action)
    new_observation.geese[goose_index] = [row * configuration.columns + col] + new_observation.geese[goose_index]
    if not is_food_there(observation, configuration, row, col):
        new_observation.geese[goose_index] = new_observation.geese[goose_index][
                                             0:len(new_observation.geese[goose_index]) - 1]
    return new_observation


def perform_action_for_goose(observation, configuration, goose_index, new_observation, action):
    row, col = get_row_col_for_action(observation, configuration, goose_index, action)
    new_observation.geese[goose_index][:0] = [row * configuration.columns + col]
    if not is_food_there(observation, configuration, row, col):
        new_observation.geese[goose_index] = new_observation.geese[goose_index][
                                             0:len(new_observation.geese[goose_index]) - 1]
    return new_observation


def is_food_there(observation, configuration, row, col):
    for food in observation.food:
        food_row, food_col = row_col(food, configuration.columns)
        if food_row == row and food_col == col:
            return True
    return False


def will_goose_be_there(observation, configuration, row, col):
    for goose in observation.geese:
        for p, part in enumerate(goose):
            if not p == len(goose) - 1:
                part_row, part_col = row_col(part, configuration.columns)
                if part_row == row and part_col == col:
                    return True
    return False


def get_row_col_for_action(observation, configuration, goose_index, action):
    global action_meanings
    goose_row, goose_col = row_col(observation.geese[goose_index][0], configuration.columns)
    action_row = (goose_row + action_meanings[action][1]) % configuration.rows
    action_col = (goose_col + action_meanings[action][0]) % configuration.columns
    return action_row, action_col


def simulate_match(observation, configuration, first_move, depth):
    global frame
    action_overrides = {observation.index: first_move}
    revert_last_actions()
    simulation_frame = frame + 1
    new_observation = clone_observation(observation)
    rewards = [0, 0, 0, 0]
    count = 0
    while count < depth:
        new_observation = random_turn(new_observation, configuration, action_overrides, rewards, simulation_frame)
        action_overrides = {}
        simulation_frame += 1
        count += 1
    return rewards


def simulate_matches(observation, configuration, num_matches, depth):
    options = get_valid_directions(observation, configuration, observation.index)
    reward_totals = []
    for o, option in enumerate(options):
        rewards_for_option = [0, 0, 0, 0]
        for i in range(num_matches):
            match_rewards = simulate_match(observation, configuration, option, depth)
            for j in range(4):
                rewards_for_option[j] += match_rewards[j]
        reward_totals.append(rewards_for_option)
    scores = []
    for o, option in enumerate(options):
        rewards = reward_totals[o]
        if len(rewards) <= 0:
            mean = 0
        else:
            mean = sum(rewards) / len(rewards)
        if mean == 0:
            scores.append(0)
        else:
            scores.append(rewards[observation.index] / mean)

    return options[scores.index(max(scores))]


def agent(obs_dict, config_dict):
    global last_observation, all_last_actions, opposites, frame
    observation = Observation(obs_dict)
    configuration = Configuration(config_dict)
    set_last_actions(observation, configuration)
    my_length = len(observation.geese[observation.index])
    if my_length < 5:
        my_action = simulate_matches(observation, configuration, 300, 3)
    elif my_length < 9:
        my_action = simulate_matches(observation, configuration, 120, 6)
    else:
        my_action = simulate_matches(observation, configuration, 85, 9)

    last_observation = clone_observation(observation)
    frame += 1
    return strValue[my_action]
