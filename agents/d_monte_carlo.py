# import time

from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration, translate, Action, Observation

from game_state import GameState
from goose import Goose
from monte_carlo import MonteCarlo

last_observation = None


def _calculate_last_action(last_head: int, new_head: int, columns: int, rows: int):
    for action in Action:
        potential_position = translate(last_head, action, columns, rows)
        if new_head == potential_position:
            return action

    return None


def agent(obs, config):
    # start = time.time()

    global last_observation

    observation = Observation(obs)
    if not last_observation:
        last_observation = observation

    configuration = Configuration(config)
    columns = configuration.columns
    rows = configuration.rows

    print(f"\n\n{observation.index} Step {observation.step} - {observation.geese[observation.index]}\n")

    geese = [
        Goose(index,
              positions,
              _calculate_last_action(last_observation.geese[index][0], positions[0], columns, rows)
              if len(positions) > 0 else None)
        for index, positions in enumerate(observation.geese)
    ]

    game_state = GameState(geese, observation.food, configuration, observation.step + 1)

    my_length = len(observation.geese[observation.index])
    if my_length < 6:
        monte_carlo = MonteCarlo(250, 4)
    elif my_length < 9:
        monte_carlo = MonteCarlo(350, 6)
    else:
        monte_carlo = MonteCarlo(500, 8)

    action = monte_carlo.select_best_move(game_state, observation.index)

    last_observation = observation

    # end = time.time()
    # print(f"{observation.index} Time: {end - start}")

    return action.name
