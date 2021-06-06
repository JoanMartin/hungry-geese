# import time

from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration, Observation

from game_state import GameState
from goose import Goose
from mcts import MCTS
from utils import calculate_last_action

last_observation = None


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
              calculate_last_action(last_observation.geese[index][0], positions[0], columns, rows)
              if len(positions) > 0 else None)
        for index, positions in enumerate(observation.geese)
    ]

    game_state = GameState(geese, observation.food, configuration, observation.step + 1)

    monte_carlo = MCTS(400, 6, 1.5)
    action = monte_carlo.select_move(game_state, observation.index)

    last_observation = observation

    # end = time.time()
    # print(f"{observation.index} Time: {end - start}")

    return action.name
