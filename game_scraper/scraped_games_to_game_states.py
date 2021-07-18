import json
import pickle
import random
from glob import glob

from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration, Action

from game_state import GameState
from goose import Goose


def create_game_state(steps):
    configuration = Configuration({"columns": 11,
                                   "rows": 7,
                                   "hunger_rate": 40,
                                   "min_food": 2,
                                   "max_length": 99})
    game_states = []
    actions = []
    for current_step, next_step in zip(steps, steps[1:]):
        step_number = current_step[0]["observation"]["step"]
        food = current_step[0]["observation"]["food"]
        geese_positions = current_step[0]["observation"]["geese"]

        geese = []
        step_actions = []
        for current_goose, next_goose in zip(current_step, next_step):
            index = current_goose["observation"]["index"]
            action = Action[next_goose["action"]] \
                if next_goose["action"] and current_goose["status"] == "ACTIVE" else None
            last_action = Action[current_goose["action"]] \
                if current_goose["action"] and step_number != 0 else None

            geese.append(Goose(index, geese_positions[index], last_action))
            step_actions.append(action)

        game_states.append(GameState(geese, food, configuration, step_number))
        actions.append(step_actions)

    return game_states, actions


def save_pickle(x, y):
    with open(r"../data/scraped_game_states.pickle", "wb") as f:
        pickle.dump(x, f)
    with open(r"../data/scraped_actions.pickle", "wb") as f:
        pickle.dump(y, f)


def main():
    xs = []
    ys = []

    for f_name in random.sample(glob('scraped_games/*.json'), 1500):
        with open(f_name) as f:
            episode = json.load(f)

        steps = episode["steps"]
        x, y = create_game_state(steps)
        xs.append(x)
        ys.append(y)

    save_pickle(xs, ys)


if __name__ == '__main__':
    main()
