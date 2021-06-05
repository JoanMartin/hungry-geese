import random

from kaggle_environments.envs.hungry_geese.hungry_geese import Action

ACTIONS = [a for a in Action]

last_action = None


def agent(obs, config):
    global last_action

    actions = ACTIONS.copy()
    if last_action:
        actions.remove(last_action.opposite())
    action = random.choice(actions)
    last_action = action

    return action.name
