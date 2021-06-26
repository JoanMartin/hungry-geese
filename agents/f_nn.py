import base64
import bz2
import pickle

import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration, Observation
from tensorflow.keras.models import model_from_json

from encoders.four_plane_encoder import FourPlaneEncoder
from game_state import GameState
from goose import Goose
from nn_config import MODEL_JSON, MODEL_WEIGHTS
from utils import calculate_last_action

model_json = pickle.loads(bz2.decompress(base64.b64decode(MODEL_JSON)))
model_weights = pickle.loads(bz2.decompress(base64.b64decode(MODEL_WEIGHTS)))

model = model_from_json(model_json)
model.set_weights(model_weights)

last_observation = None
eps = 1e-6


def agent(obs, config):
    global last_observation

    observation = Observation(obs)
    if not last_observation:
        last_observation = observation

    configuration = Configuration(config)
    columns = configuration.columns
    rows = configuration.rows

    geese = [
        Goose(index,
              positions,
              calculate_last_action(last_observation.geese[index][0], positions[0], columns, rows)
              if len(positions) > 0 else None)
        for index, positions in enumerate(observation.geese)
    ]

    game_state = GameState(geese, observation.food, configuration, observation.step + 1)
    encoder = FourPlaneEncoder(configuration.columns, configuration.rows)
    board_tensor = encoder.encode(game_state, 0)

    input_board = np.array([board_tensor]).reshape((-1, rows, columns, encoder.num_planes))
    action_probabilities = model.predict(input_board)[0]
    action_probabilities = action_probabilities ** 3
    action_probabilities = np.clip(action_probabilities, eps, 1 - eps)
    action_probabilities = action_probabilities / np.sum(action_probabilities)

    last_observation = observation

    action = encoder.decode_action_index(np.argmax(action_probabilities).item())
    return action.name
