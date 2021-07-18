import base64
import bz2
import pickle

import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration, Observation
from tensorflow.keras.models import model_from_json

from encoders.seventeen_plane_encoder import SeventeenPlaneEncoder
from game_state import GameState
from goose import Goose
from nn_config import MODEL_JSON, MODEL_WEIGHTS
from utils import calculate_last_action, center_matrix

model_json = pickle.loads(bz2.decompress(base64.b64decode(MODEL_JSON)))
model_weights = pickle.loads(bz2.decompress(base64.b64decode(MODEL_WEIGHTS)))

model = model_from_json(model_json)
model.set_weights(model_weights)

last_observation = None


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

    game_state = GameState(geese, observation.food, configuration, observation.step)
    encoder = SeventeenPlaneEncoder(configuration.columns, configuration.rows)
    board_tensor = center_matrix(encoder.encode(game_state, observation.index))

    input_board = np.transpose(board_tensor, (1, 2, 0))
    input_board = input_board.reshape((-1, rows, columns, encoder.num_planes))

    # Avoid suicide: body + opposite_side - my tail
    obstacles = input_board[:, :, :, [8, 9, 10, 11, 12]].max(axis=3) - input_board[:, :, :, [4, 5, 6, 7]].max(axis=3)
    obstacles = np.array([obstacles[0, 2, 5], obstacles[0, 3, 6], obstacles[0, 4, 5], obstacles[0, 3, 4]])

    action_probabilities = model.predict(input_board)[0]
    action_probabilities = action_probabilities - obstacles

    last_observation = observation

    action = encoder.decode_action_index(np.argmax(action_probabilities).item())

    print(f"{observation.index} - {observation.step} - {action_probabilities} - {action.name}")

    return action.name
