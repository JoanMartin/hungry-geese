import pickle

import numpy as np

from encoders.seventeen_plane_encoder import SeventeenPlaneEncoder


def main():
    game_states_encoded, actions_encoded = [], []
    encoder = SeventeenPlaneEncoder(11, 7)

    with open(r"../data/game_states.pickle", "rb") as f:
        game_states = pickle.load(f)

    with open(r"../data/actions.pickle", "rb") as f:
        round_actions = pickle.load(f)

    game_states = [game_state for i in game_states for game_state in i]
    round_actions = [actions for i in round_actions for actions in i]

    for game_state, actions in zip(game_states, round_actions):
        for index, action in enumerate(actions):
            if action:
                # The encoded board situation is appended to game_states_encoded
                game_states_encoded.append(encoder.encode(game_state, index))

                # The one-hot-encoded action is appended to actions_encoded
                action_one_hot = np.zeros(encoder.num_actions())
                action_one_hot[encoder.encode_action(action)] = 1
                actions_encoded.append(action_one_hot)

    x, y = np.array(game_states_encoded), np.array(actions_encoded)

    np.save("../data/features.npy", np.concatenate([x]))
    np.save("../data/labels.npy", np.concatenate([y]))


if __name__ == '__main__':
    main()
