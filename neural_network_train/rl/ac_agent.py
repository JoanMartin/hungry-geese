import numpy as np
from tensorflow.keras.optimizers import SGD

import kerasutil
from encoders.seventeen_plane_encoder import SeventeenPlaneEncoder
from utils import center_matrix


def clip_probs(original_probs, temperature):
    move_probs = np.power(original_probs, 1.0 / temperature)
    move_probs = move_probs / np.sum(move_probs) if np.sum(move_probs) != 0 else np.zeros(move_probs.shape)

    eps = 1e-6
    move_probs = np.clip(move_probs, eps, 1 - eps)

    # Re-normalize to get another probability distribution.
    move_probs = move_probs / np.sum(move_probs)

    return move_probs


class ACAgent:

    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self.collector = None
        self.temperature = 1.0

        self.last_state_value = 0

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state, goose_index: int):
        if not len(game_state.geese[goose_index].positions) > 0:
            return None

        board_rows = self.encoder.rows
        board_columns = self.encoder.columns
        num_actions = self.encoder.num_actions()

        board_tensor = self.encoder.encode(game_state, goose_index)
        board_tensor = center_matrix(board_tensor)
        x = np.transpose(board_tensor, (1, 2, 0))  # Channels last
        x = x.reshape((-1, board_rows, board_columns, self.encoder.num_planes))

        # Avoid suicide: body + opposite_side - my tail
        obstacles = x[:, :, :, [8, 9, 10, 11, 12]].max(axis=3) - x[:, :, :, [4, 5, 6, 7]].max(axis=3)
        obstacles = np.array([obstacles[0, 2, 5], obstacles[0, 3, 6], obstacles[0, 4, 5], obstacles[0, 3, 4]])

        actions, values = self.model.predict(x)
        action_probabilities = actions[0]
        action_probabilities = action_probabilities - obstacles
        action_probabilities[action_probabilities < 0] = 0
        estimated_value = values[0][0]
        self.last_state_value = float(estimated_value)

        # Prevent move probs from getting stuck at 0 or 1.
        move_probs = clip_probs(action_probabilities, self.temperature)

        # Turn the probabilities into a ranked list of moves.
        action_candidates = np.arange(num_actions)
        ranked_moves = np.random.choice(action_candidates, num_actions, replace=False, p=move_probs)

        final_action_idx = ranked_moves[0]

        if self.collector is not None:
            self.collector.record_decision(state=board_tensor,
                                           action=final_action_idx,
                                           estimated_value=estimated_value)

        final_action = self.encoder.decode_action_index(final_action_idx.item())

        return final_action

    def train(self, experience, lr=0.01, batch_size=128):
        opt = SGD(learning_rate=lr, momentum=0.9, clipvalue=0.5)
        self.model.compile(optimizer=opt,
                           loss=['categorical_crossentropy', 'mse'],
                           loss_weights=[1.0, 0.5],
                           metrics=['accuracy', 'mse'])

        n = experience.states.shape[0]
        num_actions = self.encoder.num_actions()
        policy_target = np.zeros((n, num_actions))
        value_target = np.zeros((n,))
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            policy_target[i][action] = experience.advantages[i]
            value_target[i] = reward

        experience_states = np.transpose(experience.states, (0, 2, 3, 1))  # Channels last

        train_samples = int(0.8 * n)
        x_train, x_test = experience_states[:train_samples], experience_states[train_samples:]
        y1_train, y1_test = policy_target[:train_samples], policy_target[train_samples:]
        y2_train, y2_test = value_target[:train_samples], value_target[train_samples:]

        self.model.fit(x_train, [y1_train, y2_train], batch_size=batch_size, epochs=1)

        score = self.model.evaluate(x_test, [y1_test, y2_test], verbose=0)
        print('Score:', score)

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.columns
        h5file['encoder'].attrs['board_height'] = self.encoder.rows
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])

    def diagnostics(self):
        return {'value': self.last_state_value}


def load_ac_agent(h5file):
    model = kerasutil.load_model_from_hdf5_group(h5file['model'])
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')

    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = SeventeenPlaneEncoder(board_width, board_height)

    return ACAgent(model, encoder)
