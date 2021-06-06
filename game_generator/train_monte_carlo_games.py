import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from encoders.three_plane_encoder import ThreePlaneEncoder
from game_state import GameState
from goose import Goose

np.random.seed(123)

X = np.load('generated_games/features.npy', allow_pickle=True)
Y = np.load('generated_games/labels.npy', allow_pickle=True)

samples = X.shape[0]

board_rows, board_cols = 7, 11
input_channels = 3
input_size = input_channels * board_rows * board_cols
input_shape = (input_channels, board_rows, board_cols)

# Transform the input into vectors, instead of 3 x 7 x 11 matrices
X = X.reshape(samples, input_channels * board_rows * board_cols)

# Hold back 10% of the data for a test set; train on the other 90%
train_samples = int(0.8 * samples)
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

model = Sequential()
model.add(Dense(1000, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.summary()

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=64,
          epochs=5,
          verbose=1,
          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

############################
# Model evaluation
############################
configuration = Configuration({"columns": 11,
                               "rows": 7,
                               "hunger_rate": 40,
                               "min_food": 2,
                               "max_length": 99})
goose_white = Goose(0, [72], Action.NORTH)
goose_blue = Goose(1, [49, 60], Action.NORTH)
goose_green = Goose(2, [18, 7, 8], Action.SOUTH)
goose_red = Goose(3, [11, 22], Action.NORTH)

game_state = GameState([goose_white, goose_blue, goose_green, goose_red],
                       [10, 73],
                       configuration,
                       11)

encoder = ThreePlaneEncoder(configuration.columns, configuration.rows)
board_tensor = encoder.encode(game_state, 0)

X = np.array([board_tensor]).reshape(1, input_channels * board_rows * board_cols)
action_probabilities = model.predict(X)[0]

# Increase the distance between the move likely and least likely moves
action_probabilities = action_probabilities ** 3
eps = 1e-6
# Prevent move probabilities from getting stuck at 0 or 1
action_probabilities = np.clip(action_probabilities, eps, 1 - eps)
# Re-normalize to get another probability distribution.
action_probabilities = action_probabilities / np.sum(action_probabilities)

print(action_probabilities)
print(encoder.decode_action_index(np.argmax(action_probabilities).item()))
