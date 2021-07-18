import base64
import bz2
import pickle

import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, Configuration
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l1_l2

from encoders.seventeen_plane_encoder import SeventeenPlaneEncoder
from game_state import GameState
from goose import Goose
from neural_network_train.networks import conv_bn_padding
from utils import center_matrix

np.random.seed(123)

X = np.load('/content/drive/MyDrive/TFM/features.npz', allow_pickle=True)['data']
Y = np.load('/content/drive/MyDrive/TFM/labels.npz', allow_pickle=True)['data']

samples = X.shape[0]

board_rows, board_cols = 7, 11
encoder = SeventeenPlaneEncoder(board_cols, board_rows)

input_channels = encoder.num_planes
input_size = input_channels * board_rows * board_cols
input_shape = (board_rows, board_cols, input_channels)

X = np.transpose(X, (0, 2, 3, 1))  # Channels last

# Hold back a X% of the data for a test set; train on the other 100% - X%
train_samples = int(0.8 * samples)
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

network_layers = conv_bn_padding.layers(input_shape, num_layers=12)

# Model Callbacks
callbacks = [
    EarlyStopping(monitor="val_accuracy",
                  min_delta=0.01,
                  patience=25,
                  verbose=1,
                  mode="max",
                  baseline=None,
                  restore_best_weights=True),
    ModelCheckpoint('/content/drive/MyDrive/TFM/weights_checkpoint.hdf5',
                    monitor='val_accuracy',
                    verbose=1,
                    save_best_only=True,
                    mode='max'),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=5, min_lr=0.0001, verbose=1)
]

model = Sequential()
for layer in network_layers:
    model.add(layer)
model.add(Dense(4, activation='softmax', kernel_regularizer=l1_l2(l1=0.0005, l2=0.0005)))
model.summary()

sgd = SGD(learning_rate=0.01, momentum=0.8, clipvalue=0.5)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=128,
          epochs=500,
          verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=callbacks)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

with open("/content/drive/MyDrive/TFM/model.txt", "wb") as f:
    f.write(base64.b64encode(bz2.compress(pickle.dumps(model.to_json()))))
with open("/content/drive/MyDrive/TFM/weights.txt", "wb") as f:
    f.write(base64.b64encode(bz2.compress(pickle.dumps(model.get_weights()), 1)))

############################
# Model evaluation
############################
configuration = Configuration({"columns": board_cols,
                               "rows": board_rows,
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

board_tensor = center_matrix(encoder.encode(game_state, 0))
input_board = np.transpose(board_tensor, (1, 2, 0))
input_board = input_board.reshape((-1, board_rows, board_cols, encoder.num_planes))

# Avoid suicide: body + opposite_side - my tail
obstacles = input_board[:, :, :, [8, 9, 10, 11, 12]].max(axis=3) - input_board[:, :, :, [4, 5, 6, 7]].max(axis=3)
obstacles = np.array([obstacles[0, 2, 5], obstacles[0, 3, 6], obstacles[0, 4, 5], obstacles[0, 3, 4]])

action_probabilities = model.predict(input_board)[0]
action_probabilities = action_probabilities - obstacles

print(action_probabilities)
print(encoder.decode_action_index(np.argmax(action_probabilities).item()))
