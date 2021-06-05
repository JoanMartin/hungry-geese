import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

np.random.seed(123)

X = np.load('generated_games/features.npy', allow_pickle=True)
Y = np.load('generated_games/labels.npy', allow_pickle=True)

samples = X.shape[0]

board_rows, board_cols = 7, 11
input_channels = 3
input_size = input_channels * board_rows * board_cols
input_shape = (input_channels, board_rows, board_cols)

# Transform the input into vectors of size 81, instead of 9 x 9 matrices
X = X.reshape(samples, input_channels * board_rows * board_cols)
Y = Y.reshape(samples)

# Hold back 10% of the data for a test set; train on the other 90%
train_samples = int(0.8 * samples)
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

c = {Action.NORTH: 0, Action.EAST: 1, Action.SOUTH: 2, Action.WEST: 3}
Y_train = to_categorical([c[i] for i in Y_train], 4)
Y_test = to_categorical([c[i] for i in Y_test], 4)

model = Sequential()
model.add(Dense(1000, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.summary()

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=10,
          epochs=25,
          verbose=1,
          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
