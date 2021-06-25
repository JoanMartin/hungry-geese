from tensorflow.keras.layers import Dense, Flatten, Conv2D, ZeroPadding2D, \
    GlobalAveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2


def layers(input_shape):
    weight_decay = 1e-7

    return [
        ZeroPadding2D(padding=(3, 3), input_shape=input_shape),
        Conv2D(64, kernel_size=(7, 7), kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation("relu"),

        ZeroPadding2D(padding=(2, 2)),
        Conv2D(64, kernel_size=(5, 5), kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation("relu"),

        ZeroPadding2D(padding=(2, 2)),
        Conv2D(64, kernel_size=(5, 5), kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation("relu"),

        ZeroPadding2D(padding=(2, 2)),
        Conv2D(48, kernel_size=(5, 5), kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation("relu"),

        ZeroPadding2D(padding=(2, 2)),
        Conv2D(48, kernel_size=(5, 5), kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation("relu"),

        ZeroPadding2D(padding=(2, 2)),
        Conv2D(32, kernel_size=(5, 5), kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation("relu"),

        ZeroPadding2D(padding=(2, 2)),
        Conv2D(32, kernel_size=(5, 5), kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation("relu"),

        GlobalAveragePooling2D(),

        Flatten(),
        Dense(1024, activation='relu'),
    ]
