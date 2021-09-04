from tensorflow.keras.layers import LeakyReLU, Dense, Flatten, Conv2D, \
    GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2


def layers(input_shape, num_layers=7, filters=48, kernel=5, weight_decay=1e-7):
    x = _conv_bn(filters, kernel, weight_decay, input_shape)

    for i in range(num_layers):
        x.extend(_conv_bn(filters, kernel, weight_decay))

    x.extend([
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(1024),
        LeakyReLU(),
        Dropout(rate=0.3),
        Dense(512),
        LeakyReLU(),
        Dropout(rate=0.3)
    ])

    return x


def _conv_bn(filters, kernel, weight_decay, input_shape=None):
    if input_shape:
        conv = Conv2D(filters, kernel_size=kernel, padding='same',
                      kernel_regularizer=l2(weight_decay), input_shape=input_shape)
    else:
        conv = Conv2D(filters, kernel_size=kernel, padding='same',
                      kernel_regularizer=l2(weight_decay))

    return [
        conv,
        BatchNormalization(),
        LeakyReLU(),
    ]
