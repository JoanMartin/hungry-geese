from tensorflow.keras.layers import Dense, Flatten, Conv2D, ZeroPadding2D, \
    GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Dropout
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
        Dropout(rate=0.5),
        Dense(512),
        LeakyReLU(),
        Dropout(rate=0.5)
    ])

    return x


def _conv_bn(filters, kernel, weight_decay, input_shape=None):
    if input_shape:
        zero_padding = ZeroPadding2D(padding=(2, 2), input_shape=input_shape)
    else:
        zero_padding = ZeroPadding2D(padding=(2, 2))

    return [
        zero_padding,
        Conv2D(filters, kernel_size=kernel, kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        LeakyReLU(),
    ]
