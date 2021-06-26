from tensorflow.keras.layers import Dense, Flatten, Conv2D, \
    ZeroPadding2D, Dropout, GlobalAveragePooling2D

from tensorflow.keras.regularizers import l2


def layers(input_shape, num_layers=7, filters=32, kernel=3, weight_decay=1e-7):
    x = _conv_dropout(filters, kernel, weight_decay, input_shape)

    for i in range(num_layers):
        x.extend(_conv_dropout(filters, kernel, weight_decay) * num_layers)

    x.extend([
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(1024, activation='relu')
    ])

    return x


def _conv_dropout(filters, kernel, weight_decay, input_shape=None):
    if input_shape:
        zero_padding = ZeroPadding2D(padding=(2, 2), input_shape=input_shape)
    else:
        zero_padding = ZeroPadding2D(padding=(2, 2))

    return [
        zero_padding,
        Conv2D(filters, kernel_size=kernel, activation='relu', kernel_regularizer=l2(weight_decay)),
        Dropout(0.15)
    ]
