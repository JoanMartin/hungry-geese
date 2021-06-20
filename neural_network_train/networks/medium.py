from tensorflow.keras.layers import Dense, Flatten, Conv2D, ZeroPadding2D, Dropout, MaxPooling2D


def layers(input_shape):
    return [
        ZeroPadding2D(padding=(3, 3), input_shape=input_shape),
        Conv2D(64, kernel_size=(7, 7), activation='relu'),
        Dropout(rate=0.5),

        ZeroPadding2D(padding=(2, 2)),
        Conv2D(64, kernel_size=(5, 5), activation='relu'),
        Dropout(rate=0.5),

        ZeroPadding2D(padding=(2, 2)),
        Conv2D(64, kernel_size=(5, 5), activation='relu'),
        Dropout(rate=0.5),

        ZeroPadding2D(padding=(2, 2)),
        Conv2D(48, kernel_size=(5, 5), activation='relu'),
        Dropout(rate=0.5),

        ZeroPadding2D(padding=(2, 2)),
        Conv2D(48, kernel_size=(5, 5), activation='relu'),
        Dropout(rate=0.5),

        ZeroPadding2D(padding=(2, 2)),
        Conv2D(32, kernel_size=(5, 5), activation='relu'),
        Dropout(rate=0.5),

        ZeroPadding2D(padding=(2, 2)),
        Conv2D(32, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(rate=0.5),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(rate=0.5),
    ]
