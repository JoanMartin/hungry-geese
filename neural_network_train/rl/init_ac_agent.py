import argparse

import h5py
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2

from encoders.seventeen_plane_encoder import SeventeenPlaneEncoder
from neural_network_train.networks import conv_bn_padding
from neural_network_train.rl.ac_agent import ACAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file')
    args = parser.parse_args()

    board_rows, board_cols = 7, 11
    encoder = SeventeenPlaneEncoder(board_cols, board_rows)

    input_channels = encoder.num_planes
    input_shape = (board_rows, board_cols, input_channels)

    board_input = Input(shape=input_shape, name='board_input')

    processed_board = board_input
    network_layers = conv_bn_padding.layers(input_shape, num_layers=12)
    for layer in network_layers:
        processed_board = layer(processed_board)

    policy_output = Dense(4, activation='softmax', kernel_regularizer=l1_l2(l1=0.0005, l2=0.0005))(processed_board)
    value_output = Dense(1, activation='tanh')(processed_board)

    model = Model(inputs=[board_input], outputs=[policy_output, value_output])

    new_agent = ACAgent(model, encoder)
    with h5py.File(args.output_file, 'w') as outf:
        new_agent.serialize(outf)


if __name__ == '__main__':
    main()
