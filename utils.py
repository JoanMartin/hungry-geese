import platform
import subprocess

import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Action, translate

from game_state import GameState

COLS = 'ABCDEFGHJKLMNOPQRST'


def print_game_state(game_state: GameState):
    columns = game_state.columns
    rows = game_state.rows
    bodies = {position: goose.index if position != goose.positions[0] else "*"
              for goose in game_state.geese
              for position in goose.positions}
    foods = {position: "F" for position in game_state.food}

    print("\n")

    for row in range(0, rows):
        bump = " " if row <= 9 else ""
        line = []

        for position in range(columns * row, columns * row + columns):
            string = foods.get(position, bodies.get(position, "·"))
            line.append(str(string))

        print('%s%d  %s' % (bump, row, '  '.join(line)))

    print('    ' + '  '.join(COLS[:columns]))


def clear_screen():
    # see https://stackoverflow.com/a/23075152/323316
    if platform.system() == "Windows":
        subprocess.Popen("cls", shell=True).communicate()
    else:  # Linux and Mac
        # the link uses print("\033c", end=""), but this is the original sequence given in the book.
        print(chr(27) + "[2J")


def calculate_last_action(last_head: int, new_head: int, columns: int, rows: int):
    for action in Action:
        potential_position = translate(last_head, action, columns, rows)
        if new_head == potential_position:
            return action

    return None


def center_matrix(b):
    dy, dx = np.where(b[0])
    center_y = (np.arange(0, 7) - 3 + dy[0]) % 7
    center_x = (np.arange(0, 11) - 5 + dx[0]) % 11

    b = b[:, center_y, :]
    b = b[:, :, center_x]

    return b
