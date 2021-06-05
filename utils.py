import platform
import subprocess

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
