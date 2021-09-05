import argparse
import os
import pickle

from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration

from game_generator.generate_monte_carlo_game import GenerateMonteCarloGame

parser = argparse.ArgumentParser()
parser.add_argument('--work-dir', required=True)
parser.add_argument('--num-geese', type=int, default=4)
parser.add_argument('--num-games', type=int, default=500)
parser.add_argument('--num-rounds', type=int, default=500)
parser.add_argument('--deep', type=int, default=6)
args = parser.parse_args()

configuration = Configuration({"columns": 11,
                               "rows": 7,
                               "hunger_rate": 40,
                               "min_food": 2,
                               "max_length": 99})

NUM_GEESE = args.num_geese
NUM_GAMES = args.num_games
NUM_ROUNDS = args.num_rounds
DEEP = args.deep


def main():
    xs = []
    ys = []

    for i in range(NUM_GAMES):
        print(f"***** Game {i} *****")
        generate_game = GenerateMonteCarloGame(configuration, NUM_GEESE)
        x, y = generate_game.generate_game(NUM_ROUNDS, DEEP)
        xs.append(x)
        ys.append(y)

        if i % 10 == 0:
            save_pickle(xs, ys)

    save_pickle(xs, ys)


def save_pickle(x, y):
    game_states_filename = f"generated_mc_game_states_{NUM_GAMES}_{NUM_ROUNDS}_{DEEP}.pickle"
    actions_filename = f"generated_mc_actions_{NUM_GAMES}_{NUM_ROUNDS}_{DEEP}.pickle"

    with open(os.path.join(args.work_dir, game_states_filename), "wb") as f:
        pickle.dump(x, f)
    with open(os.path.join(args.work_dir, actions_filename), "wb") as f:
        pickle.dump(y, f)


if __name__ == '__main__':
    main()
