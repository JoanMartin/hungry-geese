import pickle

from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration

from game_generator.generate_monte_carlo_game import GenerateMonteCarloGame


def main():
    configuration = Configuration({"columns": 11,
                                   "rows": 7,
                                   "hunger_rate": 40,
                                   "min_food": 2,
                                   "max_length": 99})
    xs = []
    ys = []

    num_games = 100
    num_geese = 4
    num_rounds = 500
    deep = 6
    for i in range(num_games):
        print(f"***** Game {i} *****")
        generate_game = GenerateMonteCarloGame(configuration, num_geese)
        x, y = generate_game.generate_game(num_rounds, deep)
        xs.append(x)
        ys.append(y)

        if i % 10 == 0:
            save_pickle(xs, ys)

    save_pickle(xs, ys)


def save_pickle(x, y):
    with open(r"../data/generated_mc_game_states.pickle", "wb") as f:
        pickle.dump(x, f)
    with open(r"../data/generated_mc_actions.pickle", "wb") as f:
        pickle.dump(y, f)


if __name__ == '__main__':
    main()
