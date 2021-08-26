import argparse
import datetime
import multiprocessing
import os
import random
import shutil
import tempfile
import time

import h5py
import numpy as np
from kaggle_environments.envs.hungry_geese.hungry_geese import Configuration

import kerasutil
from neural_network_train.rl.ac_agent import load_ac_agent
from neural_network_train.rl.experience import ExperienceCollector, combine_experience, load_experience
from neural_network_train.rl.simulate_game import SimulateGame


def load_agent(filename):
    with h5py.File(filename, 'r') as h5file:
        return load_ac_agent(h5file)


def get_temp_file():
    fd, fname = tempfile.mkstemp(prefix='dlgo-train-')
    os.close(fd)
    return fname


configuration = Configuration({"columns": 11,
                               "rows": 7,
                               "hunger_rate": 40,
                               "min_food": 2,
                               "max_length": 99})

simulate_game = SimulateGame(configuration)


def do_self_play(agent_filename, num_games, experience_filename, gpu_frac):
    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agents = [load_agent(agent_filename)] * 4
    collectors = [ExperienceCollector()] * 4

    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        for idx in range(4):
            collectors[idx].begin_episode()
            agents[idx].set_collector(collectors[idx])

        game_state = simulate_game.simulate(agents)
        winner_index = max(game_state.geese, key=lambda x: x.reward).index

        for index, goose in enumerate(game_state.geese):
            if goose.reward <= 0:
                collectors[index].complete_episode(reward=-1)
            elif goose.reward > 0:
                if goose.index == winner_index:
                    collectors[index].complete_episode(reward=1)
                else:
                    collectors[index].complete_episode(reward=0)

    experience = combine_experience(collectors)
    print('Saving experience buffer to %s\n' % experience_filename)
    with h5py.File(experience_filename, 'w') as experience_out_f:
        experience.serialize(experience_out_f)


def generate_experience(learning_agent, exp_file, num_games, num_workers):
    experience_files = []
    workers = []
    gpu_frac = 0.95 / float(num_workers)
    games_per_worker = num_games // num_workers
    for i in range(num_workers):
        filename = get_temp_file()
        experience_files.append(filename)
        worker = multiprocessing.Process(
            target=do_self_play,
            args=(
                learning_agent,
                games_per_worker,
                filename,
                gpu_frac
            )
        )
        worker.start()
        workers.append(worker)

    # Wait for all workers to finish.
    print(f'Waiting for {len(workers)} workers...')
    for worker in workers:
        worker.join()

    # Merge experience buffers.
    print('Merging experience buffers...')
    first_filename = experience_files[0]
    other_filenames = experience_files[1:]
    with h5py.File(first_filename, 'r') as exp_f:
        combined_buffer = load_experience(exp_f)
    for filename in other_filenames:
        with h5py.File(filename, 'r') as exp_f:
            next_buffer = load_experience(exp_f)
        combined_buffer = combine_experience([combined_buffer, next_buffer])
    print('Saving into %s...' % exp_file)
    with h5py.File(exp_file, 'w') as experience_outf:
        combined_buffer.serialize(experience_outf)

    # Clean up.
    for fname in experience_files:
        os.unlink(fname)


def train_worker(learning_agent, output_file, experience_file, lr, batch_size):
    learning_agent = load_agent(learning_agent)
    with h5py.File(experience_file, 'r') as expf:
        exp_buffer = load_experience(expf)
    learning_agent.train(exp_buffer, lr=lr, batch_size=batch_size)

    with h5py.File(output_file, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


def train_on_experience(learning_agent, output_file, experience_file, lr, batch_size):
    # Do the training in the background process. Otherwise some Keras
    # stuff gets initialized in the parent, and later that forks, and
    # that messes with the workers.
    worker = multiprocessing.Process(
        target=train_worker,
        args=(
            learning_agent,
            output_file,
            experience_file,
            lr,
            batch_size
        )
    )
    worker.start()
    worker.join()


def play_games(args):
    learning_agent, reference_agent, num_games, gpu_frac = args

    kerasutil.set_gpu_memory_target(gpu_frac)

    random.seed(int(time.time()) + os.getpid())
    np.random.seed(int(time.time()) + os.getpid())

    agents = [load_agent(learning_agent),
              load_agent(reference_agent),
              load_agent(reference_agent),
              load_agent(reference_agent)]

    wins, losses = 0, 0
    for i in range(num_games):
        print('Simulating game %d/%d...' % (i + 1, num_games))
        game_state = simulate_game.simulate(agents)
        winner_index = max(game_state.geese, key=lambda x: x.reward).index
        print(f'Agent {winner_index} wins')

        if winner_index == 0:
            wins += 1
        else:
            losses += 1

        print('Agent record: %d/%d' % (wins, wins + losses))

    return wins, losses


def evaluate(learning_agent, reference_agent, num_games, num_workers):
    games_per_worker = num_games // num_workers
    gpu_frac = 0.95 / float(num_workers)
    pool = multiprocessing.Pool(num_workers)
    worker_args = [(learning_agent, reference_agent, games_per_worker, gpu_frac) for _ in range(num_workers)]
    game_results = pool.map(play_games, worker_args)

    total_wins, total_losses = 0, 0
    for wins, losses in game_results:
        total_wins += wins
        total_losses += losses
    print('FINAL RESULTS:')
    print('Learner: %d' % total_wins)
    print('Reference: %d' % total_losses)
    pool.close()
    pool.join()
    return total_wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=True)
    parser.add_argument('--games-per-batch', '-g', type=int, default=1000)
    parser.add_argument('--work-dir', '-d')
    parser.add_argument('--num-workers', '-w', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('--log-file', '-l')

    args = parser.parse_args()

    log_f = open(args.log_file, 'a')
    log_f.write('----------------------\n')
    log_f.write(f'Starting from {args.agent} at {datetime.datetime.now()}\n')

    learning_agent = args.agent
    reference_agent = args.agent
    experience_file = os.path.join(args.work_dir, 'exp_temp.hdf5')
    tmp_agent = os.path.join(args.work_dir, 'agent_temp.hdf5')
    working_agent = os.path.join(args.work_dir, 'agent_cur.hdf5')

    total_games = 0
    while True:
        print('Reference: %s' % (reference_agent,))
        log_f.write('Total games so far %d\n' % (total_games,))

        generate_experience(learning_agent,
                            experience_file,
                            num_games=args.games_per_batch,
                            num_workers=args.num_workers)

        train_on_experience(learning_agent,
                            tmp_agent,
                            experience_file,
                            lr=args.lr,
                            batch_size=args.bs)

        total_games += args.games_per_batch

        wins = evaluate(learning_agent,
                        reference_agent,
                        num_games=480,
                        num_workers=args.num_workers)

        print('Won %d / 480 games (%.3f)' % (wins, float(wins) / 480.0))
        log_f.write('Won %d / 480 games (%.3f)\n' % (wins, float(wins) / 480.0))
        shutil.copy(tmp_agent, working_agent)
        learning_agent = working_agent

        if wins >= 262:
            next_filename = os.path.join(args.work_dir, 'agent_%08d.hdf5' % (total_games,))
            shutil.move(tmp_agent, next_filename)
            reference_agent = next_filename
            log_f.write('New reference is %s\n' % next_filename)
        else:
            print('Keep learning\n')

        log_f.flush()


if __name__ == '__main__':
    main()
