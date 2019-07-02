"""
This contains the code that starts the learning process. It loads the game rules and then iterates through the main loop of the algorithm, which consist of three stages:

    Self-play
    Retraining the Neural Network
    Evaluating the Neural Network

There are two agents involved in this loop, the best_player and the current_player.

The best_player contains the best performing neural network and is used to generate the self play memories.
The current_player then retrains its neural network on these memories and is then pitched against the best_player.
If it wins, the neural network inside the best_player is switched for the neural network inside the current_player, and the loop starts again.
"""

import config
import util.logger as lg
from shutil import copyfile
from agent import Agent
from keras.utils import plot_model
import pickle
from config import run_folder, run_archive_folder
from util.memory import Memory
from importlib import reload
import random
from util.nn_interface import load_nn


def initialize_run(env):
    if config.INITIAL_RUN_NUMBER is not None:
        copyfile(config.run_archive_folder + env.name + '/run' + str(config.INITIAL_RUN_NUMBER).zfill(4) + '/config.py',
                 './config.py')


def intialize_memory(env):
    if config.INITIAL_MEMORY_VERSION is None:
        memory = Memory(config.MEMORY_SIZE)
    else:
        print('LOADING MEMORY VERSION ' + str(config.INITIAL_MEMORY_VERSION) + '...')
        memory = pickle.load(open(run_archive_folder + env.name + '/run' + str(config.INITIAL_RUN_NUMBER).zfill(
            4) + "/memory/memory" + str(config.INITIAL_MEMORY_VERSION).zfill(4) + ".p", "rb"))
    return memory


def initialize_neural_network(plot=False):
    if config.INITIAL_MODEL_VERSION is not None:
        best_model = load_nn(config.INITIAL_MODEL_PATH)
        new_model = load_nn(config.INITIAL_MODEL_PATH)
        best_player_version = config.INITIAL_MODEL_VERSION
    else:
        best_model = load_nn(config.PRETRAINED_MODEL_PATH)
        new_model = load_nn(config.PRETRAINED_MODEL_PATH)
        best_player_version = 0

    if plot:
        plot_model(best_model, to_file=run_folder + 'models/model.png', show_shapes=True)

    return best_model, new_model, best_player_version


def initialize_player(env, best_model, new_model):
    best_player = Agent("best_player", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_model)
    new_player = Agent("new_player", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, new_model)
    return best_player, new_player


def self_play(env, max_iteration=2500):
    initialize_run(env)
    memory = intialize_memory(env)
    best_model, new_model, best_player_version = initialize_neural_network()
    best_player, new_player = initialize_player(env, best_model, new_model)
    iteration = 0

    while iteration is not max_iteration:

        iteration += 1
        reload(lg)
        reload(config)

        print('ITERATION NUMBER ' + str(iteration))

        lg.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)
        print('BEST PLAYER VERSION ' + str(best_player_version))

        print('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
        _, memory = playMatches(best_model, best_model, config.EPISODES, lg.logger_main,
                                      turns_until_high_noise=config.TURNS_WITH_HIGH_NOISE, memory=memory)
        print('\n')

        memory.clear_stmemory()

        if len(memory.ltmemory) >= config.MEMORY_SIZE:

            print('RETRAINING...')
            new_player.replay(memory.ltmemory)
            print('')

            if iteration % 5 == 0:
                pickle.dump(memory, open(run_folder + "memory/memory" + str(iteration).zfill(4) + ".p", "wb"))

            lg.logger_memory.info('====================')
            lg.logger_memory.info('NEW MEMORIES')
            lg.logger_memory.info('====================')

            memory_samp = random.sample(memory.ltmemory, min(1000, len(memory.ltmemory)))

            for s in memory_samp:
                current_value, current_probs, _ = new_player.get_preds(s['state'])
                best_value, best_probs, _ = best_player.get_preds(s['state'])

                lg.logger_memory.info('MCTS VALUE FOR %s: %f', s['playerTurn'], s['value'])
                lg.logger_memory.info('CUR PRED VALUE FOR %s: %f', s['playerTurn'], current_value)
                lg.logger_memory.info('BES PRED VALUE FOR %s: %f', s['playerTurn'], best_value)
                lg.logger_memory.info('THE MCTS ACTION VALUES: %s', ['%.2f' % elem for elem in s['AV']])
                lg.logger_memory.info('CUR PRED ACTION VALUES: %s', ['%.2f' % elem for elem in current_probs])
                lg.logger_memory.info('BES PRED ACTION VALUES: %s', ['%.2f' % elem for elem in best_probs])
                lg.logger_memory.info('ID: %s', s['state'].id)
                lg.logger_memory.info('INPUT TO MODEL: %s', new_player.model.convertToModelInput(s['state']))

                s['state'].render(lg.logger_memory)

            print('TOURNAMENT...')
            scores, _ = playMatches(best_model, new_model, config.EVAL_EPISODES,
                                                       lg.logger_tourney, turns_until_high_noise=0, memory=None)
            print('\nSCORES')
            print(scores)

            print('\n\n')

            if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
                best_player_version += 1
                best_model.model.set_weights(new_model.model.get_weights())

        else:
            print('MEMORY SIZE: ' + str(len(memory.ltmemory)))

    return best_model, best_player_version
