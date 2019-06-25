from shutil import copyfile
import os
import _thread
from time import sleep
from agent import Agent
import config
from util import logger as lg
import game_play
from game.game import Game
import numpy as np
from pretraining.nn import NeuralNetwork
from util.xboardInterface import XBoardInterface
from keras.models import load_model
from keras.utils import plot_model
import pickle
from config import run_folder, run_archive_folder
from util.memory import Memory
import tensorflow as tf
from importlib import reload
import random

intro_message =\
"""

                        __   __
           .,-;-;-,.   /'_\ /_'\ .,-;-;-,.
          _/_/_/_|_\_\) /    \ (/_/_|_\_\_\_
       '-<_><_><_><_>=/\     /\=<_><_><_><_>-'
         `/_/====/_/-'\_\   /_/'-\_\====\_\'

  _____             _             _____         _   _     
 |_   _|_ _ _ _  __| |___ _ __   |_   _|  _ _ _| |_| |___ 
   | |/ _` | ' \/ _` / -_) '  \    | || || | '_|  _| / -_)
   |_|\__,_|_||_\__,_\___|_|_|_|   |_| \_,_|_|  \__|_\___|

Jannis Blüml, Maximilian Otte, Florian Netzer, Magdalena Wache, Lars Wolf
License GPL v3.0                                                      
ASCII-Art: Joan Stark              
"""


def load_nn(path_to_nn=""):
    # Load pre trained model if path exists
    if path_to_nn == "":
        print("Path empty, loading clean nn")
        nn = NeuralNetwork()
        model = nn.model
    else:
        path = os.getcwd()
        print("Load nn from ", path + path_to_nn)
        nn = NeuralNetwork()
        nn.model = load_model(path + path_to_nn)
        model = nn.model
    return model


def create_and_run_agent(name, isStarting, env, model, interfaceType="websocket"):
    interface = XBoardInterface(name, interfaceType)

    agent1 = Agent(name, env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, model, interface)

    while not interface.gameStarted:
        sleep(0.1)

    game_play.play_websocket_game(agent1, lg.logger_main, interface, turns_with_high_noise=config.TURNS_WITH_HIGH_NOISE, goes_first=isStarting)


def main():
    # graph = tf.Graph()

    print(intro_message)
    np.set_printoptions(suppress=True)

    # TODO created twice why not give as parameter
    env = Game(0)

    # try to find out if server is running
    response = os.system("nc -vz localhost 80")
    SERVER_IS_RUNNING = response == 0

    # if selfplay
    local_training_mode = 0
    # with graph.as_default():
    model = load_nn(config.INITIAL_MODEL_PATH)
    model._make_predict_function()

    print(model.summary())

    writer = tf.summary.FileWriter(logdir='logdir',
                                   graph=tf.get_default_graph())
    writer.flush()

    # If we want to learn instead of playing
    if local_training_mode:

        if config.INITIAL_RUN_NUMBER != None:
            copyfile(
                config.run_archive_folder + env.name + '/run' + str(config.INITIAL_RUN_NUMBER).zfill(4) + '/config.py',
                './config.py')

            # next step is to load memory
        if config.INITIAL_MEMORY_VERSION == None:
            memory = Memory(config.MEMORY_SIZE)
        else:
            print('LOADING MEMORY VERSION ' + str(config.INITIAL_MEMORY_VERSION) + '...')
            memory = pickle.load(open(run_archive_folder + env.name + '/run' + str(config.INITIAL_RUN_NUMBER).zfill(
                4) + "/memory/memory" + str(config.INITIAL_MEMORY_VERSION).zfill(4) + ".p", "rb"))

        # LOAD NN (TODO fill this step)

        if config.INITIAL_MODEL_VERSION != None:
            best_model = load_nn(config.INITIAL_MODEL_PATH)
            new_model = load_nn(config.INITIAL_MODEL_PATH)
            best_player_version = config.INITIAL_MODEL_VERSION
        else:
            best_model = load_nn(config.PRETRAINED_MODEL_PATH)
            new_model = load_nn(config.PRETRAINED_MODEL_PATH)
            best_player_version = 0

        # copy the config file to the run folder
        copyfile('./config.py', run_folder + 'config.py')

        # if you want to plot our nn model
        plot_model(model, to_file=run_folder + 'models/model.png', show_shapes=True)

        # create players (TODO fill this step)
        best_player = Agent("best_player", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_model)
        new_player = Agent("new_player", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, new_model)

        # self-play (TODO fill this step)

        iteration = 0

        while 1:

            iteration += 1
            reload(lg)
            reload(config)

            print('ITERATION NUMBER ' + str(iteration))

            lg.logger_main.info('BEST PLAYER VERSION: %d', best_player_version)
            print('BEST PLAYER VERSION ' + str(best_player_version))

            ######## SELF PLAY ########
            print('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
            _, memory, _, _ = playMatches(best_player, best_player, config.EPISODES, lg.logger_main,
                                          turns_until_tau0=config.TURNS_UNTIL_TAU0, memory=memory)
            print('\n')

            memory.clear_stmemory()

            if len(memory.ltmemory) >= config.MEMORY_SIZE:

                ######## RETRAINING ########
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

                ######## TOURNAMENT ########
                print('TOURNAMENT...')
                scores, _, points, sp_scores = playMatches(best_player, new_player, config.EVAL_EPISODES,
                                                           lg.logger_tourney, turns_until_tau0=0, memory=None)
                print('\nSCORES')
                print(scores)
                print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
                print(sp_scores)
                # print(points)

                print('\n\n')

                if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
                    best_player_version = best_player_version + 1
                    best_model.model.set_weights(new_model.model.get_weights())
                    best_model.write(env.name, best_player_version)

            else:
                print('MEMORY SIZE: ' + str(len(memory.ltmemory)))

    #### If the server is running, create 4 clients as threads and connect them to the websocket interface ####
    elif SERVER_IS_RUNNING:
        _thread.start_new_thread(create_and_run_agent, ("Agent 1", True, env, model))
        _thread.start_new_thread(create_and_run_agent, ("Agent 2", False, env, model))
        _thread.start_new_thread(create_and_run_agent, ("Agent 3", True, env, model))
        _thread.start_new_thread(create_and_run_agent, ("Agent 4", False, env, model))

        while True:
            sleep(10)

    else:
        _thread.start_new_thread(create_and_run_agent, ("Agent 1", False, env, model, "commandlineInterface"))

        while True:
            sleep(10)


if __name__ == "__main__":
    main()
