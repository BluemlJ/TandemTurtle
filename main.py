from shutil import copyfile
import os
import _thread
from time import sleep
from agent import Agent
import config
from util import logger as lg
import funcs
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

    funcs.play_websocket_game(agent1, lg.logger_main, interface, turns_until_tau0=config.TURNS_WITH_HIGH_NOISE, goes_first=isStarting)


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
        best_model = load_nn(config.INITIAL_MODEL_PATH)
        new_model = load_nn(config.INITIAL_MODEL_PATH)

        # copy the config file to the run folder
        copyfile('./config.py', run_folder + 'config.py')

        # if you want to plot our nn model
        plot_model(model, to_file=run_folder + 'models/model.png', show_shapes=True)

        # create players (TODO fill this step)
        best_player1 = Agent("best_player1", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_model)
        best_player2 = Agent("best_player2", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_model)
        new_player1 = Agent("new_player1", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, new_model)
        new_player2 = Agent("new_player2", env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, new_model)

        # self-play (TODO fill this step)

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
