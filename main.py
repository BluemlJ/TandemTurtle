from shutil import copyfile
import os
import _thread
from time import sleep
from simple_agent import Simple_Agent
import config
from util import logger as lg
import funcs
from game.game import Game
import numpy as np
from pretraining.nn import NeuralNetwork
from util.xboardInterface import XBoardInterface
from keras.models import load_model

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

    agent1 = Simple_Agent(name, env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, model, interface)

    while not interface.gameStarted:
        sleep(0.1)

    funcs.play_websocket_game(agent1, lg.logger_main, interface, turns_until_tau0=config.TURNS_UNTIL_TAU0, goes_first=isStarting)


def main():
    print(intro_message)
    np.set_printoptions(suppress=True)

    # TODO created twice why not give as parameter
    env = Game(0)

    # try to find out if server is running
    response = os.system("nc -vz localhost 80")
    SERVER_IS_RUNNING = response == 0

    # if selfplay
    local_training_mode = 0

    model = load_nn(config.INITIAL_MODEL_PATH)
    model._make_predict_function()

    # If we want to learn instead of playing
    if local_training_mode:

        # If loading an existing neural network, copy the config file to root
        if config.INITIAL_RUN_NUMBER != None:
            copyfile(config.run_archive_folder + env.name + '/run' + str(config.INITIAL_RUN_NUMBER).zfill(4) + '/config.py',
                     './config.py')

        # next step is to load memory (TODO fill this step)
        # LOAD NN (TODO fill this step)
        model = load_nn(config.INITIAL_MODEL_PATH)
        # copy the config file to the run folder
        # create players (TODO fill this step)
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
