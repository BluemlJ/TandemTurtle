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
from util.xboardInterface import XBoardInterface
import tensorflow as tf
from self_play_training import self_play
import util.nn_interface as nni

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
    model = nni.load_nn(config.INITIAL_MODEL_PATH)
    model._make_predict_function()
    print(model.summary())

    writer = tf.summary.FileWriter(logdir='logdir',
                                   graph=tf.get_default_graph())
    writer.flush()

    # If we want to learn instead of playing
    if local_training_mode:
        new_best_model, version = self_play(env)
        nni.save_nn(f"run/models/{version}", new_best_model)

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
