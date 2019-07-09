from shutil import copyfile
import subprocess
import os
import signal
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
import threading
import sys

# intro_message =\
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


def create_and_run_agent(name, env, model, interfaceType="websocket", server_address="ws://localhost:8080/websocketclient"):
    interface = XBoardInterface(name, interfaceType, server_address)
    agent1 = Agent(name, env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, model, interface)

    while not interface.gameStarted:
        sleep(0.1)

    game_play.play_websocket_game(agent1, lg.logger_main, interface, turns_with_high_noise=config.TURNS_WITH_HIGH_NOISE)


def main(mode='auto-4', start_server=1, server_address="ws://localhost:8080/websocketclient", game_id=""):
    # graph = tf.Graph()

    # print(intro_message)
    # np.set_printoptions(suppress=True)
     # TODO created twice why not give as parameter
    env = Game(0)

    # try to find out if server is running

    if config.INITIAL_MODEL_PATH:
        model = nni.load_nn(config.INITIAL_MODEL_PATH)

    # Add to agents if you want to have a random model
    # print("loading")
    # model_rand = nni.load_nn("")
    # model_rand._make_predict_function()

    # writer = tf.summary.FileWriter(logdir='logdir',
    #                               graph=tf.get_default_graph())
    # writer.flush()

    # If we want to learn instead of playing (NOT FINISHED)
    if mode == "selfplay":
        new_best_model, version = self_play(env)
        nni.save_nn(f"run/models/{version}", new_best_model)

    #### If the server is running, create 4 clients as threads and connect them to the websocket interface ####
    elif mode == "auto-4":

        if start_server:
            os.popen("cp ../tinyChessServer/config.json.ourEngine4times ../tinyChessServer/config.json", 'r', 1)
            server = subprocess.Popen(["node", "index.js"], cwd="../tinyChessServer", stdout=subprocess.PIPE)
            sleep(5)

        _thread.start_new_thread(create_and_run_agent, ("Agent 1", env, model, "websocket", server_address))
        _thread.start_new_thread(create_and_run_agent, ("Agent 2", env, model, "websocket", server_address))
        _thread.start_new_thread(create_and_run_agent, ("Agent 3", env, model, "websocket", server_address))
        _thread.start_new_thread(create_and_run_agent, ("Agent 4", env, model, "websocket", server_address))

        while True:
            sleep(10)

    elif mode == "2vsSjeng":

        if start_server:
            os.popen("cp ../tinyChessServer/config.json.sjengVsOur ../tinyChessServer/config.json", 'r', 1)
            server = subprocess.Popen(["node", "index.js"], cwd="../tinyChessServer", stdout=subprocess.PIPE)
            sleep(5)

        _thread.start_new_thread(create_and_run_agent, ("Agent 1", env, model, "websocket"))
        _thread.start_new_thread(create_and_run_agent, ("Agent 2", env, model, "websocket"))

        while True:
            sleep(10)

    elif mode == "single_agent":

        _thread.start_new_thread(create_and_run_agent, ("TandemTurtle", env, model, "websocket", server_address))
        while True:
            sleep(10)


if __name__ == "__main__":
    mode = "auto-4"
    start_server = 1
    port = "8080"
    position = ""
    game_id = "gameid"

    if len(sys.argv) == 4:
        mode = str(sys.argv[1])
        start_server = int(sys.argv[2])
        port = str(sys.argv[3])
    if len(sys.argv) == 5:
        mode = str(sys.argv[1])
        start_server = int(sys.argv[2])
        port = str(sys.argv[3])
        game_id = str(sys.argv[4])
    if len(sys.argv) == 6:
        position = f"?{str(sys.argv[5])}="

    server_address = f"ws://localhost:{port}/websocketclient{position}"
    main(mode, start_server, server_address)
