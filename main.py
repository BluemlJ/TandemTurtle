import subprocess
import os
import sys
import signal
import _thread
import numpy as np
import tensorflow as tf
from time import sleep

from agent import Agent
import config
import game_play
from game.game import Game
from self_play_training import self_play
import util.nn_interface as nni
from util import logger as lg
from util.xboardInterface import XBoardInterface

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


def main(agent_threads, start_server, server_address, game_id):

    # print(intro_message)
    # np.set_printoptions(suppress=True)

    env = Game(0)

    if config.INITIAL_MODEL_PATH:
        model = nni.load_nn(config.INITIAL_MODEL_PATH)

    # Add to agents if you want to have a random model #
    # print("loading")
    # model_rand = nni.load_nn("")
    # model_rand._make_predict_function()
    # graph = tf.Graph()
    # writer = tf.summary.FileWriter(logdir='logdir',
    #                               graph=tf.get_default_graph())
    # writer.flush()

    #### If we want to learn instead of playing (NOT FINISHED) ####
    if agent_threads == 0:
        new_best_model, version = self_play(env)
        nni.save_nn(f"run/models/{version}", new_best_model)

    #### If the server is running, create clients as threads and connect them to the websocket interface ####
    else:
        if config.SERVER_AUTOSTART:
            if agent_threads == 2:
                os.popen("cp ../tinyChessServer/config.json.sjengVsOur ../tinyChessServer/config.json", 'r', 1)
            if agent_threads == 4:
                os.popen("cp ../tinyChessServer/config.json.ourEngine4times ../tinyChessServer/config.json", 'r', 1)
            server = subprocess.Popen(["node", "index.js"], cwd="../tinyChessServer", stdout=subprocess.PIPE)
            sleep(5)

        for i in range(0, agent_threads):
            name = "TandemTurtle"
            if agent_threads > 1:
                name = "Agent " + str(i)
            _thread.start_new_thread(create_and_run_agent, (name, env, model, "websocket", server_address))

        while True:
            sleep(10)


if __name__ == "__main__":
    agent_threads = config.GAME_AGENT_THREADS
    start_server = config.SERVER_AUTOSTART
    port = config.SERVER_PORT
    position = config.SERVER_WEBSOCKET_POSITION
    game_id = config.GAMEID

    mode = ''
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

    if mode == "auto-4":
        agent_threads = 4
    if mode == "2vsSjeng":
        agent_threads = 2
    if mode == "single_agent":
        agent_threads = 1
    if mode == "selfplay":
        agent_threads = 0

    main(agent_threads, start_server, server_address, game_id)
