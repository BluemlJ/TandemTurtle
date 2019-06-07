import os
import _thread
from time import sleep
from simple_agent import Simple_Agent
import config
import logger as lg
import funcs
from game import Game, GameState
import numpy as np

from xboardInterface import XBoardInterface

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
print(intro_message)
np.set_printoptions(suppress=True)


# TODO created twice why not give as parameter
env = Game(0)


# try to find out if server is running
response = os.system("nc -vz localhost 80")
SERVER_IS_RUNNING = response == 0


def create_and_run_agent(name, isStarting, interfaceType="websocket"):
    interface = XBoardInterface(name, interfaceType)

    agent1 = Simple_Agent(name, env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, None, interface)

    while not interface.gameStarted:
        sleep(0.1)

    funcs.play_websocket_game(agent1, lg.logger_main, interface, turns_until_tau0=config.TURNS_UNTIL_TAU0, goes_first=isStarting)


#### If the server is running, create 4 clients as threads and connect them to the websocket interface ####
if SERVER_IS_RUNNING:
    _thread.start_new_thread(create_and_run_agent, ("Agent 1", True))
    _thread.start_new_thread(create_and_run_agent, ("Agent 2", False))
    _thread.start_new_thread(create_and_run_agent, ("Agent 3", True))
    _thread.start_new_thread(create_and_run_agent, ("Agent 4", False))

    while True:
        sleep(10)

else:
    _thread.start_new_thread(create_and_run_agent, ("Agent 1", False, "commandlineInterface"))

    while True:
        sleep(10)