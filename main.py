import numpy as np
np.set_printoptions(suppress=True)

from game import Game, GameState
import funcs

import logger as lg

import config
from simple_agent import Simple_Agent

from time import sleep
import _thread

# TODO created twice why not give as parameter
env = Game(0)

import os

# try to find out if server is running
response = os.system("nc -vz localhost 80")
SERVER_IS_RUNNING = response == 0

def create_and_run_websocket_agent(name, isStarting):
    interface = XBoardInterface(name)

    agent1 = Simple_Agent(name, env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, None, interface)
    
    while not interface.gameStarted:
        sleep(0.1)

    funcs.play_websocket_game(agent1, lg.logger_main, interface, turns_until_tau0=config.TURNS_UNTIL_TAU0, goes_first = isStarting)


#### If the server is running, create 4 clients as threads and connect them to the websocket interface ####
if SERVER_IS_RUNNING:
    from websocketInterface import XBoardInterface


    _thread.start_new_thread ( create_and_run_websocket_agent, ("Agent 1", True))
    _thread.start_new_thread ( create_and_run_websocket_agent, ("Agent 2", False))
    _thread.start_new_thread ( create_and_run_websocket_agent, ("Agent 3", True))
    _thread.start_new_thread ( create_and_run_websocket_agent, ("Agent 4", False))

    while True:
        sleep(10)

else:
    from commandlineInterface import XBoardInterface

    interface = XBoardInterface("Agent 1")

    agent1 = Simple_Agent('agent1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, None, interface)
    agent2 = Simple_Agent('agent2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, None, XBoardInterface("Agent 2"))
    agent3 = Simple_Agent('agent3', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, None, XBoardInterface("Agent 3"))
    agent4 = Simple_Agent('agent4', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, None, XBoardInterface("Agent 4"))
    agents = [agent1, agent2, agent3, agent4]


    while not interface.gameStarted:
        sleep(0.1)

    funcs.play_matches(agents, config.EPISODES, lg.logger_main,
                    turns_until_tau0=config.TURNS_UNTIL_TAU0)

    exit()
