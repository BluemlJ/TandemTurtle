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

import pickle
import config
from config import run_folder, run_archive_folder
import util.logger as lg
from keras.utils import plot_model
from importlib import reload
import random
from shutil import copyfile
from util.memory import Memory
import numpy as np
from agent import Agent


def train():
    agent = Agent(current_player, state_size, action_size, config.MCTS_SIMS, config.CPUCT, model, interface)
    self_play(agent, config.NR_OF_GAMES_SELF_PLAY)


def self_play(agent, nr_of_games):
    raise NotImplemented


def retrain():
    raise NotImplemented


def evaluate():
    raise NotImplemented
