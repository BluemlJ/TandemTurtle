"""
An instance of the Memory class stores the memories of previous games, that the algorithm uses to retrain the neural network of the current_player.
"""

import numpy as np
from collections import deque

import config


class Memory:
    def __init__(self, MEMORY_SIZE):
        self.MEMORY_SIZE = config.MEMORY_SIZE
        self.ltmemory = deque(maxlen=config.MEMORY_SIZE)
        self.stmemory = deque(maxlen=config.MEMORY_SIZE)

    def commit_stmemory(self, env, state):
        # TODO commit the env with board and mcts to the memory
        pass

    def commit_ltmemory(self):
        for i in self.stmemory:
            self.ltmemory.append(i)
        self.clear_stmemory()

    def clear_stmemory(self):
        self.stmemory = deque(maxlen=config.MEMORY_SIZE)
