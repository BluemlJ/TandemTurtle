"""
This file contains the game rules.
It gives the logic behind moving from one game state to another, given a chosen action. For example, given the intial board and the move g1f3, the "takeAction" method return a new game state, with the move played.
You can replace the game.py file with any game file that conforms to the same API and the algorithm will in principal, learn strategy through self play, based on the rules you have given it.
"""

import numpy as np
import logging

class Game:

	def __init__(self):		
		raise NotImplemented

	def reset(self):
		raise NotImplemented

	def step(self, action):
		raise NotImplemented

class GameState():
	def __init__(self, board, playerTurn):
		raise NotImplemented

	def _allowedActions(self):
		allowed = []
		raise NotImplemented
		return allowed

	def _binary(self):
		raise NotImplemented
		return (position)

	def _convertStateToId(self):
		raise NotImplemented
		return id

	def _checkForEnd(self):
		raise NotImplemented
		return (1 or 0)

	def _getValue(self):
		raise NotImplemented
		return evalOfStateForPlayer

	def takeAction(self, action):
		raise NotImplemented

	def render(self, logger):
		for r in range(8):
			logger.info([self.pieces[str(x)] for x in self.board[r][0:7])
		logger.info('--------------')

