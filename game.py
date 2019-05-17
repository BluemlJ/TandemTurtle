"""
This file contains the game rules.
It gives the logic behind moving from one game state to another, given a chosen action. For example, given the intial board and the move g1f3, the "takeAction" method return a new game state, with the move played.
You can replace the game.py file with any game file that conforms to the same API and the algorithm will in principal, learn strategy through self play, based on the rules you have given it.
"""
import chess
import numpy as np
import logging
from chess.variant import BughouseBoards, SingleBughouseBoard


# board_number  0 for left 1 for right board
class Game:
    def __init__(self, board_number):
        self.board_number = board_number
        self.currentPlayer = 1
        boards = BughouseBoards()
        self.gameState = GameState(boards, self.board_number, self.currentPlayer)
        self.actionSpace = np.zeros(135)

        self.name = 'bughouse'

        self.action_size = len(self.actionSpace)
        """
        TODO
        self.state_size = len(self.gameState.binary)
        self.pieces = {'1':'X', '0': '-', '-1':'O'}
		self.grid_shape = (6,7)
		self.input_shape = (2,6,7)
        """

    def reset(self):
        self.currentPlayer = 1
        boards = BughouseBoards()
        self.gameState = GameState(boards, self.board_number, self.currentPlayer)
        return self.gameState

    def step(self, action):
        next_state, value, done = self.gameState.takeAction(action)
        self.gameState = next_state
        self.currentPlayer = -self.currentPlayer
        info = None
        return ((next_state, value, done, info))


class GameState():
    def __init__(self, boards, board_number, playerTurn):
        self.boards = boards
        self.board_number = board_number
        self.board = boards.boards[board_number]
        if board_number == 1:
            self.partner_board = boards.boards[0]
        else:
            self.partner_board = boards.boards[1]

        self.playerTurn = playerTurn  # 1 = white -1 = black

        self.binary = self._binary()
        self.id = self._convert_state_to_id()
        self.allowedActions = self._allowed_actions()
        self.isEndGame = self._check_for_end()
        self.value = self._get_value()
        self.score = self._get_score()

    def _allowed_actions(self):
        allowed = [move_as_array(m) for m in list(self.board.legal_moves)]
        return allowed

    def _binary(self):
        raise NotImplementedError
        return (position)

    def _convert_state_to_id(self):
        s = self.boards.__str__()
        id = "".join(s.split())
        return id

    def _check_for_end(self):
        if self.boards.is_game_over():
            return 1
        return 0

    def _get_value(self):
        # return (state, currentPlayerPoints, opponentPlayerPoints)
        result = self.boards.result()
        if result == "1/2-1/2":
            return (0, 0.5, 0.5)
        elif result == "0-1":
            if self.playerTurn == 1:
                return (-1, -1, 1)
            else:
                return (1, 1, -1)
        else:
            if self.playerTurn == -1:
                return (-1, -1, 1)
            else:
                return (1, 1, -1)

        return (0, 0, 0)

    def take_action(self, action):
        move = array_as_move(action)
        new_boards = copy.deepcopy(self.boards)
        new_boards.boards[self.board_number].push(move)

        newState = GameState(new_boards, self.board_number, -self.playerTurn)

        value = 0
        done = 0

        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return (newState, value, done)

    def render(self, logger):
        logger.info(self.boards.__str__())
        logger.info('--------------')


# colour necessary for drops
def move_as_array(move, color=chess.WHITE):
    move_array = np.zeros(
        128 + 6 + 1)  # first 64 from square second 64 to_ square + 6 for the drop piece + 1 for colour
    move_array[move.from_square] = 1
    move_array[64 + move.to_square] = 1
    if move.drop:
        move_array[127 + move.drop] = 1

    if color:
        move_array[-1] = 1

    return move_array


def array_as_move(action):
    fromSquare = np.argmax(action[0:64])
    toSquare = np.argmax(action[64:128])
    if np.max(action[128:]) > 0:  # move is a drop
        color = action[-1]
        piece = chess.Piece(np.argmax(action[128:-1]) + 1, color)
        return chess.Move.from_uci(piece.symbol() + "@" + chess.SQUARE_NAMES[toSquare])
    else:
        return chess.Move(fromSquare, toSquare, None)


def board_to_array(board):
    array = np.zeros((8, 8, 6, 2))#??
    raise NotImplementedError


def pocket_to_array(pocket):
    d = dict(pocket.pieces)
    array = np.zeros(6)
    for piece, count in d.items():
        array[piece - 1] = count
    return array
