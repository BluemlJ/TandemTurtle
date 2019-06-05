"""
This file contains the game rules.
It gives the logic behind moving from one game state to another, given a chosen action. For example, given the intial board and the move g1f3, the "takeAction" method return a new game state, with the move played.
You can replace the game.py file with any game file that conforms to the same API and the algorithm will in principal, learn strategy through self play, based on the rules you have given it.
"""
import copy

import chess
import numpy as np
import logging
from chess.variant import BughouseBoards, SingleBughouseBoard
from input_representation import board_to_planes


# board_number  0 for left 1 for right board
class Game:
    def __init__(self, board_number):
        """

        :param board_number: 0 for left 1 for right board
        """
        self.board_number = board_number
        self.currentPlayer = 1
        boards = BughouseBoards()
        self.gameState = GameState(boards, self.board_number, self.currentPlayer)
        #self.actionSpace = np.zeros(135)

        self.name = 'bughouse'

        self.action_size = 64*64*6*2
        # Todo not hard coded
        # self.state_size = len(self.gameState.binary)
        self.state_size = 4352

        """
        TODO Do we need this stuff?
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
        next_state, value, done = self.gameState.take_action(action)
        self.gameState = next_state
        self.currentPlayer = -self.currentPlayer
        info = None
        return ((next_state, value, done, info))


class GameState:
    """
    The gamestate consists out of both Bughouse Boards.
    The board number decides on which board the engine is playing
    Moves/actions are passed as numpy arrays.

    """

    def __init__(self, boards, board_number, player_turn):
        """
        :param boards: BughouseBoards
        :param board_number: 0 for left 1 for right board
        :param player_turn: 1 white -1 black
        """
        self.boards = boards
        self.board_number = board_number

        self.board = boards.boards[board_number]

        if board_number == 1:
            self.partner_board = boards.boards[0]
        else:
            self.partner_board = boards.boards[1]

        self.playerTurn = player_turn

        # self.binary = self._binary()
        self.id = self._convert_state_to_id()
        self.allowedActions = self._allowed_actions()
        self.isEndGame = self._check_for_end()
        self.value = self._get_value()

    def _allowed_actions(self):
        allowed = list(self.board.legal_moves)
        return allowed

    def _binary(self):
        """
        :return: The game state as a binary numpy array including both boards and pockets
        """
        b1 = board_to_planes(self.board).flatten()
        b2 = board_to_planes(self.partner_board).flatten()

        return np.concatenate([b1,b2])

    def _convert_state_to_id(self):
        s = self.boards.__str__()
        return "".join(s.split()) + str(self.playerTurn)

    def _check_for_end(self):
        if self.boards.is_game_over():
            return 1
        return 0

    def _get_value(self):
        """
        :return: (state, currentPlayerPoints, opponentPlayerPoints)
        """
        result = self.boards.result()
        if result == "1/2-1/2":
            return (0, 0.5, 0.5)
        elif result == "0-1":
            if self.playerTurn == 1:
                return (-1, -1, 1)
            else:
                return (1, 1, -1)
        elif result == "1-0":
            if self.playerTurn == -1:
                return (-1, -1, 1)
            else:
                return (1, 1, -1)

        return (0, 0, 0)

    def check_if_legal(self, action):
        """
        Check if move at current game state is correct and for current player playable
        :param action: action as chess move
        :return: True if legal, raise Exception otherwise
        """
        is_legal_move = np.any([(str(action) == str(el)) for el in self._allowed_actions()])
        if not is_legal_move:
            # TODO make Exception as concrete as possible, maybe own class
            print("allowed actions ", self._allowed_actions())
            print("action itself: ", action)
            raise Exception(f"Illegal Move: {action}  Legal Moves: ", self._allowed_actions())
        return True

    def take_action(self, action):
        """
        creates a new gamestate by copying this state and making a move
        :param action:  action  as chess move
        :return: newState, value, done
        """

        # Checks if move is correct
        # self.check_if_legal(action)

        new_boards = BughouseBoards(self.boards.fen)
        new_boards.push(action)


        newState = GameState(new_boards, self.board_number, -self.playerTurn)

        value = 0
        done = 0

        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return (newState, value, done)

    def push_action(self,action,board_id = None):
        """
        Update the gamestate by pushing a move
        :param action: python chess move
        :param board_id: If not None it will try to push the move on the specified board. If None the board_id in move is used
        """
        if board_id:
            self.boards.boards[board_id].push(action)
        else:
            self.boards.push(action)

    def render(self, logger):
        logger.info(self.boards.__str__())
        logger.info('--------------')


"""
Helper functions to convert python-chess representations to numpy arrays.
"""


def move_as_array(move, color=chess.WHITE):
    """
    Converts a python-chess move to a numpy array. The color is relevant for drops.
    :param move: a python-chess move
    :param color: chess.WHITE / 1 or chess.BLACK / 0
    :return: a numpy array (135,)
    """
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
    """
    Converts a numpy array to a python-chess move.
    :param action: move as a numpy array (135,)  see move_as_array
    :return: python-chess move
    """
    fromSquare = np.argmax(action[0:64])
    toSquare = np.argmax(action[64:128])
    if np.max(action[128:-1]) > 0:  # move is a drop
        color = action[-1]
        piece = chess.Piece(np.argmax(action[128:-1]) + 1, color)
        return chess.Move.from_uci(piece.symbol() + "@" + chess.SQUARE_NAMES[toSquare])
    else:
        return chess.Move(fromSquare, toSquare, None)

