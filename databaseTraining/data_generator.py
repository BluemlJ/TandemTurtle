"""
Generator functions for use with keras fit_generator
"""

import itertools
import sys

from chess.variant import BughouseBoards

sys.path.append("..")
from input_representation import *
from constants import MV_LOOKUP, MV_LOOKUP_MIRRORED, mirror_move

def generate_value(path_positions, path_results, both_boards):
    with open(path_positions) as positions, open(path_results) as labels:
        for fen, result in zip(positions, labels):
            boards = BughouseBoards(fen)

            y = int(result)
            x1 = board_to_planes(boards.boards[0])
            x2 = board_to_planes(boards.boards[0])

            if both_boards:
                yield ({'input_1': x1, 'input_2': x2}, {'output': y})
            else:
                yield ({'input': x1}, {'output': y})
                yield ({'input': x2}, {'output': y})


def generate_nextMove(path_positions, path_nextMove):
    with open(path_positions) as positions, open(path_nextMove) as labels:
        for fen, move in zip(positions, labels):
            move = move.strip()
            boards = BughouseBoards(fen)
            board_number = 0 if "B1" in move else 1
            board = boards.boards[board_number]
            print(board)


            if board.turn:
                y = MV_LOOKUP[move[-4:]]
            else:
                # mirror move if it is blacks turn
                m = board.parse_uci(move[-4:])
                move = str(mirror_move(m))
                y = MV_LOOKUP_MIRRORED[move[-4:]]

            x = board_to_planes(board)
            yield ({'input': x}, {'output': y})
