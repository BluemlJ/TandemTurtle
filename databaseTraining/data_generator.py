"""
Generator functions for use with keras fit_generator
"""

import sys

from chess.variant import BughouseBoards

sys.path.append("..")
from input_representation import *
from constants import MV_LOOKUP, MV_LOOKUP_MIRRORED, mirror_move



def generate_value_sample(path_positions, path_results, both_boards):
    with open(path_positions) as positions, open(path_results) as labels:
        for fen, result in zip(positions, labels):
            boards = BughouseBoards(fen)

            y = np.expand_dims(np.array(int(result)), axis=0)
            x1 = board_to_planes(boards.boards[0])
            x2 = board_to_planes(boards.boards[0])
            x1 = np.expand_dims(x1, axis=0)
            x2 = np.expand_dims(x2, axis=0)
            if both_boards:
                yield (x1, x2, y)
            else:
                yield (x1,y)
                yield (x2,y)

def generate_value_batch(batch_size,path_positions, path_results, both_boards):
    single_sample_generator = generate_value_sample(path_positions, path_results, both_boards)
    while(True):
        if both_boards:
            sample = next(single_sample_generator)
            batch_x1 = sample[0]
            batch_x2 = sample[1]
            batch_y = sample[2]
            while batch_x1.shape[0] < batch_size:
                sample = next(single_sample_generator)
                batch_x1 = np.concatenate([batch_x1,sample[0]])
                batch_x2 = np.concatenate([batch_x2,sample[1]])
                batch_y = np.concatenate([batch_y,sample[2]])
            yield ({'input_1': batch_x1,'input_2': batch_x2}, {'value_head': batch_y})
        else:
            sample = next(single_sample_generator)
            batch_x = sample[0]
            batch_y = sample[1]
            while batch_x.shape[0] < batch_size:
                sample = next(single_sample_generator)
                batch_x = np.concatenate([batch_x,sample[0]])
                batch_y = np.concatenate([batch_y,sample[1]])
            yield ({'input_1': batch_x}, {'value_head': batch_y})



def generate_nextMove_sample(path_positions, path_nextMove):
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

            y = np.array(y)
            y = np.expand_dims(y, axis=0)
            x = board_to_planes(board)
            x = np.expand_dims(x, axis=0)
            yield (x,y)

def generate_nextMove_batch(batch_size,path_positions, path_nextMove):
    single_sample_generator =generate_nextMove_sample(path_positions, path_nextMove)
    while(True):
        sample = next(single_sample_generator)
        batch_x = sample[0]
        batch_y = sample[1]
        while batch_x.shape[0] < batch_size:
            sample = next(single_sample_generator)
            batch_x = np.concatenate([batch_x,sample[0]])
            batch_y = np.concatenate([batch_y,sample[1]])
        yield ({'input_1': batch_x}, {'policy_head': batch_y})



#counts number of samples in a file
def num_samples(file_path):
    with open(file_path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1