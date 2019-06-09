"""
Generator functions for use with keras fit_generator
"""
import chess


import input_representation
import output_representation
import numpy as np
import sys

from chess.variant import BughouseBoards

sys.path.append("..")


def generate_value_sample(path_positions, path_results, both_boards):
    with open(path_positions) as positions, open(path_results) as labels:
        for fen, result in zip(positions, labels):
            boards = BughouseBoards(fen)

            y = np.expand_dims(np.array(int(result)), axis=0)
            x1 = input_representation.board_to_planes(boards.boards[0])
            x2 = input_representation.board_to_planes(boards.boards[0])
            x1 = np.expand_dims(x1, axis=0)
            x2 = np.expand_dims(x2, axis=0)
            if both_boards:
                yield (x1, x2, y)
            else:
                yield (x1, y)
                yield (x2, y)


def generate_value_batch(batch_size, path_positions, path_results, both_boards):
    single_sample_generator = generate_value_sample(path_positions, path_results, both_boards)
    while(True):
        if both_boards:
            sample = next(single_sample_generator)
            batch_x1 = sample[0]
            batch_x2 = sample[1]
            batch_y = sample[2]
            while batch_x1.shape[0] < batch_size:
                sample = next(single_sample_generator)
                batch_x1 = np.concatenate([batch_x1, sample[0]])
                batch_x2 = np.concatenate([batch_x2, sample[1]])
                batch_y = np.concatenate([batch_y, sample[2]])
            yield ({'input_1': batch_x1, 'input_2': batch_x2}, {'value_head': batch_y})
        else:
            sample = next(single_sample_generator)
            batch_x = sample[0]
            batch_y = sample[1]
            while batch_x.shape[0] < batch_size:
                sample = next(single_sample_generator)
                batch_x = np.concatenate([batch_x, sample[0]])
                batch_y = np.concatenate([batch_y, sample[1]])
            yield ({'input_1': batch_x}, {'value_head': batch_y})


def generate_nextMove_sample(path_positions, path_nextMove, both_boards):
    with open(path_positions) as positions, open(path_nextMove) as labels:
        for fen, move in zip(positions, labels):
            board_number = 0 if "B1" in move else 1
            move = move.strip().split(' ')[-1]
            move = chess.Move.from_uci(move)

            boards = BughouseBoards(fen)
            board = boards.boards[board_number]
            partner_board_number = 0 if board_number == 1 else 1
            partner_board = boards.boards[partner_board_number]

            y = output_representation.move_to_policy(move, is_white_to_move=board.turn)
            y = np.expand_dims(y, axis=0)
            x = input_representation.board_to_planes(board)
            x = np.expand_dims(x, axis=0)
            if both_boards:
                x2 = input_representation.board_to_planes(partner_board)
                x2 = np.expand_dims(x2, axis=0)
                yield (x, x2, y)
            else:
                yield (x, y)


def generate_nextMove_batch(batch_size, path_positions, path_nextMove):
    single_sample_generator = generate_nextMove_sample(path_positions, path_nextMove)
    while(True):
        sample = next(single_sample_generator)
        batch_x = sample[0]
        batch_y = sample[1]
        while batch_x.shape[0] < batch_size:
            sample = next(single_sample_generator)
            batch_x = np.concatenate([batch_x, sample[0]])
            batch_y = np.concatenate([batch_y, sample[1]])
        yield ({'input_1': batch_x}, {'policy_head': batch_y})


def generate_value_policy_batch(batch_size, path_positions, path_results, path_nextMove):
    """
    yields a batch where input_1 is the board played on as array shape (batch_size,34,8,8),
     policy_head is the next move on the board shape (batch_size, 2272), input_2 is the partner board
    and value is 1 if white wins, - black , 0 draw

    databaseTraining.data_generator.generate_value_policy_batch(3,"data/position.train","data/result.train","data/nm.train")

    """
    single_sample_generator_nm = generate_nextMove_sample(path_positions, path_nextMove, True)
    single_sample_generator_value = generate_value_sample(path_positions, path_results, True)
    while(True):
        sample_value = next(single_sample_generator_value)
        sample_nm = next(single_sample_generator_nm)
        batch_x1 = sample_nm[0]
        batch_x2 = sample_nm[1]
        batch_value = sample_value[2]
        batch_nm = sample_nm[2]
        while batch_x1.shape[0] < batch_size:
            sample_value = next(single_sample_generator_value)
            sample_nm = next(single_sample_generator_nm)
            batch_x1 = np.concatenate([batch_x1, sample_nm[0]])
            batch_x2 = np.concatenate([batch_x2, sample_nm[1]])
            batch_value = np.concatenate([batch_value, sample_value[2]])
            batch_nm = np.concatenate([batch_nm, sample_nm[2]])
        yield ({'input_1': batch_x1, 'input_2': batch_x2}, {'value_head': batch_value, 'policy_head': batch_nm})


# counts number of samples in a file
def num_samples(file_path):
    with open(file_path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
