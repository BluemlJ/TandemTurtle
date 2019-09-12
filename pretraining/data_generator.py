"""
DEPRECATED!!! use create_dataset.py and load_dataset.py
Generator functions for use with keras fit_generator
"""
# import input_representation, output_representation
from chess.variant import BughouseBoards
import chess
from game import input_representation, output_representation
import numpy as np
import sys
import time
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


def generate_value_sample(path_positions, path_results, path_nextMove, both_boards):
    with open(path_positions) as positions, open(path_results) as labels, open(path_nextMove) as nm_labels:
        for fen, result, move in zip(positions, labels, nm_labels):
            if move == 'end':
                continue
            board_number = 0 if "B1" in move else 1
            boards = BughouseBoards(fen)
            board = boards.boards[board_number]
            partner_board_number = 0 if board_number == 1 else 1
            partner_board = boards.boards[partner_board_number]

            # change the result to the perspective of the player which will move next
            if board_number == 1:
                result * -1
            if not board.turn:
                result * -1

            y = np.expand_dims(np.array(int(result)), axis=0)
            x1 = input_representation.board_to_planes(board)

            x1 = np.expand_dims(x1, axis=0)

            if both_boards:
                x2 = input_representation.board_to_planes(partner_board)
                x2 = np.expand_dims(x2, axis=0)
                yield (x1, x2, y)
            else:
                yield (x1, y)


def generate_value_batch(batch_size, path_positions, path_results, both_boards):
    single_sample_generator = generate_value_sample(path_positions, path_results, both_boards)
    while(True):
        start_time = time.time()
        sys.stdout.write(start_time)
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
            print("Time to load one value batch: ", str(time.time() - start_time))
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

            # TODO: Hotfix to skip end string
            if move == 'end':
                continue

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


def generate_value_policy_sample(path_positions, path_results, path_nextMove, both_boards):
    with open(path_positions) as positions, open(path_results) as labels, open(path_nextMove) as nm_labels:
        for fen, result, move in zip(positions, labels, nm_labels):
            board_number = 0 if "B1" in move else 1
            move = move.strip().split(' ')[-1]

            # TODO: Hotfix to skip end string
            if move == 'end':
                continue

            move = chess.Move.from_uci(move)

            boards = BughouseBoards(fen)
            board = boards.boards[board_number]
            partner_board_number = 0 if board_number == 1 else 1
            partner_board = boards.boards[partner_board_number]

            # change the result to the perspective of the player which will move next
            if board_number == 1:
                result = int(result) * -1
            if not board.turn:
                result = int(result) * -1
            y = np.array(int(result))
            # y = np.expand_dims(np.array(int(result)), axis=0)
            y2 = output_representation.move_to_policy(move, is_white_to_move=board.turn)
            # y2 = np.expand_dims(y2, axis=0)
            x1 = input_representation.board_to_planes(board)

            # x1 = np.expand_dims(x1, axis=0)

            if both_boards:
                x2 = input_representation.board_to_planes(partner_board)
                # x2 = np.expand_dims(x2, axis=0)
                yield (x1, x2, y, y2)
            else:
                yield (x1, y, y2)


def generate_value_policy_batch(batch_size, path_positions, path_results, path_nextMove):
    """
    yields a batch where input_1 is the board played on as array shape (batch_size,34,8,8),
     policy_head is the next move on the board shape (batch_size, 2272), input_2 is the partner board
    and value is 1 if the player to move will win, - 1 if the player will lose and 0  for draw

    pretraining.data_generator.generate_value_policy_batch(3,"data/position.train","data/result.train","data/nm.train")

    """

    single_sample_generator = generate_value_policy_sample(path_positions, path_results, path_nextMove, both_boards=True)
    while True:
        # start_time = time.time()
        """
        Concatenation of single inputs to fit a whole batch
        batch[0]: input_1 (own board)
        batch[1]: input_2 (partner board)
        batch[2]: value_head output with shape: (batch_size, )
        batch[3]: policy_head output with shape: (batch_size, 2272 (num of actions)) 
        """
        batches = [[], [], [], []]

        for _ in range(batch_size):
            sample = next(single_sample_generator)
            [batches[i].append(sample[i]) for i in range(4)]

        batch_x1 = np.array(batches[0])
        batch_x2 = np.array(batches[1])
        batch_value = np.array(batches[2])
        batch_nm = np.array(batches[3])
        # print({'input_1': batch_x1.shape, 'input_2': batch_x2.shape}, {'value_head': batch_value.shape, 'policy_head': batch_nm.shape})
        # print(f"\n\nTime to load one batch:{str(time.time() - start_time)} \n\n")
        yield ({'input_1': batch_x1, 'input_2': batch_x2},
               {'value_head': batch_value, 'policy_head': batch_nm})


# counts number of samples in a file
def num_samples(file_path):
    with open(file_path) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
