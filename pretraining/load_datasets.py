from chess.variant import BughouseBoards
import chess
from game import constants as game_constants
from game import input_representation, output_representation
import gzip
import random
from zipfile import ZipFile

import numpy as np
import sys
import time
import os
import tensorflow as tf

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


# with ZipFile("data/data.zip", "r") as zip_file:
#     listOfFileNames = zip_file.namelist()
#     n_samples = len(listOfFileNames)
#
#     shuffled_ids = random.shuffle(listOfFileNames)
#
#     start_test = int(0.8 * n_samples)
#     test_ids = shuffled_ids[start_test:]
#     train_ids = shuffled_ids[:start_test]
#
#     start_val = int(0.8 * len(train_ids))
#     val_ids = train_ids [start_val:]
#     train_ids = train_ids [:start_val]
#
#     partitions = {'train': train_ids , 'validation': val_ids, 'test' : test_ids}
# def generate_sample(id):
#    with ZipFile("data/data.zip", "r") as zip_file:
#       with   zip_file.open()
# def generator(partition = 'train'):
#     ids = partitions[partition]

N_SAMPLES = 1342846339


def position_to_arrays(position, result, nm, both_boards):
    nm = str(nm).strip()
    board_number = 0 if "B1" in nm else 1
    move = nm.replace('tf.Tensor(b\'', "").replace("\\n'", "").replace(", shape=(), dtype=string)", "").strip().split(' ')[-1]  # TODO can this be done less ugly?
    boards = BughouseBoards(str(position).replace('tf.Tensor(b\'', "").replace("\\n'", "").replace(", shape=(), dtype=string)", "").strip())  # TODO can this be done less ugly?
    board = boards.boards[board_number]
    partner_board_number = 0 if board_number == 1 else 1
    partner_board = boards.boards[partner_board_number]
    move = chess.Move.from_uci(move)

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
        return (x1, x2, y, y2)
    else:
        return (x1, y, y2)


def tfdata_generator(dataset, batch_size, is_training):
    # Construct a data generator using `tf.Dataset`.
    def map_fn(position, value, nm):
        x1, x2, y, y2 = tf.compat.v1.py_function(position_to_arrays, [position, value, nm, True], [tf.float32, tf.float32, tf.float32, tf.float32])
        return x1, x2, y, y2

    if is_training:
        print("shuffle")
        dataset = dataset.shuffle(10000)  # depends on sample size
    print("map")
    dataset = dataset.map(map_fn)
    print("batch")
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    print("prefetch")
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def load_data(batch_size, path="data/data.csv.gz"):
    n_samples = 0
    if N_SAMPLES == None:
        print("counting samples")
        with gzip.open(path, 'rt') as f:
            for l in f:
                n_samples += 1
    else:
        n_samples = N_SAMPLES
    print("n_samples: %i" % n_samples)
    full_dataset = tf.data.experimental.CsvDataset(path, [tf.string, tf.float32, tf.string],
                                              compression_type="GZIP", field_delim=";")
    train_size = int(0.8 * n_samples)
    val_size = int(0.10 * n_samples)
    test_size = int(0.10 * n_samples)
    print("shuffle")
    full_dataset = full_dataset.shuffle(10000, seed=42)
    print("split")
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    print("train")
    train = tfdata_generator(train_dataset, batch_size, is_training=True)
    print("val")
    val = tfdata_generator(val_dataset, batch_size, is_training=False)
    print("test")
    test = tfdata_generator(test_dataset, batch_size, is_training=False)
    return train, val, test, train_size, val_size, test_size


# def test(path="data/data.csv.gz"):
#   dataset = tf.data.experimental.CsvDataset(path, [tf.string, tf.float32, tf.string],
#                                                  compression_type="GZIP", field_delim=";")
#   #dataset = dataset.shuffle(10000)
#   iter = dataset.make_initializable_iterator()
#   el = iter.get_next()
#   with tf.Session() as sess:
#     sess.run(iter.initializer)
#     print(sess.run(el))
#     print(sess.run(el))
#     print(sess.run(el))
#   train, val, test, train_size, val_size, test_size = load_data(256)
#   iter = train.make_initializable_iterator()
#   el = iter.get_next()
#   with tf.Session() as sess:
#     sess.run(iter.initializer)
#     x = sess.run(el)
#
#     print(x[0].shape, x[1].shape,x[2].shape,x[3].shape)
#
# test()
