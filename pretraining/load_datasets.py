import keras
from game import input_representation, output_representation
import csv
import gzip
import os
import sys

import numpy as np
import tensorflow as tf

import chess
from chess.variant import BughouseBoards

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
N_SAMPLES = 1342846339


def position_to_arrays(position, result, nm, both_boards):
    nm = str(nm).strip()
    board_number = 0 if "B1" in nm else 1
    move = \
    nm.replace('tf.Tensor(b\'', "").replace("\\n'", "").replace(", shape=(), dtype=string)", "").strip().split(' ')[
        -1]  # TODO can this be done less ugly?
    boards = BughouseBoards(
        str(position).replace('tf.Tensor(b\'', "").replace("\\n'", "").replace(", shape=(), dtype=string)",
                                                                               "").strip())  # TODO can this be done less ugly?
    board = boards.boards[board_number]
    partner_board_number = 0 if board_number == 1 else 1
    partner_board = boards.boards[partner_board_number]
    move = chess.Move.from_uci(move)

    if board_number == 1:
        result = int(result) * -1
    if not board.turn:
        result = int(result) * -1
    y = np.array(int(result))
    y2 = output_representation.move_to_policy(move, is_white_to_move=board.turn)
    x1 = input_representation.board_to_planes(board)

    if both_boards:
        x2 = input_representation.board_to_planes(partner_board)
        # change channels first to channels last format
        #x1 = np.moveaxis(x1, 0, 2)
        #x2 = np.moveaxis(x2, 0, 2)
        return (x1, x2, y, y2)
    else:
        return (x1, y, y2)


# def tfdata(dataset, batch_size, is_training):
#     # Construct a data generator using `tf.Dataset`.
#     def map_fn(position, value, nm):
#         x1, x2, y, y2 = tf.compat.v1.py_function(position_to_arrays, [position, value, nm, True],
#                                               [tf.float32, tf.float32, tf.float32, tf.float32])
#         x1.set_shape((34, 8, 8))
#         x2.set_shape((34, 8, 8))
#         #x1.set_shape((8,8,34))
#        # x2.set_shape((8,8,34))
#         y.set_shape((1,))
#         y2.set_shape((2272,))
#         return ({'input_1': x1, 'input_2': x2}, {'value_head': y, 'policy_head': y2})
#
#     if is_training:
#         dataset = dataset.shuffle(10000)
#
#     dataset = dataset.map(map_fn)
#
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     dataset = dataset.repeat()
#     #dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
#
#     return dataset
def tfdata(dataset, batch_size, is_training):
    # Construct a data generator using `tf.Dataset`.
    if is_training:
        dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset.prefetch(buffer_size=1000)
    #dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset


# def tfdata_generators(dataset, batch_size, is_training):
#     # Construct a data generator using `tf.Dataset`.
#     def map_fn(position, value, nm):
#         x1, x2, y, y2 = tf.compat.v1.py_function(position_to_arrays, [position, value, nm, True],
#                                               [tf.float32, tf.float32, tf.float32, tf.float32])
#         x1.set_shape((34, 8, 8))
#         x2.set_shape((34, 8, 8))
#         y.set_shape((1,))
#         y2.set_shape((2272,))
#         return ({'input_1': x1, 'input_2': x2}, {'value_head': y, 'policy_head': y2})
#
#     #if is_training:
#      #   dataset = dataset.shuffle(10000)
#
#     dataset = dataset.map(map_fn)
#
#     dataset = dataset.batch(batch_size)
#     dataset = dataset.repeat()
#     #dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
#     iterator = dataset.make_one_shot_iterator()
#
#     next_batch = iterator.get_next()
#     while True:
#         yield keras.backend.get_session().run(next_batch)
def tfdata_generators(dataset, batch_size, is_training):
    # Construct a data generator using `tf.Dataset`.

    if is_training:
        dataset = dataset.shuffle(10000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset.prefetch(buffer_size=1000)
    iterator = dataset.make_one_shot_iterator()

    next_batch = iterator.get_next()
    while True:
        yield keras.backend.get_session().run(next_batch)


def data_generator(path="data/data.csv.gz"):
    with gzip.open(path, 'rt') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            yield row[0], row[1], row[2]

# def load_data(batch_size, path="data/data.csv.gz"):
#     n_samples = 0
#     if N_SAMPLES == None:
#         print("counting samples")
#         with gzip.open(path, 'rt') as f:
#             for l in f:
#                 n_samples += 1
#     else:
#         n_samples = N_SAMPLES
#
#     full_dataset = tf.data.experimental.CsvDataset(path, [tf.string, tf.float32, tf.string],
#                                                    compression_type="GZIP", field_delim=";")
#     # full_dataset = tf.data.Dataset.from_generator(data_generator, output_types=('string', 'string', 'string'))
#     train_size = int(0.8 * n_samples)
#     val_size = int(0.10 * n_samples)
#     test_size = int(0.10 * n_samples)
#     full_dataset = full_dataset.shuffle(10000, seed=42)
#     train_dataset = full_dataset.take(train_size)
#     test_dataset = full_dataset.skip(train_size)
#     val_dataset = test_dataset.skip(val_size)
#     test_dataset = test_dataset.take(test_size)
#     train = tfdata(train_dataset, batch_size, is_training=True)
#     val = tfdata(val_dataset, batch_size, is_training=False)
#     test = tfdata(test_dataset, batch_size, is_training=False)
#     return train, val, test, train_size, val_size, test_size


def load_data_generators(batch_size, path="data/data.csv.gz"):
    n_samples = 0
    if N_SAMPLES == None:
        print("counting samples")
        with gzip.open(path, 'rt') as f:
            for l in f:
                n_samples += 1
    else:
        n_samples = N_SAMPLES

    # full_dataset = tf.data.experimental.CsvDataset(path, [tf.string, tf.float32, tf.string],
    #                                                compression_type="GZIP", field_delim=";")
    # full_dataset = tf.data.Dataset.from_generator(data_generator, output_types=('string', 'string', 'string'))
    full_dataset = tf.data.Dataset.from_generator(data_generator_processed,
                                                output_types=({'input_1': tf.float32, 'input_2': tf.float32}, {'value_head': tf.float32, 'policy_head': tf.float32}),
                                                output_shapes=({'input_1': tf.TensorShape((34, 8, 8)), 'input_2': tf.TensorShape((34, 8, 8))},
                                                            {'value_head': tf.TensorShape(()), 'policy_head': tf.TensorShape((2272,))}))
    train_size = int(0.8 * n_samples)
    val_size = int(0.10 * n_samples)
    test_size = int(0.10 * n_samples)
    full_dataset = full_dataset.shuffle(10000, seed=42)
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    train = tfdata_generators(train_dataset, batch_size, is_training=True)
    val = tfdata_generators(val_dataset, batch_size, is_training=False)
    test = tfdata_generators(test_dataset, batch_size, is_training=False)
    return train, val, test, train_size, val_size, test_size


def load_data(batch_size, path="data/data.csv.gz"):
    n_samples = 0
    if N_SAMPLES == None:
        print("counting samples")
        with gzip.open(path, 'rt') as f:
            for l in f:
                n_samples += 1
    else:
        n_samples = N_SAMPLES

    full_dataset = tf.data.Dataset.from_generator(data_generator_processed,
                                                output_types=({'input_1': tf.float32, 'input_2': tf.float32}, {'value_head': tf.float32, 'policy_head': tf.float32}),
                                                output_shapes=({'input_1': tf.TensorShape((34, 8, 8)), 'input_2': tf.TensorShape((34, 8, 8))},
                                                            {'value_head': tf.TensorShape(()), 'policy_head': tf.TensorShape((2272,))}))

    train_size = int(0.8 * n_samples)
    val_size = int(0.10 * n_samples)
    test_size = int(0.10 * n_samples)
    full_dataset = full_dataset.shuffle(10000, seed=42)
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    train = tfdata(train_dataset, batch_size, is_training=True)
    val = tfdata(val_dataset, batch_size, is_training=False)
    test = tfdata(test_dataset, batch_size, is_training=False)
    return train, val, test, train_size, val_size, test_size


def data_generator_processed(path="data/data.csv.gz", both_boards=True):
    with gzip.open(path, 'rt') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            position, result, nm = row[0], row[1], row[2]
            result = int(result)
            nm = str(nm).strip()
            board_number = 0 if "B1" in nm else 1
            move = nm.strip().split(' ')[-1]
            boards = BughouseBoards(position)
            board = boards.boards[board_number]
            partner_board_number = 0 if board_number == 1 else 1
            partner_board = boards.boards[partner_board_number]
            move = chess.Move.from_uci(move)

            if board_number == 1:
                result = int(result) * -1
            if not board.turn:
                result = int(result) * -1
            y = np.array(int(result))
            y2 = output_representation.move_to_policy(move, is_white_to_move=board.turn)
            x1 = input_representation.board_to_planes(board)

            if both_boards:
                x2 = input_representation.board_to_planes(partner_board)
                # change channels first to channels last format
                # x1 = np.moveaxis(x1, 0, 2)
                # x2 = np.moveaxis(x2, 0, 2)
                yield ({'input_1': x1, 'input_2': x2}, {'value_head': y, 'policy_head': y2})
            else:
                yield ({'input_1': x1}, {'value_head': y, 'policy_head': y2})


def test(path="data/data.csv.gz"):
    # dataset = tf.data.experimental.CsvDataset(path, [tf.string, tf.float32, tf.string],
    #                                           compression_type="GZIP", field_delim=";")
    # dataset = dataset.shuffle(10000)
    generator = data_generator()
    dataset = tf.data.Dataset.from_generator(data_generator_processed,
                                             output_types=({'input_1': tf.float32, 'input_2': tf.float32}, {'value_head': tf.float32, 'policy_head': tf.float32}),
                                             output_shapes=({'input_1': tf.TensorShape((34, 8, 8)), 'input_2': tf.TensorShape((34, 8, 8))},
                                                            {'value_head': tf.TensorShape(()), 'policy_head': tf.TensorShape((2272,))}))
    iter = dataset.make_initializable_iterator()
    el = iter.get_next()
    with tf.Session() as sess:
        sess.run(iter.initializer)
        print(sess.run(el))
        print(sess.run(el))
        print(sess.run(el))
    train, val, test, train_size, val_size, test_size = load_data(256)
    iter = train.make_initializable_iterator()
    el = iter.get_next()
    with tf.Session() as sess:
        sess.run(iter.initializer)
        x = sess.run(el)
        print(x[0]['input_1'].shape)
        print(x[0]['input_2'].shape)
        print(x[1]['value_head'].shape)
        print(x[1]['policy_head'].shape)
    # dataset = tf.data.Dataset().from_generator(iter,{'input_1': tf.float32, 'input_2': tf.float32}, {'value_head': tf.float32, 'policy_head': tf.float32})
    # iter = dataset.make_initializable_iterator()
    # el = iter.get_next()
    # with tf.Session() as sess:
    #     sess.run(iter.initializer)
    #     x = sess.run(el)
    #
    #     print(x)


test()
