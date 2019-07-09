import csv
import gzip
import os
import sys
import numpy as np
import tensorflow as tf

if __name__ == "pretraining.load_datasets":
    # ------- CALLED FROM main -------
    import pretraining.config_training as cf
    from game import input_representation, output_representation
    import chess
    from chess.variant import BughouseBoards
elif __name__ == "load_datasets":
    # ---- CALLED FROM training_from_database -----
    import config_training as cf
    PACKAGE_PARENT = '..'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
    from game import input_representation, output_representation
    import chess
    from chess.variant import BughouseBoards
else:
    raise ImportError(f"Name: {__name__} not found")


def tfdata(dataset, batch_size, is_training):
    # Construct a data generator using `tf.Dataset`.
    if is_training:
        dataset = dataset.shuffle(cf.SHUFFLE_BUFFER_SIZE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset.prefetch(buffer_size=2)
    return dataset


def load_data(batch_size, path="data/data.csv.gz"):
    n_samples = 0
    if cf.N_SAMPLES == None:
        print("counting samples")
        with gzip.open(path, 'rt') as f:
            for l in f:
                n_samples += 1
    else:
        n_samples = cf.N_SAMPLES

    full_dataset = tf.data.Dataset.from_generator(data_generator_processed,
                                                output_types=({'input_1': tf.float32, 'input_2': tf.float32}, {'value_head': tf.float32, 'policy_head': tf.float32}),
                                                output_shapes=({'input_1': tf.TensorShape(cf.INPUT_SHAPE_CHANNELS_LAST), 'input_2': tf.TensorShape(cf.INPUT_SHAPE_CHANNELS_LAST)},
                                                            {'value_head': tf.TensorShape(()), 'policy_head': tf.TensorShape((2272,))}))

    train_size = int(0.8 * n_samples)
    val_size = int(0.10 * n_samples)
    test_size = int(0.10 * n_samples)
    full_dataset = full_dataset.shuffle(cf.SHUFFLE_BUFFER_SIZE, seed=42)
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    train = tfdata(train_dataset, batch_size, is_training=True)
    val = tfdata(val_dataset, batch_size, is_training=False)
    test = tfdata(test_dataset, batch_size, is_training=False)
    return train, val, test, train_size, val_size, test_size


def data_generator_processed(path=cf.GDRIVE_FOLDER + "data/data.csv.gz", both_boards=True):
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
            x1 = np.moveaxis(x1, 0, 2)
            if both_boards:
                x2 = input_representation.board_to_planes(partner_board)
                # change channels first to channels last format

                x2 = np.moveaxis(x2, 0, 2)
                yield ({'input_1': x1, 'input_2': x2}, {'value_head': y, 'policy_head': y2})
            else:
                yield ({'input_1': x1}, {'value_head': y, 'policy_head': y2})


# def test(path="data/data.csv.gz"):
#     # dataset = tf.data.experimental.CsvDataset(path, [tf.string, tf.float32, tf.string],
#     #                                           compression_type="GZIP", field_delim=";")
#     # dataset = dataset.shuffle(10000)
#     generator = data_generator()
#     dataset = tf.data.Dataset.from_generator(data_generator_processed,
#                                              output_types=({'input_1': tf.float32, 'input_2': tf.float32}, {'value_head': tf.float32, 'policy_head': tf.float32}),
#                                              output_shapes=({'input_1': tf.TensorShape((34, 8, 8)), 'input_2': tf.TensorShape((34, 8, 8))},
#                                                             {'value_head': tf.TensorShape(( )), 'policy_head': tf.TensorShape((2272,))}))
#     iter = dataset.make_initializable_iterator()
#     el = iter.get_next()
#     with tf.Session() as sess:
#         sess.run(iter.initializer)
#         print(sess.run(el))
#         print(sess.run(el))
#         print(sess.run(el))
#     train, val, test, train_size, val_size, test_size = load_data(256)
#     iter = train.make_initializable_iterator()
#     el = iter.get_next()
#     with tf.Session() as sess:
#         sess.run(iter.initializer)
#         x = sess.run(el)
#         print(x[0]['input_1'].shape)
#         print(x[0]['input_2'].shape)
#         print(x[1]['value_head'].shape)
#         print(x[1]['policy_head'].shape)
#     # dataset = tf.data.Dataset().from_generator(iter,{'input_1': tf.float32, 'input_2': tf.float32}, {'value_head': tf.float32, 'policy_head': tf.float32})
#     # iter = dataset.make_initializable_iterator()
#     # el = iter.get_next()
#     # with tf.Session() as sess:
#     #     sess.run(iter.initializer)
#     #     x = sess.run(el)
#     #
#     #     print(x)
#
# test()
