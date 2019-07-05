"""
Parses the bpgn data.
Creates three files:
all_boards.txt contains all positions in the data as fen notations onse line for each position
results.txt one line for each position. -1 if black will win the game, 1 white, 0 draw
next_move next move for each position as string with board number (e.g B1 e2e4)

TODO How long does this take to finish??? make faster
TODO filter games (cancelled, connection lost)
"""
import codecs
import csv
import gzip
import os
from zipfile import ZipFile

import chess.pgn
import sys

sys.path.append("..")


def parse_all_positions(in_dir="../database_extraction/data/", out_dir="data/"):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(in_dir):
        for file in f:
            if '.bpgn' in file and ".bz2" not in file:
                files.append(os.path.join(r, file))
    with open(out_dir + "all_boards.txt", "w") as all_boards, \
            open(out_dir + "results.txt", "w") as labels, \
            open(out_dir + "next_move.txt", 'w') as next_moves:

        for file in files:
            with codecs.open(file, encoding="utf-8-sig", errors='ignore') as f:
                game = chess.pgn.read_game(f)
                while game:
                    boards = game.board()
                    result = game.headers["Result"]
                    if result == '0-1':
                        label = "-1"
                    elif result == '1-0':
                        label = "1"
                    else:
                        label = "0"

                    for move in game.mainline_moves():
                        all_boards.write(boards.fen + "\n")
                        labels.write(label + "\n")
                        next_moves.write(str(move) + "\n")
                        boards.push(move)

                    all_boards.write(boards.fen + "\n")
                    labels.write(label + "\n")
                    next_moves.write("end\n")

                    game = chess.pgn.read_game(f)


def splits(dir="data/"):
    """
    Splits data into train/validation/tset file
    Use not if training with tf.data (train_from_database_tf)
    TODO make split size parameters
    :param dir: data dir
    """
    with open(dir + "all_boards.txt", "r") as all_boards, \
            open(dir + "results.txt", "r") as labels, \
            open(dir + "next_move.txt", 'r') as next_moves,\
            open(dir + "position.train", 'w') as p_train,\
            open(dir + "position.validation", 'w') as p_validation,\
            open(dir + "position.test", 'w') as p_test,\
            open(dir + "result.train", 'w') as v_train,\
            open(dir + "result.validation", 'w') as v_validation,\
            open(dir + "result.test", 'w') as v_test, \
            open(dir + "nm.train", 'w') as nm_train, \
            open(dir + "nm.validation", 'w') as nm_validation, \
            open(dir + "nm.test", 'w') as nm_test:\

        # shuffle_files(["all_boards.txt","results.txt","next_move.txt"]) TODO

        for i, (position, result, nm) in enumerate(zip(all_boards, labels, next_moves)):
            # every tenth point as test
            if i % 10 == 0:
                p_test.write(position)
                v_test.write(result)
                nm_test.write(nm)
            elif i % 15 == 0:
                p_validation.write(position)
                v_validation.write(result)
                nm_validation.write(nm)
            else:
                p_train.write(position)
                v_train.write(result)
                nm_train.write(nm)


def compressed_data(dir="data/"):
    with open(dir + "all_boards.txt", "r") as all_boards, \
            open(dir + "results.txt", "r") as labels, \
            open(dir + "next_move.txt", 'r') as next_moves:
        with gzip.open(dir + "data.csv.gz", 'wt') as f:
            n = 0
            for i, (position, result, nm) in enumerate(zip(all_boards, labels, next_moves)):

                if 'end' in nm:
                    continue
                n += 1
                writer = csv.writer(f, delimiter=';')
                writer.writerow([position, result, nm])
            print("Num samples: %i" % n)


def main():
    # parse_all_positions()
    # splits()
    compressed_data()


if __name__ == "__main__":
    main()
