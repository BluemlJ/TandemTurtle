"""
Parses the bpgn data.
Creates three files:
all_boards.txt contains all positions in the data as fen notations onse line for each position
results.txt one line for each position. -1 if black will win the game, 1 white, 0 draw
next_move next move for each position as string with board number (e.g B1 e2e4)

TODO How long does this take to finish??? make faster
TODO  add train/test/validation split
TODO filter games (cancelled, connection lost)
"""
from chess.variant import BughouseBoards
import sys
sys.path.append("..")
from database_extraction.extract import read_data
import chess.pgn
from input_representation import *
import os
import codecs

path = "../database_extraction/data/"
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.bpgn' in file and ".bz2" not in file:
            files.append(os.path.join(r, file))
with open("data/all_boards.txt", "w") as all_boards, open("data/results.txt", "w") as labels, open("data/next_move.txt",
                                                                                                   'w') as next_moves:
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