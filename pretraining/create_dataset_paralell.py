import codecs
import csv
import gzip
import multiprocessing
import os
import re
import numpy as np
from multiprocessing import Process, Queue
import chess.pgn
import progressbar


def find_files(in_dir="../database_extraction/data/"):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(in_dir):
        for file in f:
            if '.bpgn' in file and ".bz2" not in file:
                files.append(os.path.join(r, file))
    return files


def statistics(files):
    for file in files:
        with codecs.open(file, encoding="utf-8-sig", errors='ignore') as f:

            times = []
            elos = []
            averageElos = []
            averageElo = 0
            playersWithElo = 0
            for i, line in enumerate(f):
                if "Elo" in line:
                    elo = ''.join(x for x in line.split()[-1] if x.isdigit())
                    if not elo == "":
                        elo = int(elo)
                        elos.append(elo)
                        averageElo += elo
                        playersWithElo += 1

                if "TimeControl" in line:
                    time = re.search(r'"([^"]*)"', line)
                    times.append(time[0][1:len(time[0]) - 1])
                    if playersWithElo != 0:
                        averageElo = averageElo / playersWithElo
                        averageElos.append(averageElo)
                        averageElo = 0
                        playersWithElo = 0
    average = np.mean(elos)
    print("average elo:", average)

    p_80 = np.percentile(elos, 80)
    print("80 percentile", p_80)
    p_90 = np.percentile(elos, 90)
    print("90 percentile", p_90)

    pa_80 = np.percentile(averageElos, 80)
    print("80 percentile", pa_80)
    pa_90 = np.percentile(averageElos, 90)
    print("90 percentile", pa_90)

    print("n games", len(averageElos))
    return averageElos, elos, times


def parse_file(file, writerQueue, elolimit):
    with codecs.open(file, encoding="utf-8-sig", errors='ignore') as f:
        game = chess.pgn.read_game(f)
        while game:
            boards = game.board()
            result = game.headers["Result"]
            if not result == "*":
                if result == '0-1':
                    label = "-1"
                elif result == '1-0':
                    label = "1"
                else:
                    label = "0"
                elos = []
                try:
                    elos.append(int(game.headers["WhiteA"].split('"')[-1]))
                    elos.append(int(game.headers["BlackA"].split('"')[-1]))
                    elos.append(int(game.headers["WhiteB"].split('"')[-1]))
                    elos.append(int(game.headers["BlackB"].split('"')[-1]))
                    mean_elo = np.mean(elos)
                except:
                    mean_elo = 0
                if mean_elo >= elolimit:
                    for move in game.mainline_moves():
                        writerQueue.put([boards.fen, label, str(move)])
                        boards.push(move)

            game = chess.pgn.read_game(f)


def feed_files(workerQueue, in_dir="../database_extraction/data/"):
    for r, d, f in os.walk(in_dir):
        for file in f:
            if '.bpgn' in file and ".bz2" not in file:
                path = os.path.join(r, file)
                workerQueue.put(path)


def worker(workerQueue, writerQueue, elolimit):
    while True:
        try:
            file = workerQueue.get(block=False)
            print("start parsing %s" % file)
            parse_file(file, writerQueue, elolimit)
            print("%s done" % file)
        except:
            break


def write(queue, fname):
    with gzip.open(fname, 'wt') as f:
        writer = csv.writer(f, delimiter=';')
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        n = 0
        while True:
            try:
                sample = queue.get(block=True)
                writer.writerow(sample)
                n += 1
                bar.update(n)
            except:
                print("Number samples: %i" % n)
                break


def create_dataset(in_dir="../database_extraction/data/", out_dir="data/", percentile=90, max_workers=None):
    #files = find_files(in_dir)
    #averageElos, elos, times = statistics(files)
    #elolimit = np.percentile(averageElos, percentile)
    elolimit = 2200

    nworkers = multiprocessing.cpu_count()
    if max_workers:
        nworkers = min(nworkers, max_workers)

    fname = out_dir + "data_top_%i.csv.gz" % (100 - percentile)

    workerQueue = Queue()
    writerQueue = Queue()
    feedProc = Process(target=feed_files, args=(workerQueue, in_dir))
    calcProc = [Process(target=worker, args=(workerQueue, writerQueue, elolimit)) for i in range(nworkers)]
    writProc = Process(target=write, args=(writerQueue, fname))

    feedProc.start()
    for p in calcProc:
        p.start()
    writProc.start()

    feedProc.join()
    for p in calcProc:
        p.join()
    writProc.join()


create_dataset()
