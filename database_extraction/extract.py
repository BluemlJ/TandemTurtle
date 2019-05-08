import os
import re
import csv
import pandas as pd
path = '../data'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.bpgn' in file and ".bz2" not in file:
            files.append(os.path.join(r, file))

games = {"Time" : [], "Result" : [], "Moves" : []}
for f in files:
    a = open(f, "r")
    content = a.readlines()
    game = {}
    for number, line in enumerate(content):
        if "TimeControl" in line:
            time = re.search(r'"([^"]*)"', line)
            game["Time"] = time[0][1:len(time[0])-1]
        if "Result" in line:
            result = re.search(r'"([^"]*)"', line)
            if result != None:
                if result[0] == '"1-0"' or result[0] == '"0-1"':
                    game["Result"] = result[0][1:len(result[0])-1]
        if "C:" in line:
            game["Moves"] = content[number+1]
            if len(game) == 3:
                games["Time"] += [game["Time"]]
                games["Result"] += [game["Result"]]
                games["Moves"] += [game["Moves"]]
            game = {}


df = pd.DataFrame(data=games)
df.to_csv("test.csv", sep=';', encoding='utf-8')