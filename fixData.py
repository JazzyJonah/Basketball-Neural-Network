"""When formatting the file, I had a bug where games against non-D1 teams had partial data
   This script fixes that by removing those games from the formatted data file."""
   
from globals import *
from tqdm import tqdm

with open(historicalDataFile, "r") as f:
    data = f.readlines()

newData = []
for line in tqdm(data):
    if line.strip() == "":
        continue
    fields = line.strip().split(",")
    if len(fields) == 34: # If it's a complete row
        # There's this one line that had a Duke vs WF game where it says Duke scored 18 (they actually scored 101). That line is just bizzarre
        if fields[2] == "Duke" and fields[18] == "Wake Forest" and fields[4] == "18":
            print(f"Deleting bizzarre Duke vs Wake Forest line: {line}")
            continue
        # Same with Radford scoring 8 against Gardner-Webb?
        if fields[2] == "Gardner-Webb" and fields[18] == "Radford" and fields[20] == "8":
            print(f"Deleting bizzarre Gardner-Webb vs Radford line: {line}")
            continue
        # There are some bizzarre weird cases where both teams are "home." These are cursed games and other things are wrong with them as well. Delete them.
        if fields[3] == "True" and fields[19] == "True":
            print(f"Deleting cursed both-home line: {line}")
            continue
        newData.append(line)
    else:
        print(f"Found incomplete row: {line}")

with open(historicalDataFile, "w") as f:
    for line in newData:
        f.write(line)