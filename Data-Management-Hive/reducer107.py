#!/usr/bin/python
import sys
from operator import itemgetter
games = []
gamesWithSummedOuts = []
# records = []
for line in sys.stdin:
    data_mapped = line.strip().split("\t")
    if len(data_mapped) != 5:
        continue
    homeTeam, awayTeam, pitcherFirstName, pitcherLastName, outs = data_mapped
    outs = int(outs)

    games.append({
        "homeTeam": homeTeam,
        "awayTeam": awayTeam,
        "pitcherFirstName": pitcherFirstName,
        "pitcherLastName": pitcherLastName,
        "outs": outs
    })

sortedGames = sorted(games, key=itemgetter('homeTeam', 'awayTeam', 'pitcherFirstName', 'pitcherLastName'), reverse=True)
prevRecord = str(sortedGames[0]["homeTeam"]) + str(sortedGames[0]["awayTeam"]) + str(sortedGames[0]["pitcherFirstName"]) + str(sortedGames[0]["pitcherLastName"])
sumOuts = 0
for x in range(len(sortedGames)):
    currRecord = str(sortedGames[x]["homeTeam"]) + str(sortedGames[x]["awayTeam"]) + str(sortedGames[x]["pitcherFirstName"]) + str(sortedGames[x]["pitcherLastName"])
    if currRecord == prevRecord:
        sumOuts += sortedGames[x]["outs"]
    else:
        gamesWithSummedOuts.append({
            "homeTeam": sortedGames[x-1]["homeTeam"],
            "awayTeam": sortedGames[x-1]["awayTeam"],
            "pitcherFirstName": sortedGames[x-1]["pitcherFirstName"],
            "pitcherLastName": sortedGames[x-1]["pitcherLastName"],
            "outs": sumOuts
        })
        sumOuts = 0
        sumOuts += sortedGames[x]["outs"]
        prevRecord = currRecord
sortedOuts = sorted(gamesWithSummedOuts, key=itemgetter('outs'), reverse=True)
print sortedOuts[0]["homeTeam"], "\t", sortedOuts[0]["awayTeam"], "\t", sortedOuts[0]["pitcherFirstName"], "\t", sortedOuts[0]["pitcherLastName"], "\t", sortedOuts[0]["outs"]
