#!/usr/bin/python
import sys
for line in sys.stdin:
    data = line.strip().split('\t')
    if len(data) == 145:
        homeTeam = data[11]
        awayTeam = data[13]
        pitcherFirstName = data[50]
        pitcherLastName = data[51]
        outs = data[66]
        if pitcherFirstName != None and pitcherLastName != None:
            print homeTeam, "\t", awayTeam, "\t", pitcherFirstName, "\t", pitcherLastName,  "\t", outs