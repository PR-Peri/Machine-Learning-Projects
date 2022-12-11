#!/usr/bin/python

import sys

oldKey = None
awayTeam = None
pitcherFirstName = None
pitcherLastName = None
outs = 0

print "away team \t home team \t first name \t last name \t highest outs"
for line in sys.stdin:
    #parse the input we got from mapper
    data = line.strip().split("\t")
    if len(data) != 5:
        #something has gone wrong, skip this line.
        continue

    thisKey,thisAwayTeam,thisPitcherFirstName,thisPitcherLastName,thisOuts= data 
    
    #check if the key has changed 
    if oldKey and oldKey != thisKey:
        #write result to STDOUT
        print oldKey,"\t",awayTeam,"\t",pitcherFirstName,"\t",pitcherLastName,"\t",outs
        oldKey = thisKey;
        awayTeam = None
        pitcherLastName= None
        pitcherFirstName = None
        outs = 0
    else : 
        (oldKey,awayTeam,pitcherFirstName,pitcherLastName,outs)=(thisKey,thisAwayTeam,thisPitcherFirstName,thisPitcherLastName,max(outs,int(thisOuts)))
    

if awayTeam != None:
    print oldKey,"\t",awayTeam,"\t",pitcherFirstName,"\t",pitcherLastName,"\t",outs