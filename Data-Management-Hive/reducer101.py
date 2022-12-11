#!/usr/bin/env python
"""A more advanced Reducer, using Python iterators and generators."""
import sys
totalStrikes =0
highestStrikes = 0
previousID = None
previousName = None
print "hitterFirstName \t hitterLastName \t strikes"
lines = sys.stdin.readlines()
for line in lines:

    data = line.strip().split("\t")
    if len(data) <=2:
        continue 
        
    hitterFirstName, hitterLastName, strikes = data
    if previousID and previousID != hitterLastName:
        if (totalStrikes > highestStrikes):
            highestStrikes = totalStrikes
        totalStrikes = 0   
    previousID = hitterLastName

    if (strikes == ""):
        strikes = 0
    totalStrikes += int (strikes)
    
for line in lines:
    data = line.strip().split("\t")
    if len(data) <=2:
        continue
        
    hitterFirstName, hitterLastName, strikes = data
    if previousID and previousID != hitterLastName:
        if (totalStrikes == highestStrikes):
            print previousName, "\t",previousID,"\t",highestStrikes 
        totalStrikes = 0  
    previousID = hitterLastName
    previousName = hitterFirstName
    if (strikes == ""):
        strikes = 0
    totalStrikes += int (strikes)
