#!/usr/bin/env python
"""A more advanced Reducer, using Python iterators and generators."""
import sys
totallPitchCount =0
highestPitchCount = 0
previousID = None
previousName = None
print "pitcherfirstname \t pitcherlastname \t pitcherPitchCount"
lines = sys.stdin.readlines()
for line in lines:

    data = line.strip().split("\t")
    if len(data) <=2:
        continue
        
    pitcherFirstName, pitcherLastName, PitchCount = data 
    if previousID and previousID != pitcherLastName:
        if (totallPitchCount > highestPitchCount):
            highestPitchCount = totallPitchCount
        totallPitchCount = 0      
    previousID = pitcherLastName
    
    if (PitchCount == ""):
        PitchCount = 0 
    totallPitchCount += int (PitchCount)
    
for line in lines:
    data = line.strip().split("\t")
    if len(data) <=2:
        continue  
        
    pitcherFirstName, pitcherLastName, PitchCount = data
    if previousID and previousID != pitcherLastName:
        if (totallPitchCount == highestPitchCount):
            print previousName, "\t",previousID,"\t",highestPitchCount    
        totallPitchCount = 0    
    previousID = pitcherLastName
    previousName = pitcherFirstName
    if (PitchCount == ""):
        PitchCount = 0
    totallPitchCount += int (PitchCount)
