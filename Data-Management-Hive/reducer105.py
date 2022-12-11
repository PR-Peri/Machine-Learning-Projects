#!/usr/bin/env python
"""A more advanced Reducer, using Python iterators and generators."""
import sys
totallpitchSpeed =0
highestpitchSpeed = 0
previousID = None
previousName = None
print "pitcherfirstname \t pitcherlastname \t pitchSpeed"
lines = sys.stdin.readlines()

for line in lines:
    data = line.strip().split("\t")
    if len(data) <=2:
        continue
        
    pitcherFirstName, pitcherLastName, pitchSpeed = data
    if previousID and previousID != pitcherLastName:
        if (totallpitchSpeed > highestpitchSpeed):
            highestpitchSpeed = totallpitchSpeed
        totallpitchSpeed = 0   
    previousID = pitcherLastName
    if (pitchSpeed == ""):
        pitchSpeed = 0
    totallpitchSpeed += int (pitchSpeed)
    
for line in lines:
    data = line.strip().split("\t")
    if len(data) <=2:
        continue
        
    pitcherFirstName, pitcherLastName, pitchSpeed = data
    if previousID and previousID != pitcherLastName:
        if (totallpitchSpeed == highestpitchSpeed):
            print previousName, "\t",previousID,"\t",highestpitchSpeed      
        totallpitchSpeed = 0 
        
    previousID = pitcherLastName
    previousName = pitcherFirstName
    if (pitchSpeed == ""):
        pitchSpeed = 0
    totallpitchSpeed += int (pitchSpeed)
