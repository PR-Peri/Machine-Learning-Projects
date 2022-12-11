#!/usr/bin/env python
"""A more advanced Reducer, using Python iterators and generators."""
import sys
totallOuts =0
highestOuts = 0
previousID = None
previousName = None
print "hitterFirstName \t hitterLastName \t outs"
lines = sys.stdin.readlines()
for line in lines:
    data = line.strip().split("\t")
    if len(data) <=2:
        continue
        
    hitterFirstName, hitterLastName, outs = data    
    if previousID and previousID != hitterLastName:
        if (totallOuts > highestOuts):
            highestOuts = totallOuts
        totallOuts = 0        
    previousID = hitterLastName
    
    if (outs == ""):
        outs = 0   
    totallOuts += int (outs)
    
for line in lines:
    data = line.strip().split("\t")
    if len(data) <=2:
        continue        
    hitterFirstName, hitterLastName, outs = data
    
    if previousID and previousID != hitterLastName:
        if (totallOuts == highestOuts):
            print previousName, "\t",previousID,"\t",highestOuts            
        totallOuts = 0      
    previousID = hitterLastName
    previousName = hitterFirstName
    
    if (outs == ""):
        outs = 0   
    totallOuts += int (outs)
