#!/usr/bin/env python
"""A more advanced Reducer, using Python iterators and generators."""
import sys
totalBalls =0
highestBalls = 0
previousID = None
previousName = None
print "hitterFirstName \t hitterLastName \t balls"
lines = sys.stdin.readlines()
for line in lines:

    data = line.strip().split("\t")
    if len(data) <=2:
        continue
        
    hitterFirstName, hitterLastName, balls = data
    if previousID and previousID != hitterLastName:
        if (totalBalls > highestBalls):
            highestBalls = totalBalls
        totalBalls = 0    
    previousID = hitterLastName
    if (balls == ""):
        balls = 0
    totalBalls += int (balls)
    
for line in lines:
    data = line.strip().split("\t")
    if len(data) <=2:
        continue
        
    hitterFirstName, hitterLastName, balls = data
    if previousID and previousID != hitterLastName:
        if (totalBalls == highestBalls):
            print previousName, "\t",previousID,"\t",highestBalls  
        totalBalls = 0   
    previousID = hitterLastName
    previousName = hitterFirstName
    if (balls == ""):
        balls = 0
    totalBalls += int (balls)
