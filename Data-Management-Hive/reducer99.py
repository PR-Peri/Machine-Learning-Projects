#!/usr/bin/env python
"""A more advanced Reducer, using Python iterators and generators."""
import sys

oldHitterFname = None
oldHitterLname = None
weight = None
height = None
bathand = None
print "hitterFirstName \t hitterLastName \t hitterWeight \t hitterHeight \t hitterBatHand"
for line in sys.stdin:
    
    data = line.strip().split("\t")
    if len(data)!=5:
            continue
        
    dup_hitterFname , dup_hitterLname, hitterWeight, hitterHeight, hitterBatHand = data
    if (oldHitterFname and oldHitterFname != dup_hitterFname) and (oldHitterLname and oldHitterLname != dup_hitterLname):
        print oldHitterFname, "\t", oldHitterLname, "\t", weight, "\t", height, "\t", bathand
        oldHitterFname = dup_hitterFname;
        oldHitterLname = dup_hitterLname;
        weight = None
        height = None
        bathand = None
             
    oldHitterFname = dup_hitterFname
    oldHitterLname = dup_hitterLname
    weight = hitterWeight
    height = hitterHeight
    bathand = hitterBatHand
        
if oldHitterFname and oldHitterLname != None:
    print oldHitterFname, "\t", oldHitterLname, "\t", weight, "\t", height, "\t", bathand
    
