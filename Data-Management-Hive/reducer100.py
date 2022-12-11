#!/usr/bin/env python
"""A more advanced Reducer, using Python iterators and generators."""
import sys

oldPitcherFname = None
oldPitcherLname = None
hand = None
print "pitcherfirstname \t pitcherlastname \t pitcherthrowhand"
for line in sys.stdin:
    
    data = line.strip().split("\t")
    if len(data)!=3:
            continue    
    dup_pitcherFname ,dup_pitcherLname, pitcherhand = data

    if (oldPitcherFname and oldPitcherFname != dup_pitcherFname)and (oldPitcherLname and oldPitcherLname != dup_pitcherLname):
        print oldPitcherFname,"\t", oldPitcherLname, "\t", hand
        oldPitcherFname = dup_pitcherFname;
        oldPitcherLname = dup_pitcherLname;
        hand = None
                
    oldPitcherFname = dup_pitcherFname
    oldPitcherLname = dup_pitcherLname;
    hand = pitcherhand
        
if oldPitcherFname and oldPitcherLname != None:
    print oldPitcherFname, "\t", oldPitcherLname, "\t" , hand    