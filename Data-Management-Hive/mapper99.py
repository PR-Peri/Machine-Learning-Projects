#!/usr/bin/python

# format of each line is:
# we want elements 5 (data)  
# we need to write them out to standard output, separated by a tab
# this is the mapper
from itertools import groupby 

import sys
for line in sys.stdin:

    line = line.strip()
    data = line.split(",")
    
    hitterLastName = data[44]
    hitterFirstName = data[45]
    hitterWeight = data[46]
    hitterHeight = data[47]
    hitterBatHand = data[48]
    
    #check to see if there are 6 elements, if not skip
    if len(data) == 145:
        
        #this next line uses 'multiple assignment' in Python, 
        #assigning the parsed data into its own variables
        data =  hitterFirstName,hitterLastName, hitterWeight, hitterHeight, hitterBatHand
        if((hitterFirstName[:1]=="R") or (hitterFirstName[:1]=="P")):
            print hitterFirstName, "\t", hitterLastName,"\t", hitterWeight,"\t", hitterHeight,"\t", hitterBatHand
