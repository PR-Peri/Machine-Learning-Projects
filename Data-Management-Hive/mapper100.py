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
    pitcherfirstname = data[50]
    pitcherlastname= data[51]
    pitcherthrowhand= data[52]

    #check to see if there are 6 elements, if not skip
    if len(data) == 145:
        
        #this next line uses 'multiple assignment' in Python, 
        #assigning the parsed data into its own variables
        data = pitcherfirstname, pitcherlastname, pitcherthrowhand
        if((pitcherfirstname[:1]=="R") or (pitcherfirstname[:1]=="P")):
            print pitcherfirstname, "\t", pitcherlastname, "\t", pitcherthrowhand
