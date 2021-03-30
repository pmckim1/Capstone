#!/usr/bin/env python3
"""Pull out and save the time offsets for each story in each cluster.
by Tom Nicholls, 2013-2015.

This work is in the public domain, under the terms of the CC0 dedication:
http://creativecommons.org/publicdomain/zero/1.0/
"""
#####
#IMPORTS
#####

import sys
import pickle

#####
#PARAMETERS
#####

comm_file = "Output/hier_infomap_out_final/communities.pickle"
offsets_file = "Output/hier_infomap_out_final/offsets.pickle"

#####
#MAIN
#####

with open(comm_file, 'rb') as f:
    c = pickle.load(f)

offsets = {}

for comm in c:
    times = []
    for url in c[comm]:
        times.append(c[comm][url]['time'])
    earliest = min(times)
    times = [x-earliest for x in times]
    offsets[comm] = sorted(times)

with open(offsets_file, 'wb') as f:
    pickle.dump(offsets, f)


