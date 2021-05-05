#!/usr/bin/env python3
"""Classify pairs of articles as being part of the same "story" or not.
by Tom Nicholls, 2013-2015.

This work is in the public domain, under the terms of the CC0 dedication:
http://creativecommons.org/publicdomain/zero/1.0/
"""
#####
#IMPORTS
#####

import csv
import datetime

#####
#PARAMETERS
#####

nodes_file = "Output/production_nodes_final.tsv"
links_file = "Output/production_matches_final.tsv"
output_file = "Output/production_output_final.net"
weights_file = "Output/production_weights_final.tsv"

#####
#DATA SETUP
#####

node_no_to_url = {}
url_to_node_no = {}
edges = []

with open(nodes_file, 'r') as f:
    next(f) # Skip header
    reader = csv.reader(f, delimiter="\t")
    n = 0
    for row in reader:
        n += 1
        node_no_to_url[n] = row[0]
        url_to_node_no[row[0]] = n

with open(links_file, 'r') as f:
    next(f) # Skip header
    reader = csv.reader(f,delimiter="\t")
    for row in reader:
        edges.append(tuple(row))

with open(output_file, 'w') as f:
    d = datetime.datetime.now()
#    f.write("#News article pairs output file, "+d.isoformat()+"\n")
    f.write("*Vertices "+str(len(node_no_to_url))+"\n")
    for k, v in node_no_to_url.items():
        f.write(str(k)+' "'+v+'"\n')
    f.write("*Edges "+str(len(edges))+"\n")
    for row in edges:
        try:
            node_1 = url_to_node_no[row[0]]
            node_2 = url_to_node_no[row[1]]
            f.write(str(node_1)+" "+str(node_2)+" "+row[2]+"\n")
        except KeyError:
            print("KeyError: %s, %s" % (str(node_1), str(node_2)))

with open(weights_file, 'w') as f:
    for row in edges:
        f.write(row[2]+"\n")

