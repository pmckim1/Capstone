#!/usr/bin/env python3
"""Process the igraph for the news project.
by Tom Nicholls, 2013-2018.

This work is in the public domain, under the terms of the CC0 dedication:
http://creativecommons.org/publicdomain/zero/1.0/
"""
#####
#IMPORTS
#####

import pickle
#import networkx as nx
import sys
import igraph
from pprint import pprint
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

#####
#PARAMETERS
#####

#index_dir = "WhooshIndex"
#offsets_file = "Output/hier_infomap_out_final/offsets.pickle"
comm_plus_kw_file = "Output/hier_infomap_out_final/comm_plus_kw.pickle"
#hier_comm_plus_kw_file = "Output/hier_infomap_out_final/hier_comm_plus_kw.pickle"
hier_igraph_out = "Output/hier_infomap_out_final/hier_cluster_igraph.pickle"
hier_igraphml_out = "Output/hier_infomap_out_final/hier_cluster_igraph.gml"



#####
#CLASSES
#####
class print_counter(object):
    """Print an incrementing counter (by default to sys.stderr)"""
    def __init__(self, stream=sys.stderr):
        self.n = 0
        self.stream = stream
    def reset(self):
        self.n = 0
        self.stream.write('\n')
        self.stream.flush()
    def click(self):
        self.n += 1
        self.stream.write('\r'+str(self.n))
        self.stream.flush()


#####
#PROCEDURES AND FUNCTIONS
#####
def make_skeleton_igraph(g, key, childkey=None):
    # Iterate back from leaf nodes to build a whole forest.
    # Node names are split on ':'. Naive, but fast enough
    # for graphs with only a few 10k nodes.
#    print key, childkey
    foundtree = False
    try:
        # Does the node with this key exist?
        v = g.vs.find(key)
        foundtree = True
    except ValueError:
        v = g.add_vertex(name=key)
#        print "New vertex:", key
    if childkey:
        g.add_edge(key, childkey)
    if foundtree or ':' not in key:
        # Attached to a branch, or we are the trunk node for our tree
        return g
    else:
        # Still a floating twiglet
        x = key.rsplit(':', 1)
        g = make_skeleton_igraph(g, x[0], key)
        return g

def get_sub_art(k, v):
    urls = set()
    if not int(k):
        # Leaf node
        urls.update(set(v['articles'].keys()))
    else:
        for k, v in v.items():
            urls.update(get_sub_art(k, v))
    return urls
        

#####
#DATA SETUP
#####

print("Unpickling communities/keywords file from "+comm_plus_kw_file+"...")
with open(comm_plus_kw_file, 'rb') as f:
     c = pickle.load(f)
print("...done.")


cg = igraph.Graph(directed=True)
for k in tqdm(c.keys()):
    cg = make_skeleton_igraph(cg, k)

print("Dumping igraph files...")
cg.write_pickle(hier_igraph_out)
cg.write_graphml(hier_igraphml_out)
print("...done.")





