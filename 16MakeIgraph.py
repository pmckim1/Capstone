#!/usr/bin/env python3
"""Assemble an igraph for the news project.
by Tom Nicholls, 2013-2015.

This work is in the public domain, under the terms of the CC0 dedication:
http://creativecommons.org/publicdomain/zero/1.0/
"""
#####
#IMPORTS
#####

import pickle
#import networkx as nx
import igraph

#####
#PARAMETERS
#####

index_dir = "WhooshIndex"
offsets_file = "Output/hier_infomap_out_final/offsets.pickle"
communities_file = "Output/hier_infomap_out_final/communities.pickle"
pajek_file = "Output/production_output_final.net"
#networkx_file = "Output/hier_infomap_out_final/networkx.pickle"
igraph_file = "Output/hier_infomap_out_final/igraph.pickle"
igraphml_file = "Output/hier_infomap_out_final/news.graphml"


#####
#DATA SETUP
#####
# offsets = [datetime.timedelta, datetime.timedelta, ...]
print("Importing offsets from "+offsets_file+"...")
with open(offsets_file, 'rb') as f:
    offsets = pickle.load(f)

print("...done.")

print("Importing communities from "+communities_file+"...")
with open(communities_file, 'rb') as f:
    communities = pickle.load(f)

print("...done.")

print("Importing network from "+pajek_file+"...")
#g = nx.Graph(nx.read_pajek(pajek_file))
i = igraph.Graph.Read(pajek_file, 'pajek')
print("...done.")

for c, x in communities.items():
#    for n in g.nodes_iter():
#        pass
    for url in x.keys():
        n = i.vs.find(id=url)
        n['hier_infomap_comm'] = c
        n['url'] = url

#vc = igraph.clustering.VertexClustering.FromAttribute(i, 'hier_infomap_comm')

# Assign infomap cluster labels to vertices
print("Dumping igraph files...")
#nx.write_gpickle(g, networkx_file)
i.write_pickle(igraph_file)
i.write_graphml(igraphml_file)
print("...done.")

