#%%
#!/usr/bin/env python3
"""Make graphs for the news story project.
by Tom Nicholls, 2013-2015.

This work is in the public domain, under the terms of the CC0 dedication:
http://creativecommons.org/publicdomain/zero/1.0/
"""
#####
#IMPORTS
#####

import pickle
import matplotlib.pyplot as plt
import igraph
import pprint
from tqdm import tqdm

#####
#PARAMETERS
#####

offsets_file = "Output/hier_infomap_out_final/offsets.pickle"
igraph_file = "Output/hier_infomap_out_final/igraph.pickle"

#####
#DATA SETUP
#####
# offsets = [datetime.timedelta, datetime.timedelta, ...]
print("Importing offsets from "+offsets_file+"...")
with open(offsets_file, 'rb') as f:
    offsets = pickle.load(f)


print("...done.")

print("Importing igraph from "+igraph_file+"...")
with open(igraph_file, 'rb') as f:
    i = pickle.load(f)

print("...done.")


vc = igraph.clustering.VertexClustering.FromAttribute(i, 'hier_infomap_comm')
sg = vc.cluster_graph()

#for k in offsets:
#    plt.plot([(x.total_seconds()/3600) for x in offsets[k]], range(1, len(offsets[k])+1))#, alpha=0.1)
#plt.xlabel('Hours after first article publication')
#plt.ylabel('Number of articles published')
##    plt.xscale("log")
##    plt.yscale("log")
#plt.show()

d = []
for k in tqdm(offsets):
    # TODO: Lookup the cluster size of cluster k to decide how to split the
    # graph
    d.extend([(x.total_seconds()/3600) for x in offsets[k][1:]])
# Don't use first entry in each story (which is 
print(sorted(d)[int(len(d)/2)])
plt.hist(d, 300)
#plt.hist([x/2 for x in d], 50)

plt.title('Declining story attention over time')
plt.xlabel('Time published after story\'s first article')
plt.ylabel('Number of articles')
plt.show()
#for k in offsets:
#    plt.plot(offsets[k], range(1, len(offsets[k])))
#plt.show()
#    print k


# %%
