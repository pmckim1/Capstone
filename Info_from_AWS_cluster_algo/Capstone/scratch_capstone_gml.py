#%%
import igraph

#%%
g = igraph.read("hier_cluster_igraph.gml")


# %%
import pandas as pd

pickle_01 = pd.read_pickle("hier_cluster_igraph.pickle")

# %%
