#%%
#!/usr/bin/env python3
"""Assemble a Networkx graph for the news project.
by Tom Nicholls, 2013-2018.

This work is in the public domain, under the terms of the CC0 dedication:
http://creativecommons.org/publicdomain/zero/1.0/
"""
#####
#IMPORTS
#####

import pickle
#import networkx as nx
import igraph
import nltk
import whoosh.analysis
import whoosh.index
import whoosh.fields
import whoosh.qparser
import whoosh.classify
import sys
from pprint import pprint
from tqdm import tqdm

#####
#PARAMETERS
#####

index_dir = "WhooshIndex"
communities_file = "Output/hier_infomap_out_final/communities.pickle"
hier_igraph_file = "Output/hier_infomap_out_final/hier_cluster_igraph.pickle"
hier_igraph_kw_file = "Output/hier_infomap_out_final/hier_cluster_kw_igraph.pickle"
hier_igraph_kw_file_graphml = "Output/hier_infomap_out_final/hier_cluster_kw_igraph.graphml"



#####
#CLASSES:
#####
class ShoutingFilter(whoosh.analysis.Filter):
    """Strips words in ALL CAPS.

    >>> ana = RegexTokenizer() | ShoutingFilter()
    >>> [token.text for token in ana("HELLO there Bob")]
    ["there", "Bob"]
    """
    def __call__(self, tokens):
        for t in tokens:
            if t.text != t.text.upper():
                yield t

#####
#DATA SETUP
#####
# ix: A whoosh index of the contents of each document in the corpus
#        schema = whoosh.fields.Schema(
#            url=whoosh.fields.ID(stored=True, unique=True),
#            paperurl=whoosh.fields.ID(stored=True),
#            title=whoosh.fields.TEXT(stored=True,
#                                     phrase=False,
#                                     analyzer=analyser,
#                                     field_boost=opt["title_boost"]),
#            text=whoosh.fields.TEXT(stored=True,
#                                    phrase=False,
#                                    analyzer=analyser),
#            time=whoosh.fields.DATETIME(stored=True))
print("Importing index...")
ix = whoosh.index.open_dir(index_dir)

with ix.reader() as r:
    print("There are", r.doc_count(), "documents in the index.")

print("Unpickling network from "+hier_igraph_file+" and communities from "+communities_file+"...")
g = igraph.Graph().Read_Pickle(hier_igraph_file)
with open(communities_file, 'rb') as f:
     communities = pickle.load(f)
print("...done.")


print("Traversing network and assembling URLs...")
artd = {}
# Iterate through the vertex list; for each vertex, collect the URLs that
# are attached to children of that vertex (by ID name)
for vn in tqdm(g.vs["name"]):
    artd[vn] = set()
    for k in communities:
        if k.startswith(vn):
            artd[vn] |= set(communities[k].keys())
print("...done.")

cluster_titled = {}
print("Extracting URLs for nodes and adding them to the graph...")
with ix.searcher() as s:
    # Iterate through the igraph vertices (articles) and label with their own title
    # and the title of the median article in their cluster
    for node in tqdm(g.vs):
        # Generate list of community dicts, whose cluster matches (or is a
        # sub-cluster of) the node given
        salient_communities = [v for k, v in communities.items() if k.startswith(node['name'])]

        salient_articles = {}
        for d in salient_communities:
            salient_articles.update(d)

        # Sort by time and take the median item, keeping only the url (tuple part 0)
        median_url = sorted(iter(salient_articles.items()), key=lambda x: x[1]['time'])[(len(salient_communities)-1) // 2][0]

        # Look up the URL and extract the stored data
#        node_title = s.stored_fields(s.document_number(url=node['url']))['title']
        cluster_title = s.stored_fields(s.document_number(url=median_url))['title']

#        pprint(node_title)

        node['central_art_title'] = cluster_title
        node['size'] = len(salient_articles)
print("...done.")

kwd = {}
print("Extracting community keywords...")
with ix.searcher() as s:
    # Get keywords for each story
    for name, urls in tqdm(artd.items()):
        ixnos = [s.document_number(url=x) for x in urls]
        # Note that this uses the Bo1 model for scoring keywords
        kwd[name] = [kw for kw, score in s.key_terms(ixnos, "text", numterms=10)]
print("...done.")


print("Adding them to the graph...")
for x in tqdm(g.vs):
    if x["name"] in kwd:
        x["keywordl"] = ', '.join(kwd[x["name"]])
    else:
        print("No keywords found for", x['name'])
print("...done.")



print("Dumping igraph files...")
g.write_pickle(hier_igraph_kw_file)
g.write_graphml(hier_igraph_kw_file_graphml)
print("...done.")



# %%
