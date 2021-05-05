#!/usr/bin/env python3
"""Label the igraph for the news project.
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
import whoosh.analysis
import whoosh.index
import whoosh.fields
import whoosh.qparser
import whoosh.classify
import whoosh.sorting
from pprint import pprint
from tqdm import tqdm

#####
#PARAMETERS
#####

communities_file = "Output/hier_infomap_out_final/communities.pickle"
igraph_file = "Output/hier_infomap_out_final/igraph.pickle"
index_dir = "WhooshIndex"

igraph_out = "Output/hier_infomap_out_final/igraphPlusLabel.pickle"
igraphml_out = "Output/hier_infomap_out_final/igraphPlusLabel.gml"


#####
#CLASSES
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
#PROCEDURES
#####

# TODO: In principle, the code here is very generalisable and should be reused.
#get_title_by_cluster


#####
#DATA SETUP
#####
print("Unpickling network from "+igraph_file+" and communities from "+communities_file+"...")
i = igraph.Graph().Read_Pickle(igraph_file)
with open(communities_file, 'rb') as f:
    c = pickle.load(f)
print("...done.")

print("Importing index and setting 'time' as sortable...")
ix = whoosh.index.open_dir(index_dir)
with ix.writer() as w:
    try:
        # Make "time" field sortable
        whoosh.sorting.add_sortable(w, "time", whoosh.sorting.FieldFacet("time"))
    except Exception:
        # XXX This is very clumsy, but there doesn't seem to be an API way of
        # checking sortability, and adding sortability to an already sortable
        # field raises a bare Exception rather than a more specific error type
        # or being idempotent.
        print("('time' already seems to be sortable)")
with ix.reader() as r:
    print("There are", r.doc_count(), "documents in the index.")
print("...done.")



with ix.searcher() as s:
    # Iterate through the igraph vertices (articles) and label with their own title
    # and the title of the median article in their cluster
    for node in tqdm(i.vs):
    # for node in i.vs:
        # Generate list of community dicts, whose cluster matches (or is a
        # sub-cluster of) the node given
        # pprint(node)
        hier_infomap_comm_value = node['hier_infomap_comm']
        salient_communities = [v for k, v in c.items() if
                               k is not None and
                               hier_infomap_comm_value is not None
                               and k.startswith(hier_infomap_comm_value)
                               ]

        salient_articles = {}
        for d in salient_communities:
            salient_articles.update(d)

#        pprint(salient_articles)

        # Sort by time and take the median item, keeping only the url (tuple part 0)
        if len(salient_communities) <= 0:
            continue
        try:
            median_url = sorted(
                iter(
                    salient_articles.items()
                ),
                key=lambda x: x[1]['time']
            )[(len(salient_communities)-1) // 2][0]
        except Exception as e:
            print("It broke again:")
            print("salient_articles", len(salient_articles), salient_articles)
            print("salient_communities", len(salient_communities), salient_communities)
            raise
#        pprint(median_url)

        # Look up the URL and extract the stored data
        node_title = s.stored_fields(s.document_number(url=node['url']))['title']
        cluster_title = s.stored_fields(s.document_number(url=median_url))['title']

#        pprint(node_title)
#        pprint(cluster_title)

        node['title'] = node_title
        # XXX: This is a slightly unfortunate way of doing it: We are writing
        # the 'median' title of each leaf cluster to the articles contained
        # within it. We should probably be doing this by subgraphing / merging
        # nodes in the igraph, by producing an overlay graph of the infomap
        # hierarchy (with cluster titles attached to the overlay nodes) or ...

        # TODO: Yes, this wants an overlap graph producing
        node['cluster_title'] = cluster_title


print("Dumping igraph files...")
i.write_pickle(igraph_out)
i.write_graphml(igraphml_out)
print("...done.")


