#!/usr/bin/env python3
"""Make keywords for each cluster, for the news project.
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
import nltk
import whoosh.analysis
import whoosh.index
import whoosh.fields
import whoosh.qparser
import whoosh.classify
import sys
import pprint

#####
#PARAMETERS
#####

index_dir = "WhooshIndex"
#offsets_file = "Output/hier_infomap_out_final/offsets.pickle"
communities_file = "Output/hier_infomap_out_final/communities.pickle"
comm_plus_kw_file = "Output/hier_infomap_out_final/comm_plus_kw.pickle"


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

print("Unpickling communities file from "+communities_file+"...")
with open(communities_file, 'rb') as f:
     communities = pickle.load(f)
print("...done.")

comm_plus_kw = {x: {"articles": y} for x, y in communities.items()}
#keywords = {}

print("Extracting and pickling community keywords...")
pc = print_counter()
with ix.searcher() as s:
    # Get keywords for each story
    for story, d in communities.items():
        ixnos = [s.document_number(url=x) for x in d]
        # Note that this uses the Bo1 model for scoring keywords
        comm_plus_kw[story]["keywords"] = keywords_and_scores = s.key_terms(ixnos, "text", numterms=20)
        pc.click()



pprint.pprint(comm_plus_kw)

with open(comm_plus_kw_file, 'wb') as f:
     communities = pickle.dump(comm_plus_kw, f)
print("...done.")


