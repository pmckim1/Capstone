#!/usr/bin/env python3
"""Select sample clusters from the Whoosh index for validation.
by Tom Nicholls, 2018.

This work is in the public domain, under the terms of the CC0 dedication:
http://creativecommons.org/publicdomain/zero/1.0/
"""
#####
#IMPORTS
#####

import csv
import random
#import whoosh.analysis
import whoosh.index
import whoosh.fields
#import whoosh.qparser
#import whoosh.classify
import whoosh.searching
from pprint import pprint, pformat

#####
#PARAMETERS
#####

index_dir = "WhooshIndex"
n = 30
outfile = "ClusterVerification/clusters.txt"



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
#FUNCTIONS AND PROCEDURES:
#####


#####
#DATA SETUP
#####
# ix: A whoosh index of the contents of each document in the corpus
#        schema = whoosh.fields.Schema(
#            url=whoosh.fields.ID(stored=True),
#            paperurl=whoosh.fields.ID(stored=True),
#            title=whoosh.fields.TEXT(stored=True,
#                                     phrase=False,
#                                     analyzer=analyser,
#                                     field_boost=opt["title_boost"]),
#            text=whoosh.fields.TEXT(stored=True,
#                                    phrase=False,
#                                    analyzer=analyser),
#            time=whoosh.fields.DATETIME(stored=True),
#            infomap_community=whoosh.fields.NUMERIC,
#            infomap_community_size=whoosh.fields.NUMERIC)
print("Importing index...")
ix = whoosh.index.open_dir(index_dir)

with ix.reader() as r:
    ids = list(r.all_doc_ids())

    # For reproducability
    random.seed("""But I don't need your practiced lines,
                   your school of charm mentality, so...""")
    clusters = set([r.stored_fields(y)['infomap_hier_community'] for y
                    in random.sample(ids, n)])

    print(clusters)
    output = {}
    with ix.searcher() as s:
        for c in clusters:
            output[c] = list([x['url'] for x in
                                s.documents(infomap_hier_community=c)])

    with open(outfile, 'w') as f:
        f.write(pformat(output))

