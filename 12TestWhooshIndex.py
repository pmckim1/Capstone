#!/usr/bin/env python3
"""Test the the Whoosh index.
by Tom Nicholls, 2013-2015.

This work is in the public domain, under the terms of the CC0 dedication:
http://creativecommons.org/publicdomain/zero/1.0/
"""
#####
#IMPORTS
#####

import csv
#import whoosh.analysis
import whoosh.index
import whoosh.fields
#import whoosh.qparser
#import whoosh.classify
import whoosh.searching
from collections import OrderedDict

#####
#PARAMETERS
#####

index_dir = "WhooshIndex"



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

def nice_dict_format(d):
    """Pretty format dictionaries into a multi-line string"""
    return ''.join([key+": "+str(d[key])+"\n" for key in list(d.keys())])

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
    print(r.doc_count())
    comms = [x[0] for x in r.iter_field("infomap_hier_community")]
    print(sorted(comms))
    print(len(comms))

