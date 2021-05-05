#!/usr/bin/env python3
"""Merge in community data from Infomap.
by Tom Nicholls, 2013-2015.

This work is in the public domain, under the terms of the CC0 dedication:
http://creativecommons.org/publicdomain/zero/1.0/
"""
#####
#IMPORTS
#####

import csv
import sys
import whoosh.index
import whoosh.fields
import whoosh.searching
from collections import defaultdict
import pickle

#####
#PARAMETERS
#####

communities_file = "Output/hier_infomap_out_final/production_output_final.tree"
failures_file = "Output/hier_infomap_out_final/failures.log"
index_dir = "WhooshIndex"
pickle_file = "Output/hier_infomap_out_final/communities.pickle"

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

with ix.writer() as w:
    for n, t in [("infomap_hier_community",
                    whoosh.fields.ID(stored=True)),
                 ("infomap_hier_flow",
                    whoosh.fields.NUMERIC(stored=True)),
                 ("infomap_hier_community_size",
                    whoosh.fields.NUMERIC(stored=True))]:
        try:
            w.add_field(n, t)
        except whoosh.fields.FieldConfigurationError:
            # Already added
            pass

with ix.reader() as r:
    print("There are", r.doc_count(), "documents in the index.")

communities = defaultdict(dict)
pc = print_counter()
with open(communities_file, 'r') as f:
    # Skip header line
    next(f)
    reader = csv.reader(f, delimiter=' ')
    with ix.writer() as w:
        for line in reader:
            try:
#                print(line)
                node, flow, url, _ = tuple(line)
                # Look up the URL and extract the stored data
                with ix.searcher() as s:
                    d = s.stored_fields(s.document_number(url=url))
                if d is None:
                    raise Exception
                # Discard the leaf node number and store only the rest of the
                # community hierarchy (so the "community" field actually represents
                # the lowest-level community).
                clabel = node.rpartition(':')[0]
                d["infomap_hier_community"] = str(clabel)
                d["infomap_hier_flow"] = float(flow)
                communities[clabel][url] = {'time': d['time']}
    #            d["infomap_hier_community_size"] = size
                w.update_document(**d)

            except Exception as e:
                print("Line Failure, skipping.")
                with open(failures_file, 'a', encoding='utf-8') as ff:
                    ff.write(', '.join(line))
                    ff.write('\n')
                continue
            finally:
                pc.click()
        print("Committing index...")
    print("...done.")


pc.reset()
print("Calculating cluster sizes...")
for k, l in communities.items():
    size = len(l)
    with ix.writer() as w:
        for kurl in l:
            with ix.searcher() as s:
                d = s.stored_fields(s.document_number(url=kurl))
            if d is None:
                raise Exception
            d["infomap_hier_community_size"] = size
            w.update_document(**d)
            pc.click()
#    print("Committing index...")
print("...done.")

print("Pickling communities file to "+pickle_file+"...")
with open(pickle_file, 'wb') as f:
     pickle.dump(dict(communities), f)
print("...done.")


# Times of related documents can probably be had with:
# with ix.searcher() as s:
#     for d in {x["url"]:x["time"] for x in s.documents("infomap_community"=COMMNUM)}:
#         ...
