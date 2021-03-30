#!/usr/bin/env python3
""".

This work is in the public domain, under the terms of the CC0 dedication:
http://creativecommons.org/publicdomain/zero/1.0/
"""
#####
#IMPORTS
#####

import numpy as np
import sys
import datetime
import time
from tqdm import tqdm
from pprint import pprint
import csv
from krippendorff import alpha
import whoosh.index
import whoosh.fields
#import whoosh.qparser
#import whoosh.classify
import whoosh.searching

#####
#PARAMETERS
#####

index_dir = "WhooshIndex"
mappings_file_a = "HandCoded/story_pairs_validation_august.csv"
mappings_file_b = "HandCoded/story_pairs_validation_august_second_coder.csv"

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
#Input processing
#####

def mappings_from_file(fn, test_in_index):
    """Open manual mappings file and write results to list for IRR.
       Value is '1' iff the text in the 3rd col is "Related", otherwise 0."""
    vals = []
    urls = set()
    to_remove = []

    # Get URLs
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            urls.update(set([str(line[0]),str(line[1])]))

    # Filter them, if needed
    if test_in_index:
        with IX.reader() as r:
            # The sample for coding was taken from all articles.
            # The classifier has various exclusions, particularly
            # the requirement of 200 words. So some of our hand
            # codings are actually outside the corpus and should
            # be excluded. So we do that here.
             with IX.searcher() as s:
                for url in urls:
                    if not s.document_number(url=url):
                        to_remove.append(url)
        urls = urls.difference(to_remove)        

    # Extract mappings from each file    
    with open(fn, 'r') as f:
        print("Extracting mappings from", fn)
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            # Skip those not in set
            if str(line[0]) not in urls or str(line[1]) not in urls:
                continue
            relation = str(line[2])
            relvalue = 0
            if relation == "Related":
                relvalue = 1
            vals.append(relvalue)
    return vals


def get_mappings_array(fna, fnb, test_in_index=False):
    """Open an pair of CSVs with manual mappings data. Returns a numpy
    array suitable for feeding into alpha().

    Iff test_in_index, look up the URLs in Whoosh to check we've got them.
    """
    vals_a = mappings_from_file(fna, test_in_index)
    vals_b = mappings_from_file(fnb, test_in_index)
    
    print("Manual mappings a: {}/{} related.".format(len([x for x in vals_a if x]),
                                                     len(vals_a)))
    print("Manual mappings b: {}/{} related.".format(len([x for x in vals_b if x]),
                                                     len(vals_b)))
    
    assert len(vals_a) == len(vals_b)
    return np.array((vals_a, vals_b))

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
IX = whoosh.index.open_dir(index_dir)

# manual_mappings: array of hand codings.
manual_mappings = get_mappings_array(mappings_file_a,
                                     mappings_file_b,
                                     test_in_index=True)

manual_mappings_full = get_mappings_array(mappings_file_a,
                                          mappings_file_b,
                                          test_in_index=False)


#####
#TEST: Calculate Krippendorff's alpha
#####

# Print setup info    
print("***** VALIDATION TEST RESULTS at", time.ctime())

print("Only valid articles:")
print("Coder 1 Related/Total: {}/{}".format(sum(manual_mappings[0]),
                                            len(manual_mappings[0])))
print("Coder 2 Related/Total: {}/{}".format(sum(manual_mappings[1]),
                                            len(manual_mappings[1])))
print("Krippendorff's alpha: {}".format(
        alpha(reliability_data=manual_mappings,
            level_of_measurement='nominal')))
print("")
print("All articles:")
print("Coder 1 Related/Total: {}/{}".format(sum(manual_mappings_full[0]),
                                            len(manual_mappings_full[0])))
print("Coder 2 Related/Total: {}/{}".format(sum(manual_mappings_full[1]),
                                            len(manual_mappings_full[1])))
print("Krippendorff's alpha: {}".format(
        alpha(reliability_data=manual_mappings_full,
            level_of_measurement='nominal')))


