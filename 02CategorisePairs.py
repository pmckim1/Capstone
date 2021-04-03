#%%
#!/usr/bin/env python3
"""Classify pairs of articles as being part of the same "story" or not.
by Tom Nicholls, 2013-2013.

This work is in the public domain, under the terms of the CC0 dedication:
http://creativecommons.org/publicdomain/zero/1.0/
"""
#####
#IMPORTS
#####

import sys
import nltk
import datetime
import itertools
import time
import pickle
import pprint
import re
import os
import csv
import multiprocessing
import concurrent.futures
import functools
import socket
import matplotlib.pyplot as plt
#import copy
import whoosh.analysis
import whoosh.index
import whoosh.fields
import whoosh.qparser
import whoosh.classify
import whoosh.scoring
from math import sqrt
from math import log
#from collections import defaultdict
from collections import OrderedDict, Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from krippendorff import alpha
import html.parser
import numpy as np

#####
#PARAMETERS
#####

mappings_file = "HandCoded/story_pairs_validation_august.csv"
index_dir = "WhooshIndex"
# file_list = ["ArticleTexts/bbc-out.csv", "ArticleTexts/express-out.csv","ArticleTexts/guardian-out.csv", "ArticleTexts/mail-out.csv","ArticleTexts/mirror-out.csv"]
file_list = ["ArticleTexts/Guar_Small_pol-out.csv"]
# file_list = ["ArticleTexts/bbc-out.csv"]


# Words to be ignored. Note that this is case-sensitive unless opt["to_lower"]
# is turned on. Likewise, opt["reject_stopwords"] without opt["to_lower"]
# will not remove them all.
badwords = {'autoplay', 'cookies', 'Please', 'turn', 'embed'}
# badwords optionally added to rejectwords later.
rejectwords = set()
# Pre-process functions, run on the title/text if opt["pre_process_fns"] = True
pre_process_title_fns = [
    lambda x: x.replace('  | Mail Online', ''),                # Mail
    lambda x: x.replace('BBC News - ', ''),                    # BBC
    lambda x: x.replace(' - Mirror Online', ''),               # Mirror
    lambda x: re.sub(re.compile(r'\|.*\| (Daily Express|guardian\.co\.uk )'), '', x), # Express/Guardian
    lambda x: re.sub(re.compile(r'\| The Sun.*'), '', x)       # Sun
]
pre_process_text_fns = [
    # ALL
    lambda x: x.replace('\n', ' '),
    lambda x: x.replace('\r', ' '),
    # Mail
    lambda x: re.sub(re.compile(r'^ +By +PUBLISHED: +[0-9:]+, [0-9]+ [A-Za-z]+ [0-9]{4}'), '', x),
    lambda x: re.sub(re.compile(r'\| +UPDATED: +[0-9:]+, [0-9]+ [A-Za-z]+ [0-9]{4}'), '', x),
    lambda x: re.sub(re.compile(r'Published by Associated Newspapers.*'), '', x),
    lambda x: re.sub(re.compile(r'The comments below have (not )?been moderated.*'), '', x),
    lambda x: re.sub(re.compile(r'DM\.has\([^\)]\)'), '', x),
    lambda x: re.sub(re.compile(r' More\.\.\. +'), '', x),
    # Guardian
    lambda x: x.replace('Please activate cookies in order to turn autoplay off', ''),
    lambda x: re.sub(re.compile(r"We can't load the discussion.*"), '', x),
    lambda x: re.sub(re.compile(r'guardian.co.uk today is our daily snapshot.*'), '', x),
    lambda x: re.sub(re.compile(r'To contact the MediaGuardian news desk email.*'), '', x),
    lambda x: re.sub(re.compile(r' {10} .by.*Search the Guardian bookshop.*'), '', x),
    # Express
    lambda x: re.sub(re.compile(r'(\r.*)?{%#o.Comment.comment%}'), '', x),
    lambda x: re.sub(re.compile(r'([A-Za-z\-]+ ){,8} {21,22}'), '', x),
    # Mirror
    lambda x: re.sub(re.compile(r"This website uses 'cookies' to give you the best most relevant experience. Using this website means you.re Ok with this\. You can change which cookies are set at any time - and find out more about them - by following this +\(or by clicking the cookie link at the top of any page\)\."), '', x),
    lambda x: re.sub(re.compile(r"You've turned off story recommendations.*"), '', x),
    lambda x: re.sub(re.compile(r"         View gallery  .*"), '', x),
    # BBC
    lambda x: x.replace(r"We use cookies to ensure that we give you the best experience on our website. If you continue without changing your settings, we'll assume that you are happy to receive all cookies on the BBC website. However, if you would like to, you can   at any time.", ''),
    lambda x: x.replace(r'Please turn on JavaScript.  Media requires JavaScript to play.', ''),
    lambda x: re.sub(re.compile(r"You're using the Internet Explorer 6 browser to view the BBC website.*"), '', x),
    lambda x: re.sub(re.compile(r"The BBC is not responsible for the content of external websites.*"), '', x),
    lambda x: re.sub(re.compile(r"Comments [0-9]+ of [0-9]+.*"), '', x)
]

opt = OrderedDict()
## Tweakable parameters - pre-processing of articles
# Input encoding
opt["input_encoding"] = 'utf-8'
# Run the pre-process regex functions on the text
opt["pre_process_fns"] = True
# Remove standard stopwords from the corpora.
opt["reject_stopwords"] = False
# Remove words in the "badwords" set from the corpora.
opt["reject_badwords"] = True
# Replace "." with ". " in texts.
opt["pad_fullstops"] = True
# Force all words to lowercase.
opt["to_lower"] = False
# Discard all fully-uppercase words.
opt["ignore_shouting"] = True
# Tokeniser, to ensure consistent processing and wordcount in different places
opt["tokeniser"] = whoosh.analysis.RegexTokenizer()

## Tweakable parameters - both classifiers
# Number of days either side of publication to check.
opt["days_window"] = 7
# Manual mappings for validation. Remainder for testing.
opt["prop_validate"] = 1
# Minimum length of article (in words) for it to be considered for matching.
# Must rebuild the index if this changes.
opt["min_length"] = 200
# If the final "score" for a match is > 1, the classifier believes, on balance
# that these are matched articles (the range is from 0 upwards, with each
# component classifier arranged to provide 1 for a marginal match but with no
# top limit).
# This options selects which links are output by the software as part of the
# final matchings. The sensible values for this option are either 0, to get all
# pairs and link strengths no matter how weak, or 1 to select only those that
# are in some sense felt to be "real" links.
opt["final_match_threshold"] = 1

## Tweakable parameters - BM25F classifier
# Minimum score on the other classifier needed for attempting IR class.
opt["cc_threshold"] = 0.5
# Query Expansion model
opt["query_expansion_model"] = whoosh.classify.Bo1Model
# The free parameter B for the BM25F scoring model. Default = 0.75.
opt["bm25f_b"] = 0.75
# The free parameter K1 for the BM25F scoring model. Default = 1.2.
opt["bm25f_k1"] = 1
# Normalise inputs?
opt["do_normalise"] = True
# Number of matches to fetch
opt["nummatches"] = 5000000
# Number of terms to calculate
opt["numterms"] = 5
# Boost to score to give to words in the title
opt["title_boost"] = 1.0
# Minimum score to accept as a match
opt["bm25f_score_cutoff"] = 1.25

## Tweakable parameters - Cosine Similarity classifier
# Minimum score to accept as a match
opt["cs_score_cutoff"] = 1

## Tweakable parameters - Category classifier
# Min proportion of item matches for +ve classification.
opt["prop_accept"] = 0.15
# Number of keywords to use for category classifier
opt["n_keywords"] = 200
# Min number of keywords to attempt a category classifier match.
opt["min_keywords"] = 10
# Min relative frequency for a word to be a keyword.
opt["rel_freq_cutoff"] = 100
# Minimum ll for a word to be a keyword. Stat. sig.: 3.84 => p<0.05
opt["log_lik_cutoff"] = 0

## Tweakable parameters - NER and NERStanford classifiers
# Min proportion of item matches for +ve classification.
opt["prop_ner"] = 0.20
# Min number of named entities to attempt a category classifier match.
opt["min_ner"] = 5

## Tweakable parameters - NERStanford classifier
opt["classifierdir_ners"] = '/home/tom/Documents/dev/stanford-ner-2016-10-31/classifiers/'
opt["classpath_ners"] = '/home/tom/Documents/dev/stanford-ner-2016-10-31/'

## Configure scope of run
# Use multiprocessing's pool.Map or simple map?
opt["multiprocessing"] = True
# File suffix to prevent clobbering of output files
opt["fsuffix"] = "_final"
# True: Rebuild Whoosh index. False: Use existing index.
opt["rebuild_index"] = False
# True: Rebuild FDs for all texts. False: Unpickle FDs.
opt["rebuild_FreqDists"] = False
try:
    f = open('Cache/FreqDists.pickle', 'rb')
    f.close()
except FileNotFoundError:
    opt["rebuild_FreqDists"] = True
if opt["rebuild_index"]:
    opt["rebuild_FreqDists"] = True
# True: Rebuild keyword lists. False: Unpickle keywords.
opt["rebuild_keywords"] = False
try:
    f = open('Cache/Studykeywords.pickle', 'rb')
    f.close()
except FileNotFoundError:
    opt["rebuild_keywords"] = True
if opt["rebuild_FreqDists"]:
    opt["rebuild_keywords"] = True
# True: Dev/validation test. False: Main run of stories.
opt["do_test"] = False
# True: Development test. False: Validation test.
opt["dev_test"] = False
if not opt["do_test"]:
    opt["dev_test"] = False


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

class PairClassifier(object):
    """Template class for classifying pairs of URLs. The various
       sub-classifiers use caching, so changing the underlying data structures
       after establishing the classifier will return inconsistent results."""
    def __init__(self, cachematches=True):
        try:
            # This should be set by the subclass, but is a necessary interface.
            self.name
        except NameError:
            self.name = "PairClassifier"
        self._cachematches = cachematches
        self._classified_pairs = {}
    def classify(self, url1, url2):
        """Should return "True" if url1 and url2 are classified as related,
           "False" otherwise"""
        return self.score(url1, url2) >= 1
    def score(self, url1, url2):
        """Should return a normalised float indicating the strength of match
           detected; <1 indicates a "no match" judgement, >=1 indicates
           "match". """
        try:
            return self._getcachedscore(url1, url2)
        except KeyError:
            return self._rescore(url1, url2)
    def _rescore(self, url1, url2):
        """Recalculate a score for the given URLs, returning it and also
           storing it as self._classified_pairs[(url1, url2)]"""
        raise NotImplementedError
    def _getcachedscore(self, url1, url2):
        return self._classified_pairs[tuple(sorted([url1, url2]))]
    def _setcachedscore(self, url1, url2, score):
        if self._cachematches:
            self._classified_pairs[tuple(sorted([url1, url2]))] = score
#            pprint.pprint(list(self._classified_pairs.values()))

class CosineSimilarityClassifier(PairClassifier):
    """Scores documents based on cosine similarity.
       At present, this does not tf-idf weight the articles."""
    def __init__(self, ix, analyser, whoosh_content="text",
                 tf_idf=False, vectors={}, score_cutoff=0.9,
                 name="CosineSimilarityClassifier", *args, **kw):
        self.name = name
        self._whoosh_content = whoosh_content
        # This shouldn't be necessary, but I can't work out how to extract
        # the appropriate analyser object from the field in the index
#        self._analyser = analyser
#        self._reader = ix.reader()
        self._searcher = ix.searcher()
#        self._schema = ix.schema
        self._tf_idf = tf_idf
#        from sklearn.feature_extraction.text import DictVectorizer
#        self.vectorizer = DictVectorizer()
#        self.vectorizer = self.vectorizer.fit( reference_FreqDists.iteritems() )
        if tf_idf:
            raise NotImplementedError
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer()
            print("Fitting TfidfVectorizer for "+self.name+"...")
            print("...done.")
#        self._idfs = {}
#        self._vectors = vectors
#        self.whoosh_content = "text"
        self.score_cutoff = score_cutoff
        super(CosineSimilarityClassifier, self).__init__(*args, **kw)

    def _url_to_sparse(self, url1, url2):
        v = CountVectorizer()
        return v.fit_transform([self._get_text(url1), self._get_text(url2)])

    def _get_text(self, url):     
        docnum = self._searcher.document_number(url=url)
        return self._searcher.stored_fields(docnum)[self._whoosh_content]
        
    def _rescore(self, url1, url2):
        result = cosine_similarity(self._url_to_sparse(url1, url2))[0][1]
        self._setcachedscore(url1, url2, result)
        return result
        

#    def _rescore_old(self, url1, url2):
#        v1 = self._get_vector(url1)
#        v2 = self._get_vector(url2)
#        result = self._cosine_similarity(v1, v2) * (1.0/self.score_cutoff)
#        self._setcachedscore(url1, url2, result)
#        return result
#
#    def _get_idf(self, term):
#        try:
#            return self._idfs[term]
#        except KeyError:
#            if not self._tf_idf:
#                return 1.0
#            df = self._reader.doc_frequency(self.whoosh_content, term)
#            self._idfs[term] = 1.0 / df
#            return self._idfs[term]

#    def _get_vector(self, url):
#        try:
#            return self._vectors[url]
#        except KeyError:
#            docnum = self._searcher.document_number(url=url)
#            text = self._searcher.stored_fields(docnum)[self._whoosh_content]
#            # TODO: Should methodise this at some point.
#            tn = nltk.FreqDist(make_tokenize([text], self._analyser))
##            print([x for x in tn.items()])
#            v = {t:(float(n)/self._get_idf(t)) for t, n in tn.items()}
#            # XXX: Whoosh's term vector implementation chokes on some of our
#            # input, so sadly we need to work around it.
##            v = {t:(float(n)/self._get_idf(t)) for t, n in
##                        self._reader.vector_as("frequency",
##                                               docnum,
##                                               self._whoosh_content)}
#            self._vectors[url] = v
#            return v
#
#    def _cosine_similarity(self, v1, v2):
#        """Based on code snippet at stackoverflow.com/questions/15173225/
#        by user vpekar"""
#        intersection = set(v1.keys()) & set(v2.keys())
#        numerator = sum([v1[x] * v2[x] for x in intersection])
#        sum1 = sum([v1[x] ** 2 for x in v1.keys()])
#        sum2 = sum([v2[x] ** 2 for x in v2.keys()])
#        denominator = sqrt(sum1) * sqrt(sum2)
#
#        if not denominator:
#            return 0.0
#        else:
#            return float(numerator) / denominator

class CompositeClassifier(PairClassifier):
    """Composite classifier, classifying pairs of URLs using an arbitrary
       list of PairClassifier-type classifiers and averaging their scores
       together."""
    def __init__(self, classifiers, name="CompositeClassifier", *args, **kw):
        self.name = name
        self.classifiers = classifiers
        self.allclassifiers = classifiers
        super(CompositeClassifier, self).__init__(*args, **kw)
    def _rescore(self, url1, url2):
        results = {c.name: c.score(url1, url2) for c in self.classifiers}
        result = sum(results.values()) / len(results)
        self._setcachedscore(url1, url2, result)
        return result

class CompositeConditionalClassifier(PairClassifier):
    """Composite classifier, classifying pairs of URLs using an arbitrary
       list of PairClassifier-type classifiers and averaging their scores
       together. If this result is promising (score >= threshold) then
       also add in a set of conditional classifiers (which are conceptually
       too computationally intensive to use for an initial filtering."""
    def __init__(self, classifiers, conditionalclassifiers, threshold = 0.05,
            name = "CompositeConditionalClassifier", *args, **kw):
        self.name = name
        self.threshold = threshold
        self.classifiers = classifiers
        self.conditionalclassifiers = conditionalclassifiers
        self.allclassifiers = classifiers + conditionalclassifiers
        super(CompositeConditionalClassifier, self).__init__(*args, **kw)
    def _rescore(self, url1, url2):
        # dict of subclassifiername: score pairs
        results = {c.name: c.score(url1, url2) for c in self.classifiers}
        result = sum(results.values()) / len(results)
        if result >= self.threshold:
            # If result looks promising, try the expensive classifiers too
            results.update({c.name: c.score(url1, url2) for
                                c in self.conditionalclassifiers})
            result = sum(results.values()) / len(results)
#        pprint.pprint((result, results))
        self._setcachedscore(url1, url2, result)
        return result

class WhooshBM25FClassifier(PairClassifier):
    """A classifier for pairs of URLs using a pre-built whoosh index and
       the BM25F information retrieval algorithm to select matches"""
    def __init__(self, ix, score_cutoff = 1, whoosh_content = "text",
                 numterms = 5, nummatches = 500000, days_window = 3,
                 normalise = True, cachemorelike = "conservative",
                 B = 0.75, K1 = 1.2, name="WhooshBM25FClassifier",
                 scoringfn = lambda x: x,*args, **kw):
        self.searcher = ix.searcher(weighting=whoosh.scoring.BM25F(B=B, K1=K1))
        self.days_window = days_window
        self.score_cutoff = score_cutoff
        self.whoosh_content = whoosh_content
        # XXX twiddle this
        self.numterms = numterms
        self.nummatches = nummatches
        self.normalise = normalise
        self.name = name
        self.cachemorelike = cachemorelike
        self.scoringfn = scoringfn
        self._morelikethiscache = {}
        super(WhooshBM25FClassifier, self).__init__(*args, **kw)
    def _rescore(self, url1, url2):
        r = self.get_more_like(url1, url2)
        score1 = self._find_in_results(r, url2)
        r = self.get_more_like(url2, url1)
        score2 = self._find_in_results(r, url1)
#        if score1 > 0.5 and score2 > 0.5:
#            print "Score for", url1, "WRT", url2, ":", score1
#            print "Score for", url2, "WRT", url1, ":", score2
        result = self.scoringfn((score1+score2) * (1.0/self.score_cutoff) / 2)
        self._setcachedscore(url1, url2, result)
        return result
    def _makedatefilter(self, t, field_name = "time"):
        return whoosh.query.DateRange(field_name,
                   t-datetime.timedelta(self.days_window),
                   t+datetime.timedelta(self.days_window))
    def _find_in_results(self, results, url):
        for hit in results:
            if hit["url"] == url:
                return hit.score
        return 0
    def get_more_like(self, url, otherurl):
        """Performs a more_like() operation on the underlying index to find
           URLs similar to the given URL. Caching is attempted to speed up
           searches."""
        try:
            # If memory becomes a problem on a large dataset, this could
            # be adapted into a weakref dictionary (would be slower, as the
            # references would be discarded) or better replaced by a database
            # which keeps as much in memory as possible.
            return self._morelikethiscache[url]
        except KeyError:
            docnum = None
            docnum = self.searcher.document_number(url=url)
            assert docnum is not None, url
            morelike = self.searcher.more_like(
                      docnum,
                      self.whoosh_content,
                      top=self.nummatches,
                      numterms=self.numterms,
                      normalize=self.normalise,
                      filter=self._makedatefilter(self.searcher.stored_fields(
                                                  docnum)["time"])
                      )
            self._add_to_more_like_cache(url, otherurl)
            return morelike
    def _add_to_more_like_cache(self, url, otherurl):
        # XXX this is an ugly hack. Something like the following might
        # be more effective!
        # @functools.lru_cache(maxsize=4)
        # def get_more_like(self, url): ...
        # With the rest of the function refactored to remove its own cache.
        # The resulting cache could be memory profiled to work out how much
        # each entry costs, then an appropriate maxsize set (possibly by
        # parameter).
        if self.cachemorelike != None:
            if self.cachemorelike == "conservative":
                try:
                    self._morelikethiscache[otherurl]
                    # The other URL is in the cache, so we're not going to
                    # add this one or displace the old one.
                    return 
                except KeyError:
                    # We don't have either URL; clearly a new round of
                    # classifications has started. Wipe the cache and cache
                    # the new URL.
                    self._morelikethiscache = {}
        # FIXME: This seems obviously wrong
        self._morelikethiscacheurl1 = None
        return


class KWClassifier(PairClassifier):
    """Classify using keywords"""
    def __init__(self, study_keywords, prop_accept = 0.15, min_keywords = 10,
                 name = "KWClassifier", *args, **kw):
        # TODO: Could calculate and cache these on the fly if not provided.
        # TODO: Could experiment with reader.most_distinctive_terms()
        self.kws = study_keywords
        self.prop_accept = prop_accept
        self.min_keywords = min_keywords
        self.name = name
        super(KWClassifier, self).__init__(*args, **kw)
    def _rescore(self, url1, url2):
        """Get a proportion of keywords which match"""
        prop_match = self.compare_texts(self.kws[url1], self.kws[url2])
        result = prop_match * (1.0 / self.prop_accept)
        self._setcachedscore(url1, url2, result)
        return result
    def compare_texts(self, kw1, kw2):
        """Compare two lists of keywords. Return the proportion which match."""
        n = min(len(kw1), len(kw2))
        if n == 0 or n <= self.min_keywords:
            return float(0)
        m = 0
        for word in kw1:
            for word2 in kw2:
                if word == word2:
                    m += 1
                    break
        return float(m)/float(n)

class NERClassifier(PairClassifier):
    """Classify using Named Entity Recognition"""
    def __init__(self, ix, whoosh_content="text",
                 prop_ner = 0.25, min_ner = 10,
                 name = "NERClassifier", *args, **kw):
        self._whoosh_content = whoosh_content
        # This shouldn't be necessary, but I can't work out how to extract
        # the appropriate analyser object from the field in the index
        self._searcher = ix.searcher()
        # TODO: Should cache the NER results - calculation is expensive.
        self.prop_accept = prop_ner
        self.min_ner = min_ner
        self.name = name
        super(NERClassifier, self).__init__(*args, **kw)
    def _rescore(self, url1, url2):
        """Get a proportion of named entities which match"""
        prop_match = self.compare_texts(self.select_ne(url1),
                                        self.select_ne(url2))
        result = prop_match * (1.0 / self.prop_accept)
        self._setcachedscore(url1, url2, result)
        return result
    def compare_texts(self, nes1, nes2):
        """Compare two lists of named entities.
           Return the proportion which match."""
        return (self.prop_x_in_y(nes1,nes2) + self.prop_x_in_y(nes2,nes1)) / 2
    def _get_text(self, url):     
        docnum = self._searcher.document_number(url=url)
        return self._searcher.stored_fields(docnum)[self._whoosh_content]
    def select_ne(self, url):
        # Get text
        text = self._get_text(url)
        # Chunk 'named entities' (people, places, organisations etc.) as 
        # (possibly hierarchical) nltk.Tree.
        ne = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
        # Extract them. Note that the NE
        # recognition code is far from perfect and will pick up some bogosities
        # here. It's possible that we'd rather use a more robust library
        # (like the Stanford NER java library) but difficult to do in-process
        # on ScrapingHub.
        tup = [tuple(i.flatten()) for i in ne if isinstance(i, nltk.Tree)]
        return Counter((' '.join(list(zip(*x))[0]).lower()) for x in tup)
    def prop_x_in_y(self, x, y):
        ct  = sum(wt for term, wt in x.items() if term in y)
        tot = sum(wt for term, wt in x.items())
        if tot == 0 or tot < self.min_ner:
            return 0
        return float(ct) / float(tot)

class NERStanfordClassifier(NERClassifier):
    """Classify using the Stanford NER library"""
    def __init__(self, modeldir=os.environ.get('STANFORD_MODELS'),
                 classpath=os.environ.get('CLASSPATH'),
                 stanfordmodel='english.all.3class.distsim.crf.ser.gz',
                 name = "NERStanfordClassifier", *args, **kw):
        os.environ['STANFORD_MODELS'] = modeldir
        os.environ['CLASSPATH'] = classpath
        # TODO: Consider running this as a server, to avoid startup time every
        #       tag. Currently very slow.
        self.tagger = nltk.tag.stanford.StanfordNERTagger(stanfordmodel)
        super(NERStanfordClassifier, self).__init__(*args, **kw)
        self.name = name
    def select_ne(self, url):
        # Get text
        text = self._get_text(url)
        ne = self.tagger.tag(nltk.word_tokenize(text))
        # Extract them.
        return Counter(self.group_by_tag(ne))
    @staticmethod
    def group_by_tag(nes):
        for tag, chunk in itertools.groupby(nes, lambda x:x[1]):
            if tag != "O":
                yield " ".join(w for w, t in chunk)

class NERStanfordServerClassifier(NERClassifier):
    """Classify using the Stanford NER library, already running in server mode
       using something like `java -server -mx400m -cp stanford-ner.jar 
       edu.stanford.nlp.ie.NERServer -loadClassifier 
       classifiers/english.all.3class.distsim.crf.ser.gz -port 1234"""
    def __init__(self, hostname = '127.0.0.1', port = '1234',
                 name = "NERStanfordServerClassifier", *args, **kw):
        self.hostname = hostname
        self.port = port
        super(NERStanfordServerClassifier, self).__init__(*args, **kw)
        self.name = name
    def select_ne(self, url):
        # Get text
        text = self._get_text(url)
        sock = socket.create_connection((self.hostname, self.port))
        sock.sendall(bytes(text.replace('\n', ' ')+'\n', 'utf-8'))
        buf = self.recv_basic(sock)
        return Counter(self.group_by_tag(self.buffer_to_tup(buf)))
    @staticmethod
    def buffer_to_tup(buffer):
        s = str(buffer, 'utf-8').replace('\n', '').strip()
        t = [tuple(x.rsplit('/', maxsplit=1)) for x in s.split(' ')]
        return t
    @staticmethod
    def recv_basic(the_socket):
        total_data=[]
        while True:
            data = the_socket.recv(8192)
            if not data:
                break
            total_data.append(data)
        return b''.join(total_data)
    @staticmethod
    def group_by_tag(nes):
        for tag, chunk in itertools.groupby(nes, lambda x:x[-1]):
            if tag != "O":
                yield " ".join(w for w, t in chunk)

class LockedWriter(object):
    """Write to a file, protected with a multiprocessing.Lock()"""
    def __init__(self, fn='file', printchar=False):
        self._lock = multiprocessing.Lock()
        self.printchar = printchar
        self.fg = open(fn, 'w', 1) # Line buffer only
    def write(self, txt):
        with self._lock:
            self.fg.write(txt)
        if self.printchar:
            sys.stdout.write(self.printchar)
            sys.stdout.flush()



#####
#FUNCTIONS AND PROCEDURES:
#####
#Mathematics
#####
def calc_precision(tp, fp, tn, fn):
    """Calculate the precision statistic.
    
    Keyword arguments:
    tp -- True Positives
    fp -- False Positives
    tn -- True Negatives
    fn -- True Negatives
    
    """
    return tp/(tp+fp)

def calc_recall(tp, fp, tn, fn):
    """Calculate the recall statistic.
    
    Keyword arguments:
    tp -- True Positives
    fp -- False Positives
    tn -- True Negatives
    fn -- True Negatives
    
    """
    return tp/(tp+fn)

def calc_F1(tp, fp, tn, fn):
    """Calculate the F1 score.
    
    Keyword arguments:
    tp -- True Positives
    fp -- False Positives
    tn -- True Negatives
    fn -- True Negatives
    
    """
    return (2.0*((tp/(tp+fp) * tp/(tp+fn)) / (tp/(tp+fp) + tp/(tp+fn))))

def calc_Matthews_cc(tp, fp, tn, fn):
    """Calculate the Matthews correlation coefficient.
    
    Keyword arguments:
    tp -- True Positives
    fp -- False Positives
    tn -- True Negatives
    fn -- True Negatives
    
    """
    return ((tp*tn) - (fp*fn)) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

def calc_dunning_ll(feat, FD1, FD2):
    """Return Dunning's Log Likelihood (G^2) score for the significance of a
    given feature in two FreqDists.

    Formula given in Rayson, Paul, Damon Berridge, and Brian Francis. 2004.
    "Extending the Cochran Rule for the Comparison of Word Frequencies Between
    Corpora." In 7th International Conference on Statistical Analysis of
    Textual Data (JADT 2004).
    http://eprints.lancs.ac.uk/12424/1/rbf04_jadt.pdf. (p.3)
    """
    a = FD1[feat]
    b = FD2[feat]
    c = FD1.N() - a
    d = FD2.N() - b
    N = a + b + c + d
    try:
        return (2 * (a*log(a) + b*log(b) + c*log(c) + d*log(d) + N*log(N) -
            (a+b)*log(a+b) - (a+c)*log(a+c) - (b+d)*log(b+d) - (c+d)*log(c+d)))
    except ValueError as e:
        print(("Error calculating dunning_ll. Feat==%s; Corpus1[feat]==%i, "
               "Corpus2[feat]==%i, Corpus1[!feat]==%i, Corpus2[!feat]==%i." %
                (feat, a, b, c, d)))
        print(FD1, FD2)
        raise e

#####
#Input processing
#####

def extract_keywords(study_FD, ref_FD, n):
    """Return a list of n keywords, representing the most informative words
    from study_FD, as measured by relative frequency ratio:
    RF = Freq/N
    RFRatio = RF(study corpus) / RF(reference corpus)
    """
    study_RF = []
    for w in study_FD:
        rf = study_FD.freq(w) / ref_FD.freq(w)
        ll = calc_dunning_ll(w, ref_FD, study_FD)
        if rf >= opt["rel_freq_cutoff"] and ll >= opt["log_lik_cutoff"]:
            study_RF.append((w, rf))
    # Sort in place, by relative frequency:
    study_RF.sort(reverse=True, key=lambda tup: tup[1])
    return [x[0] for x in study_RF[:n]]
def produce_item_keywords(item):
    """Take a dict-like object including "text" and "url" keys, such as that
    returned from whoosh.reader.all_stored_fields() for a document, and return
    a (url, [keywords]) tuple.

    This function is aplit out to allow the use of pool.map()"""
    sys.stderr.write('.')
    item_FD = nltk.FreqDist(make_tokenize([item["text"]], analyser))
    return ( item['url'],
             extract_keywords(item_FD,
                              reference_FreqDists[item["paperurl"]],
                              opt["n_keywords"]) )

def split_mappings(d, prop):
    """Partition a dictionary, d, into prop and 1-prop proportions.

    Pops the last (prop*100)% of d to d2, which is returned.

    """
    d2 = {}
    for _ in range(int(len(d)*prop)):
        key, val = d.popitem()#last=True)
        d2[key] = val
    return d2

def get_mappings(fn):
    """Open CSVs with manual mappings data. Returns a dict with the
    first URL as the key, and a dict of {URL2: relvalue} for each
    mapping. relvalue == 1 iff the text is "Related", otherwise 0.

    """
    mappings_dict = {}
    n = 0
    with open(fn, 'r') as f:
        # Skip header line
        next(f)
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            url1 = str(line[0])
            url2 = str(line[1])
            relation = str(line[2])
            if url1 not in mappings_dict:
                mappings_dict[url1] = {}
            relvalue = False
            if relation == "Related":
                relvalue = True
            mappings_dict[url1][url2] = relvalue
            n += 1
    print("Manual mappings:", n, "in", len(mappings_dict), "slices.")
    return mappings_dict

def build_analyser():
    """Build a whoosh analyser, based on the configuration options"""
    analyser = opt["tokeniser"]
    if opt["ignore_shouting"]:
        analyser = analyser | ShoutingFilter()
    if opt["to_lower"]:
        analyser = analyser | whoosh.analysis.LowercaseFilter()
    if opt["reject_stopwords"] or opt["reject_badwords"]:
        analyser = analyser | whoosh.analysis.StopFilter(stoplist=rejectwords)
    return analyser

def process_raw_content(fn, writer):
    """Extract content from a file"""
    print("Processing %s..." % fn)
    # For unescaping HTML entities
    h = html.parser.HTMLParser()
    with open(fn, newline='', encoding=opt['input_encoding']) as f:
        reader = csv.reader(f, delimiter=',')
#        line = unicode(f.read())
        # Skip leading separator
#        chunks = text.lstrip("#\r\n").split("####\r\n")
#        for header,text in zip(chunks[0::2], chunks[1::2]):
        for url, paperurl, title, date, text in reader:
            # Can't just split on ";" as HTML numbered entities in title
            # have ';'s
#            l = header.split(";",2)
            # Remove date from end
#            l[-1] = l[-1][:-42]
            # And re-add it as its own item
#            l.append(datetime.datetime.strptime(
#                        header.rstrip()[-20:],"%Y-%m-%dT%H:%M:%SZ"))


            time = (datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ"))
            # time = (datetime.datetime.strptime(date, "%Y-%m-%d"))
#            paperurl = re.sub(re.compile(r'(http://[^/]+/).*'), '\g<1>', url)

            # And finally the article body (not converted to nltk.Text here
            # as memory usage is horrendous for large datasets).
            # Pre-tokenisation mangling needs to happen here.

            if opt["pre_process_fns"]:
                for f in pre_process_title_fns:
                    title = f(title)
                for f in pre_process_text_fns:
                    text = f(text)

            # Unescape HTML entities. From Python 3.4 onwards, this is properly
            # text = html.unescape(text)
            text = h.unescape(text)
            if opt["pad_fullstops"]:
                text = text.replace('.', '. ')
#            l.append(text.rstrip())
            if opt["rebuild_index"]:
                # Check if the article is sufficiently long, using the standard
                # tokeniser
                if (len([x for x in opt["tokeniser"](text)]) >=
                        opt["min_length"]):
                    # Then tokenise, filter and add to the index

                    try:
                        writer.add_document(url=url,
                                            paperurl=paperurl,
                                            title=title,
                                            time=time,
                                            text=text)
                    except Exception as e:
                        print("Failed to index: "+url+", "+paperurl+", "+title+", "+text)
                        raise e
#                else:
#                    print("Too short:", url, text)
    return

def make_tokenize(ls, analyser = build_analyser()):
    """Produce a tokenized list of words from a list of strings"""
    # This now uses a whoosh analyser to ensure that it tokenises consistently
    # between techniques.
    return [w.text for w in analyser(' '.join(ls))]

def index_articles(fl, analyser = build_analyser()):
    """Open output article files, parse and establish whoosh index."""
    if not opt["rebuild_index"]:
        print("Importing index...")
        try:
            ix = whoosh.index.open_dir(index_dir)
            print("Done. %i articles retrieved." % ix.doc_count())
        except Exception as e:
            print("Failed:", str(e))
            opt["rebuild_index"] = True
            opt["rebuild_FreqDists"] = True
            opt["rebuild_keywords"] = True
    if opt["rebuild_index"]:
        print("Rebuilding index.")
        schema = whoosh.fields.Schema(
            url=whoosh.fields.ID(stored=True, unique=True),
            paperurl=whoosh.fields.ID(stored=True),
            title=whoosh.fields.TEXT(stored=True,
                                     phrase=False,
                                     analyzer=analyser,
                                     field_boost=opt["title_boost"]),
            text=whoosh.fields.TEXT(stored=True,
                                    phrase=False,
                                    analyzer=analyser),
            time=whoosh.fields.DATETIME(stored=True))
        ix = whoosh.index.create_in(index_dir, schema)
        writer = ix.writer()
        for fn in fl:
            process_raw_content(fn, writer)
        print("Committing index...")
        writer.commit()
        print("Done.")
    return ix


#####
#Output processing
#####

def nice_dict_format(d):
    """Pretty format dictionaries into a multi-line string"""
    return ''.join([key+": "+str(d[key])+"\n" for key in list(d.keys())])

#####
#DATA SETUP
#####
if not os.path.exists(index_dir):
    os.mkdir(index_dir)

if opt["reject_stopwords"]:
    rejectwords.update(set(nltk.corpus.stopwords.words('english')))
if opt["reject_badwords"]:
    rejectwords.update(badwords)

analyser = build_analyser()
# ix: A whoosh index of the contents of each document in the corpus
ix = index_articles(file_list, analyser)
reader = ix.reader()

# Two dict of nltk.FreqDists for each reference/study corpus.
# key = main URL/URL
# d["front_page_url"] = [text for text in articles from that source]
if opt["rebuild_FreqDists"]:
    print("Rebuilding FreqDists.")
    reference_FreqDists = {}
    for item in reader.all_stored_fields():
        try:
            _ = reference_FreqDists[item["paperurl"]]
        except KeyError:
            print("Setting up reference_FreqDists['"+item['paperurl']+"']")
            reference_FreqDists[item["paperurl"]] = nltk.FreqDist()
        reference_FreqDists[item["paperurl"]].update(
                make_tokenize([item["text"]], analyser))

    pickle.dump( reference_FreqDists, open('Cache/FreqDists.pickle', 'wb'))
    print("Reference corpora FreqDists generated and pickled.")
else:
    reference_FreqDists = pickle.load( open('Cache/FreqDists.pickle', 'rb'))
    print("Corpora FreqDists unpickled and loaded.")

if opt["rebuild_keywords"]:
    print("Rebuilding keywords list (very time-consuming).")
#    study_keywords = {}

    if opt["multiprocessing"]:
        pool = multiprocessing.Pool(processes=None, maxtasksperchild=5)   # Use cpu# of subprocesses
        mapper = functools.partial(pool.map, chunksize=100)
    else:
        mapper = map

    study_keywords = dict(mapper(produce_item_keywords,
                                 reader.all_stored_fields(), 
                                 chunksize=100))

#    for item in reader.all_stored_fields():
#        item_FD = nltk.FreqDist(make_tokenize([item["text"]], analyser))
#        study_keywords[item["url"]] = extract_keywords(item_FD,
#                                      reference_FreqDists[item["paperurl"]],
#                                      opt["n_keywords"])
    pickle.dump(study_keywords, open('Cache/Studykeywords.pickle', 'wb'))
    print("Keywords generated and pickled.")
else:
    study_keywords = pickle.load( open('Cache/Studykeywords.pickle', 'rb'))
    print("Keywords unpickled and loaded.")

if opt["do_test"]:
    # manual_mappings: A dict of hand-codings{URL1 : {URL2: relation, ...} }.
    # Partitioned into main and validation sets.
    manual_mappings = get_mappings(mappings_file)

    if opt["dev_test"]:
        # Discard validation sample
        _ = split_mappings(manual_mappings, opt["prop_validate"])
    else:
        # Keep only validation sample
        manual_mappings = split_mappings(manual_mappings, opt["prop_validate"])

    # A set of all URLs which feature at least once on either side of the
    # manual mappings data
    url_times = {}
    to_del = []
    for item in reader.all_stored_fields():
        url_times[item["url"]] = item["time"]

    # This is a catch against the case when the manual mappings data
    # contains pairings which are outside the sliding time frame.
    # If we process them as a special case, we break the honest of the
    # test. If we don't, we get false classification errors as there is
    # always no result against them. Should not be needed.

    for i in manual_mappings:
        for j in manual_mappings[i]:
            try:
                if (abs(url_times[i] - url_times[j]) >
                        datetime.timedelta(opt["days_window"])):
                    print("Deleting mapping: time too great", i, j)
                    to_del.append((i, j))
            except KeyError:
                # One of the mappings not present in the index
#                print("Deleting mapping: at least one article not present", i, j)
                to_del.append((i, j))
            # A simple len(manual_mappings) doesn't give the total number of
            # items to classify as it's a nested dict.
    for tup in to_del:
        del manual_mappings[tup[0]][tup[1]]


# Set up classifiers
kw_classifier = KWClassifier(study_keywords,
                             prop_accept = opt["prop_accept"],
                             min_keywords = opt["min_keywords"],
                             cachematches = False)

#ne_classifier = NERClassifier(ix,
#                              whoosh_content = "text",
#                              prop_ner = opt["prop_ner"],
#                              min_ner = opt["min_ner"],
#                              cachematches = False)
#
#
#nes_classifier = NERStanfordServerClassifier(ix = ix,
#                                       whoosh_content = "text",
#                                       prop_ner = opt["prop_ner"],
#                                       min_ner = opt["min_ner"],
#                                       cachematches = False,
#                                       modeldir = opt["classifierdir_ners"],
#                                       classpath = opt["classpath_ners"],
#                                      )

#cs_classifier = CosineSimilarityClassifier(ix, analyser,
#                                           whoosh_content = "text",
#                                           score_cutoff = opt["cs_score_cutoff"])

bm25f_classifier = WhooshBM25FClassifier(ix,
                                         score_cutoff = opt["bm25f_score_cutoff"],
                                         whoosh_content = "text",
                                         nummatches = opt["nummatches"],
                                         numterms = opt["numterms"],
                                         normalise = opt["do_normalise"],
                                         B = opt["bm25f_b"],
                                         K1 = opt["bm25f_k1"],
                                         days_window = opt["days_window"],
                                         cachematches = False)

#bm25fsquared_classifier = WhooshBM25FClassifier(ix,
#                                         score_cutoff = opt["bm25f_score_cutoff"],
#                                         whoosh_content = "text",
#                                         nummatches = opt["nummatches"],
#                                         numterms = opt["numterms"],
#                                         normalise = opt["do_normalise"],
#                                         B = opt["bm25f_b"],
#                                         K1 = opt["bm25f_k1"],
#                                         days_window = opt["days_window"],
#                                         cachematches = False,
#                                         scoringfn = lambda x: x*x)

final_classifier = CompositeConditionalClassifier(
                        [kw_classifier],
                        [bm25f_classifier],
                        threshold = opt["cc_threshold"],
                        cachematches = True)

## Array of classifiers to test
# For validation - can get results from individual classifiers plus ensemble
# Note that this sends *everything* through the slow classifiers
#classifiers = final_classifier.allclassifiers + [final_classifier]

# For main run - just the final classifier
classifiers = [final_classifier]

#####
#TEST: PRODUCE CONFUSION MATRIX AND PRINT TEST STATISTICS
#####
if opt["do_test"]:
    # FreqDists to produce confusion matrices for scoring
    FD_results = {k.name: nltk.FreqDist() for k in classifiers}
    scores = {k.name: [] for k in classifiers}
    # Count number of items to classify
    num_mappings = sum(len(d) for d in manual_mappings.values())
    # Print setup info and options dictionary    
    if opt["dev_test"]:
        print("***** Development test results at", time.ctime())
    else:
        print("***** VALIDATION TEST RESULTS at", time.ctime())
    print("\n", nice_dict_format(opt))
#    pprint.pprint(manual_mappings)
    for classifier in classifiers:
        print("\nStarting classifier", classifier.name)
#        n = 1

        def _inner(t):
#            pprint.pprint(t)
            n, t2 = t
            url1, d = t2
            scores = []
            r = {"TruePositive": 0,
                 "FalsePositive": 0,
                 "FalseNegative": 0,
                 "TrueNegative": 0}
            for url2, manual in d.items():
#                print("\rClassifying", n, "of", num_mappings, end=' ')
#                sys.stdout.flush()
                # Do the actual classification
                judgement = classifier.classify(url1, url2)
                scores.append( (classifier.score(url1, url2), manual) )
                # File the results in the appropriate confusion matrix cell
                if judgement and manual:
                    r["TruePositive"] += 1
#                    FD_results[classifier.name]["TruePositive"] += 1
#                    print("TruePositive:", url1, url2, scores[-1])
                elif judgement and not manual:
                    r["FalsePositive"] += 1
#                    FD_results[classifier.name]["FalsePositive"] += 1
                    print("FalsePositive:", url1, url2, scores[-1])
                elif manual:
                    r["FalseNegative"] += 1
#                    FD_results[classifier.name]["FalseNegative"] += 1
                    print("FalseNegative:", url1, url2, scores[-1])
                else:
                    r["TrueNegative"] += 1
#                    FD_results[classifier.name]["TrueNegative"] += 1
#                n += 1
            return (scores, r)

        if opt["multiprocessing"]:
            pool = multiprocessing.Pool(processes=None, maxtasksperchild=5)   # Use cpu# of subprocesses
            mapper = functools.partial(pool.map, chunksize=100)
        else:
            mapper = map
        
        # Classify all articles through the pool.
        for (s, r) in mapper(_inner, enumerate(manual_mappings.items())):
            scores[classifier.name].extend(s)
            for label in r:
                FD_results[classifier.name][label] += r[label]

        print("\nClassification method:", classifier.name)

        # Plot histograms
        match_values = [x[0] for x in scores[classifier.name] if x[1]]
        unmatch_values = [x[0] for x in scores[classifier.name] if not x[1]]
        fig = plt.figure()
        fig.subplots_adjust(left=0.2, wspace=0.6)
        ax1 = fig.add_subplot(221)
        ax1.hist(match_values, 50, log=True)
#        ax1.yscale('log', nonposy='clip')
        ax1.set_title("Matches: "+classifier.name)
        ax2 = fig.add_subplot(222)
        ax2.hist(unmatch_values, 50, log=True)
#        ax2.yscale('log', nonposy='clip')
        ax2.set_title("Unmatches: "+classifier.name)
        plt.show()

        judgement_l = [x[0] >= opt["final_match_threshold"] for x in scores[classifier.name]]
        hand_judgement_l = [x[1] for x in scores[classifier.name]]
        alpha_array = np.array((judgement_l, hand_judgement_l))

        # Print confusion table:
        print(FD_results[classifier.name])
        FD_results[classifier.name].tabulate()

        fp = float(FD_results[classifier.name]["FalsePositive"])
        fn = float(FD_results[classifier.name]["FalseNegative"])
        tp = float(FD_results[classifier.name]["TruePositive"])
        tn = float(FD_results[classifier.name]["TrueNegative"])

        try:
            print("Accuracy: %f" % ( (tp + tn) / (tp + tn + fp + fn) ))
            print("Precision: %f " % calc_precision(tp, fp, tn, fn))
            print("Recall: %f" % calc_recall(tp, fp, tn, fn))
            print("F1: %f" % calc_F1(tp, fp, tn, fn))
            print("Matthews correlation coefficient: %f" % (
                    calc_Matthews_cc(tp, fp, tn, fn) ))
            print("Krippendorff's alpha: %f" % alpha(alpha_array,
                                                     level_of_measurement='nominal'))
            print("")
        except (ZeroDivisionError, FloatingPointError) as e:
            print(e)
    for k, v in scores.items():
        print(k, sum([t[0] for t in v])/len(v))
        if k == "CosineSimilarityClassifier":
            print(sorted([(t[0], t[1]) for t in v if t[0] >= 0.8])[::-1])
    exit("Test complete")

#####
#MAIN CLASSIFICATION
#####

# Set up a TSV to get partial output as we go
csv_writer = LockedWriter(fn='Output/production_matches'+opt['fsuffix']+'.tsv', printchar='.')
csv_writer.write("From\tTo\tStrength\n")
# Set up a TSV to record node info
node_writer = LockedWriter(fn='Output/production_nodes'+opt['fsuffix']+'.tsv', printchar='#')
node_writer.write("URL\tWords\n")
# XXX
#skip_writer = LockedWriter('Output/production_nodes_skipped'+opt['fsuffix']+.tsv, printchar='!')


# Classify all articles through the pool executor.
with concurrent.futures.ProcessPoolExecutor() as ex:

    # Separated out for Executor.map
    def _classify_article(article):
        wc = len((article["text"]+" "+article["title"]).split(" "))
        # Skip excessively short articles - REMOVED: handled on indexing (with
        # a better (but incompatible) word count algorithm
#        if wc < opt["min_length"]:
#            print('Too short, skipping:', article['url'], '('+str(wc)+' words)')
#            return (None, None)
        # Save URL and word count for writing to nodes file
        node = article["url"]+"\t"+str(wc)+"\n"
        # List of matches (formatted as tsv strings)
        csv = []
    #    node_writer.write(article["url"]+"\t"+str(wc)+"\n")
        # Start the keyword classification process for this URL, all-against-all
        for cmp_article in reader.all_stored_fields():
            # Don't compare against ourselves
            if article["url"] == cmp_article["url"]:
                continue
            # Don't compare against articles more than 3 days in the future, or
            # in the past at all (ensures only one matching for each article pair)
            elif ((article["time"] - cmp_article["time"]) >
                    datetime.timedelta(opt["days_window"])):
                continue
            elif article["time"] < cmp_article["time"]:
                continue

            # The final classifier will automatically call the other classifiers
            # for scores. The others will calculate if necessary.
            score_final = final_classifier.score(article["url"],
                                                cmp_article["url"])
    #        judgement_final = finalclassifier.classify(article["url"],
    #                                                   cmp_article["url"])

            if score_final >= opt["final_match_threshold"]: # Matched!
                # Write the TSV as we go
                csv.append("%s\t%s\t%s\n" % (article["url"],
                                             cmp_article["url"],
                                             score_final))
    #            csv_writer.write("%s\t%s\t%s\n" % (article["url"],
    #                                               cmp_article["url"],
    #                                               score_final))
        return (node, csv)

    if opt["multiprocessing"]:
        mapper = ex.map
    else:
        mapper = map


    for node, csv in mapper(_classify_article, reader.all_stored_fields()):
        if node is not None:
            node_writer.write(node)
            for line in csv:
                csv_writer.write(line)

#num_mappings = reader.doc_count()
#i = 0
#groups = []
#searcher = whoosh.searching.Searcher(reader, closereader = False)
#for article in reader.all_stored_fields():
#    i += 1
#    if i<30000:
#        continuue
##    if (i % 1000 == 0):

#    print "\rClassifying", i, "of", num_mappings,
##    if article["url"] == article["paperurl"]:
##        print "Warning, trying to handle front page article."
##        continue
            


"""Cat classifier improvements:
Add inferred category as feature?
Double-weight titles?
Discard all non-Initial capital input?
Bigrams?
Stemming?
POS-tag and accept only proper nouns / placenames / ...
Split intra-capitalisation?
Issues with excessive story chaining for sports: Arsenal 1:0 Chelsea -> Chelsea 2:5 Wigan -> Wigan 0:1 Man Utd... (This can be justified as the "premier league" story, but maybe should be separated).
Limit to n links per article?




Done:
Handle HTML entities
Don't link (and, indeed, exclude?) if words <= x: photos, chatterbox pages, cartoons etc. -> Disproportionately buggering the story detection.
Re-run analysis generating probabilities for the links? Would let us dynamically adjust the linking probability, at the expense of buggering the reported accuracy until re-run.
Multi-thread the classification
Manual discard of troublesome words: Trivial effect, but obviously correct
Minimum amount of keywords before we just guess "No match": ++++++ Precision
Discard all fully-uppercase input: ++ Precision
Establish a cutoff for the relative frequency ratio (10): +++
Lowercase all input: (mildy worse)
Discard stopwords. (Zero effect)
Discard manual badwords list separately from stopwords.
Discard words with full stops in the middle (or better, split them). Replace "." with ". " (Tiny positive effect?)
Allow LL statistical significance cutoff. (Zero effect at RF>=16, p<0.05 or p<0.01)
Added score histogram plotter to base class 
"""
