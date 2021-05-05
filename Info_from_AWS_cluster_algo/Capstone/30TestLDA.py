#!/usr/bin/env python3
"""Make keywords for each cluster, for the news project.
by Tom Nicholls, 2013-2015.

This work is in the public domain, under the terms of the CC0 dedication:
http://creativecommons.org/publicdomain/zero/1.0/
"""
#####
#IMPORTS
#####

import nltk
import whoosh.analysis
import whoosh.index
import whoosh.fields
import whoosh.qparser
import whoosh.classify
import sys
import pprint
import string
import gensim
from tqdm import tqdm
from pprint import pprint

#####
#PARAMETERS
#####

index_dir = "WhooshIndex"
k = 5000
output_model_fname = "Cache/GensimLDA.model"


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




stop = set(nltk.corpus.stopwords.words('english'))
exclude = set(string.punctuation)
lemma = nltk.stem.wordnet.WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized



with ix.reader() as r:
    print("There are", r.doc_count(), "documents in the index.")
    print("Cleaning text...")

    art_clean = [clean(doc['text']).split()
                    for doc in tqdm(r.all_stored_fields())]

print("...done. Making term dictionary...")
dictionary = gensim.corpora.Dictionary(art_clean)
# Remove hapaxes - no value
dictionary.filter_extremes(no_below=2, no_above=1.0, keep_n=None)

print("...done. Making document term matrix...")
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
dtm = [dictionary.doc2bow(art) for art in art_clean]

print("...done. Training LDA model...")
lmod = gensim.models.ldamodel.LdaModel(dtm,
                                       num_topics=k,
                                       id2word = dictionary,
                                       passes=1)
lmod.save(output_model_fname)

