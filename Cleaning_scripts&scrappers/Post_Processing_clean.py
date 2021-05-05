#%%
import datetime as dt
import glob
import matplotlib as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import spacy
from collections import Counter
import os
import spacy
#%%
os.chdir("/Users/paulinemckim/Desktop/capstone/Data/Guardian")

# set directory 
print("Current Working Directory " , os.getcwd())   




#%%
#df=pd.read_csv("Guar_medium_api.CSV")
df=pd.read_csv("Guar_Small.CSV") # originally went into 
df = df.loc[:, ~df.columns.str.startswith('Unnam')]
df1=pd.read_csv("gephi_news_capstone_mine_.csv") # gephi 

print(df.columns)
print(df1.columns)
print(df.dtypes)
print(df1.dtypes)

print(len(df))
print(len(df1))

# %%
#df["text"]= df[["text"]].astype(str)
df = df[~df['text'].str.contains("please sign up")] 
df = df[~df['text'].str.contains("live blog")] 
df = df[~df['text'].str.contains("That’s all from us")] 
df = df[~df["text"].str.contains("reading our blog")]
df = df[~df["text"].str.contains("Top story")]
print(df)
print(len(df))

#%%
#df1=df1.drop(["timeset","Id", "v_name", "v_url"], axis=1)
#df=df.drop(["t"], axis=1)
# %%
print(df)

# %%
print(df1.head())
#%%
print(df.head())
# %%
#df1[["cleaned_id", "cleaned"]]= df1["v_id"].str.split("-", n = 1, expand = True)




#df1=df1.rename(columns={"cleaned": "headline"})
'''
df1['headline'] = df1['headline'].astype(str)
df['headline'] = df['headline'].astype(str)
df1['headline'] = df1['headline'].str.strip()
df['headline'] = df['headline'].str.strip()
df["text"]=df["text"].astype(str)
'''
df["t"]=df['t'].astype(str)
df["v_id"]=df["t"] + " - " + df["headline"].replace('\n', ' ') ## fixes newlines into spaces 
# %%

#df1=df1.add_suffix('_df1')
#df=df.add_suffix('_df')


# %%
df = pd.merge(df, 
                      df1, 
                      on ='v_id', 
                      how ='right')

# %%
print(len(df))


# %%
import spacy
nlp = spacy.load("en_core_web_sm")

nlp.max_length = 3865772
# NER
tokens = []
lemma = []
pos = []
ents=[]
ents_label=[]

for doc in nlp.pipe(df['headline'].astype('unicode').values, batch_size=50):
    if doc.is_parsed:
        tokens.append([n.text for n in doc])
        lemma.append([n.lemma_ for n in doc])
        pos.append([n.pos_ for n in doc])
        ents.append([n.ents for n in doc.ents])
        ents_label.append([n.label_ for n in doc.ents])
        ### want to update this for person and orgs only
    else:
        # We want to make sure that the lists of parsed results have the
        # same number of entries of the original Dataframe, so add some blanks in case the parse fails
        tokens.append(None)
        lemma.append(None)
        pos.append(None)

df['headline_tokens'] = tokens
df['headline_lemma'] = lemma
df['headline_pos'] = pos
df["headline_ents"]= ents
df["head_ents_label"]=ents_label


tokens = []
lemma = []
pos = []
ents=[]
ents_label=[]

for doc in nlp.pipe(df['text'].astype('unicode').values, batch_size=50):
    if doc.is_parsed:
        tokens.append([n.text for n in doc])
        lemma.append([n.lemma_ for n in doc])
        pos.append([n.pos_ for n in doc])
        ents.append([n.ents for n in doc.ents])
        ents_label.append([n.label_ for n in doc.ents])
        ### want to update this for person and orgs only
    else:
        # We want to make sure that the lists of parsed results have the
        # same number of entries of the original Dataframe, so add some blanks in case the parse fails
        tokens.append(None)
        lemma.append(None)
        pos.append(None)

df['text_tokens'] = tokens
df['text_lemma'] = lemma
df['text_pos'] = pos
df["text_ents"]= ents
df["text_ents_label"]=ents_label

# %%
print(df)

# %%
df.to_csv("ready_for_bert.csv")

# %%
print(df)

# %%
