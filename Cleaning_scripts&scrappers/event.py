#%%
import numpy as np
np.random.seed(42)

import json 

import pandas as pd
import datetime as dt
#import gensim
#from gensim.summarization import summarize
import warnings
warnings.filterwarnings('ignore')

os.chdir("/Users/paulinemckim/Desktop/")


df = pd.read_csv('ready_for_bert.csv', header=0)
df=df.drop(["Unnamed: 0"], axis=1)


#%%
print(df.info())
#%%

df.dropna()
                    
print(len(df))
df=df.sort_values(by='pub_date_dto')

df["headline"] = df["headline"].astype(str)
#print(len(df))
#print(df.head())
#print(df.sectionname.value_counts())
# sort dates from first to last 
df=df.sort_values(by='pub_date_dto')
df["text"]=df['text'].astype(str)
df = df[~df["text"].str.contains("Top story")]
df = df[~df["headline"].str.contains("a visual guide")]
df = df[~df["headline"].str.contains("as it happened")]
print(len(df))
df["text"] = df["text"].astype(str)
df["headline"] = df["headline"].astype(str)

#%%
df1=df[df.v_hier_infomap_comm == "22"] 
df1=df1.sort_values(by='pub_date_dto')
#df1 = df1[df1['text'].notna()]
#df1 = df1[df1['pub_date_dto'].notna()]
docs=df1.text.to_list()
docs = list(map(str, docs))
timestamps = df1["pub_date_dto"].to_list()
text=docs[0]
date_publish=timestamps[0]


#%%

from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor

extractor = MasterExtractor()
doc = Document.from_text(text, date_publish)
# or: doc = Document(title, lead, text, date_publish) 
doc = extractor.parse(doc)

top_who_answer = doc.get_top_answer('who').get_parts_as_text()
print(top_who_answer)

# %%


# %%
