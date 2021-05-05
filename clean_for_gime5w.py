#%%
import csv
import json
import os
import pandas as pd
import numpy as np

#%%
os.chdir("/Users/paulinemckim/Desktop/capstone/Data/Guardian")

# set directory 
print("Current Working Directory " , os.getcwd())

df = pd.read_csv('./ready_for_bert.csv', header=0)
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
df = df.copy(deep=True)
df['cluster_counts'] = df.groupby('v_hier_infomap_comm')['v_hier_infomap_comm'].transform('count')
df = df[df['cluster_counts'] <5]
df = df[df['cluster_counts'] >= 2]

# %%
print(len(df))

# %%
#df.to_csv("litte_set_5_to_9.csv")

# %%
