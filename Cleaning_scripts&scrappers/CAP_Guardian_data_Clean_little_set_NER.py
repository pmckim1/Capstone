
# %%
import datetime as dt
import glob
import matplotlib as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import spacy
import json
from collections import Counter
import en_core_web_sm

import os
import datetime
# Standard quick checks
def dfChkBasics(dframe, valCnt = False): 
  cnt = 1
  print('\ndataframe Basic Check function -')
  
  try:
    print(f'\n{cnt}: info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  print(f'\n{cnt}: describe(): ')
  cnt+=1
  print(dframe.describe())

  print(f'\n{cnt}: dtypes: ')
  cnt+=1
  print(dframe.dtypes)

  try:
    print(f'\n{cnt}: columns: ')
    cnt+=1
    print(dframe.columns)
  except: pass

  print(f'\n{cnt}: head() -- ')
  cnt+=1
  print(dframe.head())

  print(f'\n{cnt}: shape: ')
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print('\nValue Counts for each feature -')
    for colname in dframe.columns :
      print(f'\n{cnt}: {colname} value_counts(): ')
      print(dframe[colname].value_counts())
      cnt +=1

# examples:
#dfChkBasics(df)

#%%
os.chdir("/Users/paulinemckim/Desktop/capstone/Data/Guardian/little_results_run")

# set directory 
print("Current Working Directory " , os.getcwd())


# %%
'''
k=pd.read_csv('results_guar1K.csv')
j=pd.read_csv('results_guar1J.csv')
i=pd.read_csv('results_guar1I.csv')
h=pd.read_csv('results_guar1H.csv')
l=pd.read_csv('results_guar_1L.csv')
b=pd.read_csv('results_guar_1B.csv')
c=pd.read_csv('results_guar_1C.csv')
a=pd.read_csv("results_guar_1A.csv")
e=pd.read_csv("results_guar1E.csv")
d=pd.read_csv('results_guar1D.csv')
g=pd.read_csv('results_guar1G.csv')
f=pd.read_csv('results_guar1F.csv')
d=pd.read_csv('results_guar1D.csv')
g=pd.read_csv('results_guar1G.csv')
f=pd.read_csv('results_guar1F.csv')
p=pd.read_csv('results_guar1P.csv')
m=pd.read_csv('results_guarM.csv')
o=pd.read_csv('results_guarO.csv')
n=pd.read_csv('results_guarN.csv')

print("a")
print(len(a))
print("b") 
print(len(b))
print("c") 
print(len(c))
print("d") 
print(len(d))
print("e")
print(len(e))
print("f")
print(len(f))
print("g") 
print(len(g))
print("h") 
print(len(h))
print("i") 
print(len(i))
print("j")
print(len(j))
print("k")
print(len(k))
print("l")
print(len(l))
print("m") 
print(len(m))
print("n") 
print(len(n))
print("o") 
print(len(o))
print("p") 
print(len(p))

'''

#%%



data=glob.glob("lit*.csv") 
print(data)
#%%
df = []

for file in data:
    try:
        print(file)
        df.append(pd.read_csv(file))
    except Exception as e:
        print("Error on line loading file, continuing.")
  
df = pd.concat(df)

print("complete")


#%%
print(df.columns)

#%%#
# drop columns that dont have info 
df = df.loc[:, ~df.columns.str.startswith('Unnam')]
print(len(df))

# %%
print(len(df))

# %%
section = ["politics", "us-news", "uk-news", "global-development", "world"]
df= df[df.sectionId.isin(section)]
print(len(df))

#%%
df=df.drop_duplicates()
print(len(df))

print(df.sectionName.value_counts())

df.drop_duplicates(inplace=True)
#%%
# checking for nan values & drop nan values 
print(df[df.columns[df.isnull().any()]].isnull().sum())
#%%
df.dropna(inplace=True)
print(len(df))

#%%
# rename columns 

df=df.rename(columns={"sectionId": "sectionid", "sectionName": "sectionname", "webTitle": "headline"})
# check columns 
print(df.columns)


#%%
#read to excel for easier checking 
#df.to_excel("full_excel.xlsx")


#%%
# check text column for blog, streaming subscribe, briefing type stories 

df = df[~df['text'].str.contains("please sign up")] 
df = df[~df['text'].str.contains("live blog")] 
df = df[~df['text'].str.contains("That’s all from us")] 
df = df[~df["text"].str.contains("reading our blog")]
print(df)
print(len(df))

briefing=("Monday briefing", "Tuesday briefing", "Wednesday briefing", "Thursday briefing", "Friday briefing", "Saturday briefing", "Sunday briefing")

df = df[~df['text'].isin(briefing)]



#%%
#clean up date column 

def create_dto(row, colname):
# for index, row in dc_data.iterrows():
    if type(row[colname]) is not str:
        return "Unknown"
    else:
        # Try the various known time formats.
        dtFormat = [
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%y %H:%M',
            '%m/%d/%y',
            "%Y-%m-%dT%H:%M:%SZ",
        ]
        # save cell data to local variable
        cell_contents = row[colname]
        # Drop decimal timestamp precision, if it exists.
        cell_contents = cell_contents.split('.')[0]
        for i in dtFormat:
            try:
                dto = dt.datetime.strptime(cell_contents,i)
                return (
               
                    dto.strftime("%Y-%m-%d")

                 )
            except ValueError:
                pass
        else:
            print("Failed to parse: {:s}".format(cell_contents))
            
#%%            
df['pub_date_dto'] = df.apply(lambda row: create_dto(row,"pub_date"), axis=1)
#%%
df.pub_date.value_counts()



#%%

df['pub_date_dto'] = pd.to_datetime(df['pub_date'])
print(df)
print(df.dtypes)

#%%
print(df.pub_date_dto.value_counts())
#%%
df["pub_date_dto"]=df.pub_date_dto.map(lambda x: x.replace(second=0))
#%%
print(df.head())
#%%
df.pub_date.isnull().sum(axis = 0)
#%%
df.dtypes
#%%
print(len(df))





# %%
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
df["text_ents"] = json.dumps(ents)
df["text_ents_label"] = json.dumps(ents_label)


#%%

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
df["headline_ents"] = json.dumps(ents)
df["head_ents_label"] = json.dumps(ents_label)


# %%
print(df.head())

# %%
df.to_csv("guardian_test.csv")




# %%
