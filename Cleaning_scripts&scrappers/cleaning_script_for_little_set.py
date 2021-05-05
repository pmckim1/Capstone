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
import en_core_web_sm

import os
import datetime

os.chdir("/Users/paulinemckim/Desktop/capstone/Data/Guardian/Big_results_run")


# %%
data=glob.glob("res*.csv") 
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

df=df.drop_duplicates()
#%%#
# drop columns that dont have info 
df = df.loc[:, ~df.columns.str.startswith('Unnam')]
print(len(df))

# %%
print(len(df))


df=df.rename(columns={"sectionId": "sectionid", "sectionName": "sectionname", "webTitle": "headline"})
# check columns 
print(df.columns)


section = ["politics", "global-development", "us-news", "uk-news", "world"]
df= df[df.sectionid.isin(section)]
print(len(df))

# %%
df["text"]= df[["text"]].astype(str)
df = df[~df['text'].str.contains("please sign up")] 
df = df[~df['text'].str.contains("live blog")] 
df = df[~df['text'].str.contains("That’s all from us")] 
df = df[~df["text"].str.contains("reading our blog")]
print(df)
print(len(df))


#%%
briefing=["Monday briefing", "Tuesday briefing", "Wednesday briefing", "Thursday briefing", "Friday briefing", "Saturday briefing", "Sunday briefing"]

df = df[~df['text'].isin(briefing)]


#%%
print(df.info())


#%%

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
print(df)
print(len(df))

print(df.pub_date_dto.min())
print(df.pub_date_dto.max())

#%%
df=df[(df['pub_date_dto'] > '2015-12-31')] 
print(df)

#%%
print(len(df))

print(df.pub_date_dto.min())
print(df.pub_date_dto.max())

#%%



                  
                    

print(len(df))



#%%
print(df.sectionname.value_counts())
#%%
#df.to_csv("Guar_medium_api.CSV")

# %%
