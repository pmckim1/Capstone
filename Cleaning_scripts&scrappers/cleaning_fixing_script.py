
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
# Get the absolute path of the current folder
abspath_curr = '/content/drive/My Drive/capstone/capstone/Data/Guardian/'

df = pd.read_csv(abspath_curr +'stream-topped-1.csv', header=0)

df.columns