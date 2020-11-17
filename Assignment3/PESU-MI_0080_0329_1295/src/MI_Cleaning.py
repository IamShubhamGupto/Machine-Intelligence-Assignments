#!/usr/bin/env python
# coding: utf-8
#THIS FILE WAS DEVELOPED IN JUPYTER NOTEBOOK
#IMPORTED TO .PY, IGNORE THE FILE STRUCTURE
import os.path
import pandas as pd
import numpy as np

def normalize(df, columns):
    result = df.copy()
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

cwd = os.getcwd()
ccwd = cwd
ccwd = ccwd.replace('PESU-MI_0080_0329_1295/src','')

#PATH = r'F:/Engi_Books/Sem5/MI/UE18CS303_Assignment/Assignment3/LBW_Dataset.csv'
PATH = r'LBW_Dataset.csv'
SAVE = False

try:
    df=pd.read_csv(ccwd+PATH)
except:
    print("File does not exist")
    exit(0)
df

df.describe()

# drop any row with more than 7 nans
df=df.dropna(thresh=7)
df.describe()

# replace age nan based on community age mean
df.loc[:,'Age'] = df.groupby("Community").transform(lambda x: x.fillna(round(x.mean())))
df.describe()
print("NaNs in Age",df['Age'].isnull().values.any())


# replace weight nan based on weight mean
df.loc[:,'Weight']=df['Weight'].fillna(df['Weight'].mean())
df.describe()
print("NaNs in Weight",df['Weight'].isnull().values.any())


# replace HB nan based on HB mean
df.loc[:,'HB']=df['HB'].fillna(df['HB'].mean())
#df.describe()
print("NaNs in HB",df['HB'].isnull().values.any())


# replace BP nan based on BP mean
df.loc[:,'BP'] = df['BP'].fillna(df['BP'].mean())
print("NaNs in BP",df['BP'].isnull().values.any())

df['Education'].fillna(5.0, inplace=True)
df['Residence']=df['Residence'].fillna(method='ffill')
df['Delivery phase']=df['Delivery phase'].fillna(method='ffill')
df.describe()

# fill remaining NaNs with forward fill
df=df.fillna(method="ffill")
df.describe()

normalized_df = normalize(df, ['Age','Weight','HB','BP',])
normalized_df.describe()

cwd = cwd.replace('src','')
if SAVE:
    normalized_df.to_csv(cwd + '/data/LBW_clean.csv',index=False)





