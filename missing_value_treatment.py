# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:40:55 2018

@author: admin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import sys
import types

df=pd.read_csv("German_credit.csv")
print (df.head())

# Missing value treatment

def Missing_value(df):
    for x in list(df.columns):
        if df[x].dtype=='int64':
            df=df.fillna(df.mean()).dropna(axis=1,how='all')
#        elif df[x].dtype=='float64':
#            df=df.fillna(df.median()).dropna(axis=1,how='all')
#        elif df[x].dtype=='object':
#            df.replace(' ',np.nan,inplace=True)
        low=0.05
        high=.95
        quant_val=df.quantile([low,high])
        print(quant_val)
           
        for x in list(df.columns):
            if df[x].dtypes=="int64" or df[x].dtypes=="float64":
               df = df[(df[x] >= quant_val.loc[low, x]) & (df[x] <= quant_val.loc[high, x])]
               df=df.reset_index(drop=True)
            if df[x].dtypes=='object':
                df[x].fillna(df[x].mode()[0])
                print('dadta frame is',df)
    df.to_csv('filename_1.csv')
    return (df)  
                    
Missing_value(df)




#outlier detection

#def Outlier_detection(df):
#    low=0.05
#    high=.95
#    quant_val=df.quantile([low,high])
#    print(quant_val)
#       
#    for x in list(df.columns):
#        if df[x].dtypes=="int64" or df[x].dtypes=="float64":
#           df = df[(df[x] >= quant_val.loc[low, x]) & (df[x] <= quant_val.loc[high, x])]
#           df=df.reset_index(drop=True)
#        if df[x].dtypes=='object':
#            df[x].fillna(df[x].mode()[0])
#            print('dadta frame is',df)
#    df.to_csv("filename_1.csv")               
#    
#    return df

#Outlier_detection(df)








#Feature Engg

#def feature_engg(df):
#    for x in list(df.columns):
#        nunique = input_1.apply(pd.Series.nunique)
#        cols_to_drop = nunique[nunique == 1].index
#        input_1.drop(cols_to_drop, axis=1)

    
#feature_engg(df)
        