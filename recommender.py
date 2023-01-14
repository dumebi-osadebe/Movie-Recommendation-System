#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:11:14 2023

@author: dumebi
"""

import pandas as pd
from tensorflow.keras import models, layers, utils, backend
import matplotlib.pyplot as plt
import numpy as np

# import the preprocessing and text libraries from scikitlearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

# string manipulation libraries
import re
import string

df_movies = pd.read_csv("./ml-latest-small/movies.csv")
df_users = pd.read_csv("./ml-latest-small/ratings.csv")
movie_tags = pd.read_csv("./ml-latest-small/tags.csv")

#for the content based system we will be utilizing the movie genres features

#preprocessing the sci-fi genre for each movie

def processing_text(text: str) -> str:
    """Args:
        text (str): the input text you want to clean
    Returns:
        str: the cleaned text
    """
    
    text = re.sub("-", "", text)
    
    # return text in lower case and stripped of whitespaces
    return text.lower().strip()

df_movies["genres"] = df_movies["genres"].apply(lambda x: processing_text(x))
    
# users
df_users.drop(columns = ["timestamp"], inplace= True)

# match movies only with existing user ratings
df_users = df_users.merge(df_movies, how = "left")

#make columns of the genres using set() to identify the unique genres
genres = [i.split("|") for i in df_users["genres"].unique()]
columns = list(set([i for lst in genres for i in lst]))
columns.remove("(no genres listed)")

#create Product Feature matrix
for col in columns:
    df_movies[col] = df_movies["genres"].apply(lambda x: 1 if col in x else 0)
    

df_movies.drop(columns = ["genres"], inplace = True)

# create User Product matrix
tmp = df_users.copy()
df_users = tmp.pivot_table(index = "userId", columns = "movieId", values = "rating")

missing_cols = list(set(df_users.columns) - set(df_movies.index))

for col in missing_cols:
    df_users[col] = np.nan
    

    











































