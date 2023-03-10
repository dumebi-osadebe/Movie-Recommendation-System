#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:11:14 2023

@author: dumebi
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, utils, backend
import matplotlib.pyplot as plt
import numpy as np

# import the pr eprocessing and text libraries from scikitlearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error

from math import sqrt
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
df_users["userId"] = df_users["userId"].apply(lambda x: x-1)
df_users.drop(columns = ["timestamp"], inplace= True)

# match movies only with existing user ratings
df_users = df_users.merge(df_movies[["movieId", "title"]], how = "left")

# set the index of the movie dataframe to be the movieIds to account for nonconsistent integer usage for movieIds
df_movies = df_movies.set_index("movieId")

#make columns of the genres using set() to identify the unique genres
genres = [i.split("|") for i in df_movies["genres"].unique()]
columns = list(set([i for lst in genres for i in lst]))
columns.remove("(no genres listed)")

#create Product Feature matrix
for col in columns:
    df_movies[col] = df_movies["genres"].apply(lambda x: 1 if col in x else 0)
    

df_movies.drop(columns = ["genres"], inplace = True)

# create User Product matrix 
tmp = df_users.copy()

df_users = tmp.pivot_table(index = "userId", columns = "movieId", values = "rating")

missing_cols = list(set(df_movies.index) - set(df_users.columns))


for col in missing_cols:
    df_users[col] = np.nan

df_users = df_users[sorted(df_users.columns)]
    
#scaling the data to be in a sizeable range for modelling
scaler = MinMaxScaler()
df_users = pd.DataFrame(scaler.fit_transform(df_users), columns=df_users.columns, index = df_users.index)


#split data for training and testing vertically assigning 70% of the movie data for training
split = int(0.7*df_users.shape[1])
df_train = df_users.loc[:, :split-1]
df_test = df_users.loc[:, split:]

#select a user to try out our model on and name a dataframe specifically for that user
i =  1
train = df_train.iloc[i].to_frame(name = "y")
test = df_test.iloc[i].to_frame(name = "y")

tmp = test.copy()
train = train.append(tmp)

# shapes
# fill all the nan values with 0 then transpose to have a user as the column and the movies as rows
usr = train[["y"]].fillna(0).values.T  
# drop these columns and turn dataframe into return them as an ndarray      
prd = df_movies.drop(columns = ["title"]).values  

#create user feature matrix as user_ft(user, feature) = usr(users, products) x prd(products, features)
usr_ft = tf.matmul(usr, prd)

# normalize
weights = usr_ft / tf.reduce_sum(usr_ft, axis=1, keepdims=True)

# create user products matrix as rating(users,products) = weights(users,fatures) x prd.T(features,products)
pred = tf.matmul(weights, prd.T)

test = test.merge(pd.DataFrame(pred[0], columns=["yhat"]), how="left", left_index=True, right_index=True).reset_index()
test = test[~test["y"].isna()]


def rmse(y_test, pred):
    score = []
    for movie in y_test:
        rms = sqrt(mean_squared_error(y_test, pred))
        score.append(rms)
    return np.means(score)

print("--- user", i, "---")
top = 5
y_test = test.sort_values("y", ascending=False)["movieId"].values[:top]
print("y_test:", y_test)

predicted = test.sort_values("yhat", ascending=False)["movieId"].values[:top]
print("predicted:", predicted)

true_positive = len(list(set(y_test) & set(predicted)))
print("true positive:", true_positive, "("+str(round(true_positive/top*100,1))+"%)")
print("accuracy:", str(round(metrics.accuracy_score(y_test,predicted)*100,1))+"%")
#print("mrr:", mean_reciprocal_rank(y_test, pred))
print("rmse:", rmse(y_test, pred))
































