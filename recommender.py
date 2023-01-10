#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:11:14 2023

@author: dumebi
"""

import pandas as pd
from tensorflow.keras import models, layers, utils, backend
import matplotlib.pyplot as plt

# import the preprocessing and text libraries from scikitlearn
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

# string manipulation libraries
import re
import string
import nltk
from nltk.corpus import stopwords

movies = pd.read_csv("./ml-latest-small/movies.csv")
ratings = pd.read_csv("./ml-latest-small/ratings.csv")
movie_tags = pd.read_csv("./ml-latest-small/tags.csv")

#for the content based system we will be utilizing the movie genres features

#preprocessing the genres for each movie
def processing_text(text: str) -> str:
    """Args:
        text (str): the input text you want to clean
    Returns:
        str: the cleaned text
    """
    
    text = re.sub("[|]", "", text)
    text = re.sub("-", "", text)
    
    # return text in lower case and stripped of whitespaces
    return text.lower().strip()

movies["genres"] = movies["genres"].apply(lambda x: processing_text(x))

print(movies["genres"])
    
