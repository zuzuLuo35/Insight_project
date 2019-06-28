# feature extraction libraries
import re
import time
import pypatent as pp
# data manipulation libraries
import os
import glob
import pickle
import pandas as pd
import numpy as np
import math
from scipy import spatial
# machine learning packages
from gensim.models import FastText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
from sklearn.feature_extraction import stop_words
stopwords = stop_words.ENGLISH_STOP_WORDS
import nltk, string
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
lemmer = nltk.stem.WordNetLemmatizer()
# visualization libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
# set number of keywords for feature vector construction
num_key = 1000
search_limit = 1000
