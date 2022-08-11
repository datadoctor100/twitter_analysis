#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 09:10:23 2022

@author: zan
"""

# Import libraries
import os
import pandas as pd
import re
import string
import nltk
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from bertopic import BERTopic

# Read data
#os.chdir('/home/zan/data/')
#os.system("snscrape --jsonl --max-results 10000 --since 2020-01-01 twitter-search 'data science'> tweets.json")

# Init language
words = set(nltk.corpus.words.words())
stop_words = stopwords.words('english')

# Function to get hashtags
def extract_hashtags(x):
    
    hashtag_list = []
      
    # splitting the text into words
    for word in x.split():
          
        # checking the first charcter of every word
        if word[0] == '#':
              
            # adding the word to the hashtag_list
            hashtag_list.append(word[1:])
      
    return hashtag_list

# Function to process text
def clean_tweet(tweet):

    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet.lower()).split())

# Function to further process text
pattern = re.compile(r'\s+')

def more_cleaning(tweet):
    
    tweet = re.sub("@[A-Za-z0-9_]+","", tweet)
    tweet = re.sub("#[A-Za-z0-9_]+","", tweet)
    tweet = ''.join([i for i in tweet if not i.isdigit()])
    tweet = " ".join(re.split("\s+", tweet, flags = re.UNICODE))
    
    return tweet

# Function to filter pos
def filter_pos(x):

    x = nltk.pos_tag(x)
    x = [i[0] for i in x if i[1].startswith(('N', 'A', 'J'))]

    return x

# Reade data
tweets = pd.read_json('tweets.json', lines = True)

# Apply functions
tweets['content'].iloc[1]
clean_tweet(tweets['content'].iloc[1])
more_cleaning(clean_tweet(tweets['content'].iloc[1]))

tweets['tweet'] = tweets['content'].apply(lambda x: more_cleaning(clean_tweet(x)).translate(str.maketrans('', '', string.punctuation)).replace("`", '').strip().split())
tweets['pos'] = tweets['tweet'].apply(filter_pos)
tweets['hashtags'] = tweets['content'].apply(extract_hashtags)

# # Topic modeling
# words = corpora.Dictionary(tweets['pos'])
# corpus = [words.doc2bow(text) for text in tweets['pos']]

# lda = LdaMulticore(corpus, id2word = words, num_topics = 10)
# lda.print_topics()

# cm = CoherenceModel(model = lda, texts = corpus, dictionary = words, coherence = 'c_v')
# cm.get_coherence()

# BERT
bert = BERTopic()
topics, probs = bert.fit_transform(tweets['pos'].apply(lambda x: ' '.join(x)))
bert.get_topic_info()







