# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import libraries
import pandas as pd
import os
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, ArrayType, FloatType
import re
import string
from pyspark.sql.functions import udf, col, size, lit, explode
import pyspark.sql.functions as F
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, IndexToString, StringIndexer, VectorIndexer, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, DecisionTreeClassifier

# Initialize spark
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.getOrCreate()

# Drop unwated fields
drop = ('_type', 'cashtags', 'conversationId', 'id', 'inReplyToTweetId', 'inReplyToUser', 'lang', 'media', 'outlinks', 'renderedContent', 'sourceUrl', 'tcooutlinks', 'url')
df = spark.read.json('news.json').drop(*drop)

# Separate user information
users = df.select('user.created',
                  'user.description',
                  'user.favouritesCount',
                  'user.followersCount',
                  'user.friendsCount',
                  'user.location',
                  'user.mediaCount',
                  'user.username',
                  'user.verified').distinct()

u1 = users.toPandas()
u1.to_csv('news_tweeters.csv', index = False)