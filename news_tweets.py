#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 13:18:58 2022

@author: dataguy
"""

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
drop = ('_type', 'cashtags', 'conversationId', 'id', 'inReplyToTweetId', 'inReplyToUser', 'lang', 'media', 'outlinks', 'renderedContent', 'sourceUrl', 'tcooutlinks', 'url', 'source', 'mentionedUsers')
df = spark.read.json('news.json').drop(*drop)

# Get users
df1 = df.withColumn('username', col('user.username')).withColumn('country', col('place.country')).withColumn('country_cd', col('place.countryCode')).drop('user', 'coordinates', 'place')

string_maker = udf(lambda x: str(x), StringType())

df2 = df1.withColumn('hashtags', string_maker(col('hashtags')))
df3 = df2.withColumn('quoted_content', col('quotedTweet.content')).drop('quotedTweet')

#df3.write.format('csv').save('news_tweets.csv')
#df3.toPandas().to_csv('news_tweets.csv', index = False)