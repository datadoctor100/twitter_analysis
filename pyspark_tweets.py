#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 08:57:36 2022

@author: zan
"""

# Import libraries
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, ArrayType, FloatType
from pyspark.sql import SparkSession
import re
import string
from pyspark.sql.functions import udf, col, size
import nltk
from nltk.corpus import stopwords
import pyspark.sql.functions as F
from pyspark.sql.functions import lit
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import CountVectorizer
from nltk.stem import WordNetLemmatizer
from pyspark.ml.clustering import LDA
from datetime import datetime
  
# Init
wordnet = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
words = set(nltk.corpus.words.words())

# Function to check for english
def get_words(words):
    
    return [word for word in wordnet.lemmatize(words) if word in nltk.corpus.words.words()]

# Function to process text
def prepare_txt(x):
    
    data = [i for i in x if i not in stop_words]
    tags = nltk.pos_tag(data)
    x1 = [x[0] for x in tags if x[0] in words and len(x[0]) > 3 and x[1].startswith(('N', 'J', 'V'))]

    if len(x1) > 0:
        
        return x1
    
    else:
        
        return None

# Function to get terms from cv
def obtain_terms(token_list):
    
    return [vocab[token_id] for token_id in token_list]
    
# Initialize spark
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
spark = SparkSession.builder.getOrCreate()

# Drop unwated fields
drop = ('_type', 'cashtags', 'conversationId', 'id', 'inReplyToTweetId', 'inReplyToUser', 'lang', 'media', 'outlinks', 'renderedContent', 'sourceUrl', 'tcooutlinks', 'url')
df = spark.read.json('tweets.json').drop(*drop)

# Inspect
df.columns
df.count()
df.agg(F.min("date"), F.max("date")).show()

'''
    Text processing
'''

# Remove punctuation and digits
puncs = re.compile('[%s]' % re.escape(string.punctuation))
nums = re.compile('(\\d+)')

puncsudf = udf(lambda x: re.sub(puncs,' ', x))
numsudf = udf(lambda x: re.sub(nums,' ', x).lower().split())

df1 = df.withColumn('tweet', puncsudf('content'))
df2 = df1.withColumn('txt', numsudf('tweet'))

text_prepper = udf(lambda x: prepare_txt(x), ArrayType(StringType()))

# Apply function
df3 = df2.withColumn('pos', text_prepper('txt'))
df3.select('pos').show(5, truncate = False)

# Filter empty tweets
df4 = df3[~df3.pos.isNull()] 

'''
    Topic Modeling
'''

# Count vector
cv = CountVectorizer(inputCol = 'pos', outputCol = 'features', vocabSize = 100)
cv_model = cv.fit(df4)
df5 = cv_model.transform(df4)

idf = IDF(inputCol = "features", outputCol = "fts")
idf_model = idf.fit(df5)
df6 = idf_model.transform(df5) 

# Topic modeling
lda = LDA(k = 5, maxIter = 10, featuresCol = 'fts')

start = datetime.now()
lda_model = lda.fit(df6)
print(f'Execution Time: {datetime.now() - start}')

vocab = cv_model.vocabulary

words_udf = udf(obtain_terms, ArrayType(StringType()))

topics = lda_model.describeTopics(10).withColumn('words', words_udf(col('termIndices')))
topics = topics.toPandas()
  
  