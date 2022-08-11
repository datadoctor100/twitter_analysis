#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 09:10:23 2022

@author: zan
"""

# Import libraries
import os
import ast
import networkx as nx
from collections import Counter
import pandas as pd
from datashader.layout import random_layout, circular_layout, forceatlas2_layout
from datashader.bundling import connect_edges, hammer_bundle
from datashader.bundling import connect_edges, hammer_bundle
from itertools import chain
from holoviews.element.graphs import layout_nodes
import json
import holoviews as hv
cvsopts = dict(plot_height = 750, plot_width = 1500)
from holoviews import opts
from holoviews.element.graphs import layout_nodes
import colorcet as cc
from holoviews.operation.datashader import (datashade, dynspread, directly_connect_edges, bundle_graph, stack)
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from colorcet import fire, glasbey, bmw
import time
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

rgb_opts = opts.RGB(width = 1500, height = 1000, xaxis = None, yaxis = None)
hv.output(png = 'svg')

# Get data
#os.chdir('/home/zan/data/')
#os.system("snscrape --jsonl --max-results 10000 --since 2020-01-01 twitter-search 'data science'> tweets.json")

# Read data
tweets = pd.read_json('tweets.json', lines = True)
replies = tweets.dropna(subset = ['inReplyToUser'])
mentions = tweets.dropna(subset = ['mentionedUsers'])

# Inspect
print(f"Min Date- {min(tweets['date'])}")
print(f"Max Date- {max(tweets['date'])}")

# Separate users
users = pd.DataFrame([x for x in tweets['user']]).drop_duplicates(subset = ['username'])
additional_users = pd.DataFrame([x for x in replies['inReplyToUser']]).drop_duplicates(subset = ['username'])

# Iterate to extract mentions
mentioned_users = []

for x in mentions['mentionedUsers']:
    
    for i in x:
        
        mentioned_users.append(i)
        
mentioned_users = pd.DataFrame(mentioned_users)

'''
    Clustering
'''

# Combine
all_users = pd.concat([users, additional_users, mentioned_users]).drop_duplicates(subset = ['username'])
all_users.drop(['_type', 'id', 'rawDescription', 'descriptionUrls', 'linkUrl', 'linkTcourl', 'profileImageUrl', 'profileBannerUrl', 'url', 'location', 'protected', 'label', 'displayname', 'description'], axis = 1, inplace = True)
all_users = all_users.set_index('username', drop = True).dropna(subset = ['created'])
all_users['created'] = all_users['created'].apply(lambda x: time.mktime(pd.to_datetime(x).timetuple()))
all_users['verified'] = all_users['verified'].astype(int)

real_users = all_users[all_users['verified'] == True]

# Optimize clustering 
ssd = []
clusters = range(1, 20)

# Iterate
for k in clusters:
    
    # Fit
    km = KMeans(n_clusters = k).fit(all_users)
    km.fit(all_users)
    ssd.append(km.inertia_)
    
# Plot the elbow
plt.plot(clusters, ssd, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Fir optimal value
km = KMeans(n_clusters = 5, random_state = 100).fit(all_users)
km.fit(all_users)

all_users['cluster'] = km.labels_
    
'''
    Networking
'''

# Separate components
nodes = [i for i in real_users.index]
edges = []

# Iterate the data to add initial relationships
for idx, row in replies.iterrows():
    
    edges.append((row['user']['username'], row['inReplyToUser']['username']))
    
for idx, row in mentions.iterrows():
    
    for mention in row['mentionedUsers']:
        
        edges.append((row['user']['username'], mention['username']))
        
# Calculate edge weights
weights = Counter(edges)
weights = pd.DataFrame.from_dict(weights, orient = 'index').reset_index()    
weights = pd.DataFrame({'source': [x[0] for x in weights['index']],
                        'target': [x[1] for x in weights['index']],
                        'weight': [i for i in weights[0]]})

#weights.to_csv('twitter_networks.csv', index = False)
