# created by Kamila Kolpashnikova 2023
# replication code for "Fear of falling: Scoping review and topic analysis using natural language processing" published in PLOS ONE 2023

# REQUIREMENTS:
# Python    3.9.7
# pandas    1.4.2
# pybliometrics 3.3.0
# numpy     1.22.3
# matplotlib    3.5.1
# scikit-learn  1.2.2
# seaborn   0.11.2
# networkx  2.6.3
# spacy     3.4.4
# gensim    4.1.2
# joblib    1.2.0
# scipy     1.7.1


### Load packages

import numpy as np
np.random.seed(0)
from pprint import pprint
from time import time
import os,sys
import math
import csv

# add path to custom packages
src_dir = os.path.join('helpers')
sys.path.append(src_dir)

from filter_words import run_stopword_statistics, make_stopwords_filter, remove_stopwords_from_list_texts

import spacy
nlp = spacy.load('en_core_web_md')

from helper_functions import clean_stopwords, coherence_per_topic, find_best_n_topics, get_clean_output, get_list, get_top_n_words, get_topics_from_model, has_numbers, my_lemmatizer, plot_top_words, plot_top_words_colors, process_words

import pandas as pd
import gensim
import re
import matplotlib.pyplot as plt

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection  import GridSearchCV
from gensim.models.coherencemodel import CoherenceModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from joblib import dump, load
import seaborn as sns
from itertools import chain
from collections import defaultdict

### Load the Clean Dataset

df_d = pd.read_csv('data/searches/df_d.csv')

# fill nas as strings
df_d = df_d.fillna('')

# remove empty titles and abstracts
df_d = df_d[(df_d['title']!='') & (df_d['Abstract']!='')]
df_d.reset_index(inplace=True, drop=True)


### Additional Cleaning

# remove parts in abstracts that contain additional Copyright infromation
df_d['abstract'] = [abs.split('Copyright')[0] for i in df_d['Abstract']]

# for titles and abstracts
texts = (df_d['title'] +[' ' for i in range(106)] + df_d['abstract'])
#texts_orig = (df_d['title'] +[' ' for i in range(106)] + df_d['abstract'])


### Choosing Stopword Paradigm

# creating set of stopwords using Sarica and Luo 2021 paper
path_stopword_list =  os.path.join('data','sarica_and_luo_2021.txt')
if path_stopword_list != None:
    with open(path_stopword_list,'r', encoding='utf-8') as f:
        x = f.readlines()
    stopwords = set([word.lower() for h in x for word in h.strip().split(' ')])
    
# remove all keywords and subject headings used in search, including abbreviations and derivative words
stopwords = stopwords.union({'fear', 'falling', 'fall', 'falls', 'fof', 'faller', 'nonfaller', 'nursinghome',
                            'aged', 'and over', 'centenarian', 'frail', 'elderly',
                            'nonagenarians', 'octogenerian', 'long-term', 'care', 'assisted',
                            'living', 'facilities', 'homes', 'nursing', 'skilled', 'elder',
                            'senior', 'pensioner', 'old', 'adult', 'people', 'person'})

# get cleaned list of abstracts with removed stopwords 
# this also applies the Gerlach, Shi, and Nunes Amaral 2019
lst = get_list(texts = texts, stop_words = stopwords, processing_choice='nouns', N_s=100, cutoff_val=0.5)
output = get_clean_output(lst)

#lst_add = get_list(texts = texts, stop_words = stopwords, processing_choice='nouns', N_s=100, cutoff_val=0.5, path_to_file = 'data/searches/filtered_list_')
#output_add = get_clean_output(lst_add)


### Additional functions

# function to calculate coherence score
def scorer_sklearn(estimator, X,y=None):
    
    topics = get_topics_from_model(
        estimator,
        vectorizer,
        n_top_words
    )
    cm = CoherenceModel(
        topics=topics,
        texts = output['list_texts_filter'],
        corpus=output['corpus_filter'], 
        dictionary=output['dictionary_filter'],  
        coherence='c_v', 
        topn=n_top_words,
        processes=1
    )

    return cm.get_coherence()


# updated find_best_n_topics function from helpers
def find_best_n_topics(model_name, texts, dictionary, corpus, n_features, n_top_words, coherence, rs = 0, alp = 0.1, l1 = 0, start = 50, end = 437, step = 50):
    # Create a list of the topic numbers we want to try
    topic_nums = list(np.arange(start, end, step))
    
    texts_nouns = [" ".join(i) for i in texts]

    # Run the nmf model and calculate the coherence score
    # for each number of topics
    coherence_scores = []

    for num in topic_nums:
        
        if model_name == 'nmf':
            # For NMF
            vectorizer = TfidfVectorizer(
                #max_features=n_features, stop_words=list(stopwords)
            )

            tfidf = vectorizer.fit_transform(texts_nouns)

            model = NMF(n_components=num,
                        max_iter=200, 
                        init="nndsvda", 
                        random_state=rs, 
                        alpha=alp, 
                        l1_ratio=l1).fit(tfidf)
            
        elif model_name == 'kullback-leibler':
            # For NMF
            vectorizer = TfidfVectorizer(
                #max_features=n_features, stop_words=list(stopwords)
            )

            tfidf = vectorizer.fit_transform(texts_nouns)

            model = NMF(
                n_components=num,
                beta_loss="kullback-leibler",
                init="nndsvda",
                solver="mu",
                random_state=rs, 
                alpha=alp, 
                l1_ratio=l1).fit(tfidf)
        
        elif model_name == 'lda':
            ## for LDA

            vectorizer = CountVectorizer(
                max_features=n_features, stop_words=list(stopwords),
            )

            tf = vectorizer.fit_transform(texts_nouns)

            model = LatentDirichletAllocation(
                n_components=num,
                max_iter=10,
                learning_method="online",
                learning_offset=150,
                random_state=0,
            ).fit(tf)
        
        elif model_name =='gensim_lda':
            model = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                id2word=dictionary,
                num_topics=num, 
                random_state=117, 
                #update_every=1,
                #chunksize=1500, 
                #passes=5, iterations=100,
                #alpha='asymmetric', eta=1/100,
                #alpha='auto', eta=1/100,
                #per_word_topics=True
               )
            
        elif model_name =='gensim_nmf':
            model = gensim.models.Nmf(corpus=corpus, 
                num_topics=num,
                random_state =1
               )


        # Run the coherence model to get the score
        if model_name =='gensim_lda' or model_name =='gensim_nmf':            
            
            cm = CoherenceModel(
                model = model,
                texts = texts,
                corpus=corpus, 
                dictionary=dictionary, 
                coherence=coherence,
                topn = n_top_words
            )
            
        else:        
        
            topics = get_topics_from_model(
                model,
                vectorizer,
                n_top_words
            )
            cm = CoherenceModel(
                topics=topics,
                texts = texts,
                corpus=corpus, 
                dictionary=dictionary, 
                coherence=coherence,
                topn=n_top_words
            )
        
        
        coherence_scores.append(round(cm.get_coherence(), 5))

    # Get the number of topics with the highest coherence score
    return list(zip(topic_nums, coherence_scores))

def find_elbow_point(x, y):
    # Calculate the first derivative
    dydx = np.gradient(y, x)

    # Calculate the second derivative
    d2ydx2 = np.gradient(dydx, x)

    # Find the index of the maximum curvature
    elbow_idx = np.argmax(np.abs(d2ydx2))

    return x[elbow_idx], y[elbow_idx]

def find_elbow_point_max(x, y):
    # Find the index of the maximum curvature
    elbow_idx = np.argmax(np.abs(y))

    return x[elbow_idx], y[elbow_idx]

# updated topic plotting function from helpers
def plot_top_words_colors(model, feature_names, n_top_words, title):
    num = len(model.components_)
    h = math.ceil(num/5)
    f = 8*h
    fig, axes = plt.subplots(h, 4, figsize=(30, f), sharex=True)
    axes = axes.flatten()
    
    palette = sns.color_palette("Spectral", 10).as_hex()
    
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7, color = palette[topic_idx])
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        #fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


### NMF Hyperparameter Tuning - Do not run often

for i in ['nouns', 'with adj', 'noun chunks', 'all']:
    
    print(i)
    print()
    
    lst = get_list(texts, stop_words = stopwords, processing_choice=i, N_s=100, cutoff_val=0.5)
    output = get_clean_output(lst)
    
    texts_nouns = output['texts_filter']

    n_samples = len(texts_nouns)
    n_features = output['n_features'] # from above at corpus
    n_components = 25
    n_top_words = 10


    vectorizer = TfidfVectorizer(
        #max_features=n_features, stop_words=list(stopwords)
    )

    tfidf = vectorizer.fit_transform(texts_nouns)

    parameters = {
        'alpha': (0.01, 0.1, 0.5, 0.8),
        'n_components':(2, 5, 10, 15),
        'l1_ratio': (0, 0.2, 0.5, 0.8, 1),
    }

    grid_search = GridSearchCV(NMF(
                                n_components=n_components,
                                random_state=25,
                                beta_loss="kullback-leibler",
                                init="nndsvda",
                                solver="mu",
                                max_iter=200,
                                alpha=0.1,
                                l1_ratio=0.5,
                            ),
                            parameters,
                            scoring=scorer_sklearn,
                            cv=5)

    print("Performing grid search...")
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(tfidf)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))







### Graphing

## the following reproduces Fig 3
topic_nums, coherence_scores = zip(*find_best_n_topics('kullback-leibler', 
                            texts = output['list_texts_filter'], 
                            corpus=output['corpus_filter'], 
                            dictionary=output['dictionary_filter'], 
                            n_features=n_features, 
                            n_top_words=n_top_words, coherence='c_v',
                           rs = 0,
                           alp = 0.8,
                           l1 = 1,
                            start = 2,
                            end = 50, 
                            step = 1))

plt.plot(topic_nums, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence score')

# Get the elbow point
elbow_point = find_elbow_point_max(topic_nums, coherence_scores)

# Plot the elbow point
plt.scatter(elbow_point[0], elbow_point[1], marker='o', color='red')

plt.show()

## the following reproduces Fig 2
topic_nums, coherence_scores = zip(*find_best_n_topics('kullback-leibler', 
                            texts = output['list_texts_filter'], 
                            corpus=output['corpus_filter'], 
                            dictionary=output['dictionary_filter'], 
                            n_features=n_features, 
                            n_top_words=n_top_words, coherence='c_v',
                           rs = 0,
                           alp = 0.8,
                           l1 = 1,
                            start = 2,
                            end = 437, 
                            step = 1))

plt.plot(topic_nums, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence score')

# Get the elbow point
elbow_point = find_elbow_point_max(topic_nums, coherence_scores)

# Plot the elbow point
plt.scatter(elbow_point[0], elbow_point[1], marker='o', color='red')

# Add label to elbow point
plt.annotate(f'{elbow_point[0]}', xy = (elbow_point[0], elbow_point[1]), 
             xytext = (elbow_point[0] + 0.01, elbow_point[1] + 0.01), fontsize = 8)

plt.show()

### Running the Model

texts_nouns = output['texts_filter']

n_samples = len(texts_nouns)
n_features = output['n_features'] # from above at corpus
n_components = 7 # identified through the max/elbow solution above
n_top_words = 10


vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(texts_nouns)

model = NMF(
            n_components=n_components,
            random_state=83,
            beta_loss="kullback-leibler",
            init="nndsvda",
            solver="mu",
            max_iter=200, # identified though hyperparameter tuning
            alpha=0.1, # identified though hyperparameter tuning
            l1_ratio=0, # identified though hyperparameter tuning
        ).fit(tfidf)

scorer_sklearn(model, texts_nouns)

### Graph Topics

# the following reproduces Fig 4
tf_feature_names = vectorizer.get_feature_names()
plot_top_words_colors(model, tf_feature_names, n_top_words, "title")