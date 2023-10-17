# created by Kamila Kolpashnikova 2023
# replication code for "Fear of falling: Scoping review and topic analysis using natural language processing" published in PLOS ONE 2023

# REQUIREMENTS:
# Python    3.9.7
# pandas    1.4.2
# pybliometrics 3.3.0
# numpy 1.22.3
# matplotlib    3.5.1
# scikit-learn  1.2.2
# seaborn   0.11.2
# networkx  2.6.3
# spacy 3.4.4


### Load packages

import pandas as pd
from pybliometrics.scopus import ScopusSearch, AuthorSearch, AuthorRetrieval
import numpy as np
import matplotlib.pyplot as plt
import string
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import networkx as nx
import spacy
nlp = spacy.load('en_core_web_md')
import datetime


### Names for merging

# I created a table for cross-referencing across different databases, which you can download from data folder:
col_names = pd.read_excel('data/Cross-reference for column names.xlsx', header = 0)


### Scopus data

# Getting publications from Scopus using pybliometrics
s = ScopusSearch("TITLE( {fear of falling} ) AND ALL ( ( elder* OR senior*or AND pensioner* OR ( ( old* OR aged* ) W/3 ( adult* OR people* OR person* ) ) ) OR ( ( long-term AND care ) OR ( nursing AND home* ) OR ( assisted AND living ) OR ( home W/3 aged ) ) )")

scopus = pd.DataFrame(s.results)

### Loading data from other databases

# You need to save your search data in data folder
# Data file names will be the ones you use, not the ones here

# function to load all databases
def createDB():
    dbs = dict()

    db_names = {
        'Medline': 'data/searches/FoF Medline 2023-01-24.xls',
        'Embase': 'data/searches/FoF Embase 2023-01-24.xls',
        'PsycInfo':'data/searches/FoF Psycinfo 2023-01-24.xls',
        'CINAHL': 'data/searches/FoF CINAHL 2023-01-24.xls',
        'Web of Science': 'data/searches/FoF WoS 2023-01-24.xls'    
    }

    for i in db_names.keys():
        if i in ('Medline', 'Embase', 'PsycInfo'):
            columns = [j.strip() for j in (", ").join([col for col in col_names[col_names['Database']==i].iloc[0, 1:] if isinstance(col, str)]).split(", ")]
            
            dbs[i] = pd.read_excel(db_names[i], header = 1)[columns]

        if i in ('CINAHL', 'Web of Science'):
            columns = [j.strip() for j in (", ").join([col for col in col_names[col_names['Database']==i].iloc[0, 1:] if isinstance(col, str)]).split(", ")]
            
            dbs[i] = pd.read_excel(db_names[i], header = 0)[columns]
            
            
    return dbs

# Load databases into db
db = createDB()


# Creating dictionary for renaming columns of databases for merging
renaming_dict = dict()

for i in range(len(col_names)):
    renaming_dict[col_names.loc[i, 'Database']] = dict(zip([a.strip() if isinstance(a, str) else math.nan for a in col_names.iloc[i, 1:]], col_names.columns[1:]))


# Selecting needed columns from scopus data
columns = [j.strip() for j in (", ").join([col for col in col_names[col_names['Database']=='Scopus'].iloc[0, 1:] if isinstance(col, str)]).split(", ")]
db['Scopus'] = scopus[columns]


# Web of Science keywords
db['Web of Science'][['Author Keywords', 'Keywords Plus']] = db['Web of Science'][['Author Keywords', 'Keywords Plus']].fillna("")
db['Web of Science'][['Author Keywords', 'Keywords Plus']]
db['Web of Science']['Author Keywords'] = ['; '.join(filter(None, i)) for i in zip(db['Web of Science']['Author Keywords'], [word.lower() for word in db['Web of Science']['Keywords Plus']])]
renaming_dict['Web of Science']['Keywords'] = renaming_dict['Web of Science']['Author Keywords, Keywords Plus'] 
renaming_dict['Web of Science'].pop('Author Keywords, Keywords Plus', None)

# clean nans in renaming dictionary
for keys in renaming_dict:
    renaming_dict[keys] = {k: renaming_dict[keys][k] for k in renaming_dict[keys] if not isinstance(k, float)}
            
# standardize column names 
for k,df in db.items():
    df.rename(columns=renaming_dict[k], inplace=True)

# creating full dataframe with all databases
df_full = pd.concat([
    db['Medline'],
    db['Embase'],
    db['PsycInfo'],
    db['CINAHL'],
    db['Web of Science'],
    db['Scopus']
          ])[col_names.columns[1:]]

### Cleaning

# get all nan into strings
df_full = df_full.fillna("")

# remove extra endings in titles -- b/c some naming conventions adds the type of study at the end
df_full['title'] = [i.split("[")[0].strip() if len(i.split("[")) >1 else i for i in df_full['Title']]

# make all to lowercase to compare by title
df_full['title'] = [i.lower().translate(str.maketrans('','',string.punctuation)) for i in df_full['title']]

# create short authors for comparing by authors
df_full['auth'] = [i.lower()[0:3] for i in df_full['Authors']]

# drop duplicates by title and short author
df = df_full.drop_duplicates(['title', 'auth'])

# clean different starts for DOI strings first
df['doi'] = [i.split('doi.org/')[1] if len(i.split('doi.org/'))>1 else i for i in df['DOI']]

# remove duplicates by doi
df = df[(~df['doi'].duplicated()) | (df['doi'] == "")] 

# clean by publication type
ren_type = {'Journal\n\nPeer Reviewed Journal': 'Article', 
            'Review': 'Article',
            'Conference Abstract': 'Conference',
            'Meeting Abstract': 'Meeting',
            'Systematic Review.': 'Article',
       'Article; Proceedings Paper': 'Article',
       'Book\n\nEdited Book': 'Book',
       'Book\n\nAuthored Book': 'Book',
       'Article; Early Access': 'Article', 'Review; Early Access': 'Article',
            'Article; Book Chapter': 'Book Chapter'
           }

df["Type"] = ["Article" if set(i.strip().split("\n\n")).intersection({"Journal Article"}) == {"Journal Article"} else i.strip() for i in df["Type"]]
df["Type"] = ["Letter" if set(i.strip().split("\n\n")).intersection({"Letter"}) == {"Letter"} else i.strip() for i in df["Type"]]
df["Type"] = ["Comment" if set(i.strip().split("\n\n")).intersection({"Comment"}) == {"Comment"} else i.strip() for i in df["Type"]]


df=df.replace({"Type": ren_type})

# keep only original research
df = df[(df['Type'] == 'Article') | (df['Type'] == 'Conference')| (df['Type'] == 'Book Chapter') 
       | (df['Type'] == 'Dissertation Abstract')| (df['Type'] == 'Conference Paper')| (df['Type'] == 'Proceedings Paper')
       | (df['Type'] == 'Thesis')] 

# remove empty titles and abstracts
df = df.fillna('')
df = df[(df['title']!='') & (df['Abstract']!='')]
df.reset_index(inplace=True)

### Pairwise Similarity for Titles (for deduplications)

corpus = df['title']
vect = TfidfVectorizer(min_df=1, stop_words="english") 
tfidf = vect.fit_transform(corpus)
cosine_s = cosine_similarity(tfidf.toarray())

# instead of by value you need to use the diagonal
# change all cosine dist in diagonal = -1
for i in range(len(cosine_s)):
    cosine_s[i][i] = -1
    
# calculations work faster if using np.arrays
cos_idxs = []
for i in cosine_s:
    cos_idxs.append(np.argmax(i))

# using a cut-off of 0.90
cos_idxs = np.argwhere(cosine_s>0.90)

### Deduplication with Word2Vec

# Word2Vec is used in spaCy to create word vectors, one important thing to note: 
# In order to create word vectors we need larger spaCy models. For example, 
# the medium or large English model, but not the small one. 
# So if we want to use vectors, we will go with a model that ends in 'md' or 'lg'. 
# 
# Install spaCy and download one of the larger models:
# here: https://newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python

docs = [nlp(abst) for abst in df['title']]
similarity = []
for i in range(len(docs)):
    row = []
    for j in range(len(docs)):
        row.append(docs[i].similarity(docs[j]))
    similarity.append(row)


similarity = np.array(similarity)
similarity[similarity >= 1] = -1

# calculations work faster if using np.arrays
idxs = []
for i in similarity:
    idxs.append(np.argmax(i))
    
## using arbitrary cut-off 0.9998
idxs = np.argwhere(similarity>0.9998)
idxs

### Check which similarity measure is better for id'ing

plt.hist(similarity[similarity<1].flatten())
plt.show()

plt.hist(cosine_s[cosine_s >= 0.2].flatten())
plt.show()

# from above seems like cosine similarity is better for deduplication, but
# the other one might be better for finding repetitive works, so that's one more exclusion criteria


### Clean Foreign Language

df['language'] = [i.split('Language: ')[1].split('. Entry Date:')[0] if i[0:5] == 'Acces' else i for i in df['Language']]

# keep only English publications and publications for which language data was not available
df_d = df[(df['language']=='English') | (df['language']=='')]

### Re-check using word2vec

df_d.reset_index(inplace=True, drop=True)

docs = [nlp(abst) for abst in df_d['title']]
similarity = []
for i in range(len(docs)):
    row = []
    for j in range(len(docs)):
        row.append(docs[i].similarity(docs[j]))
    similarity.append(row)

similarity = np.array(similarity)
similarity.shape

# similarity[similarity >= 1] = -1
for i in range(len(similarity)):
    similarity[i][i] = -1

# calculations work faster if using np.arrays
idxs = []
for i in similarity:
    idxs.append(np.argmax(i))
idxs = np.argwhere(similarity>0.99)

df_d.reset_index(inplace=True, drop=True)

# save the cleaned data
df_d.to_csv('data/df_d.csv')