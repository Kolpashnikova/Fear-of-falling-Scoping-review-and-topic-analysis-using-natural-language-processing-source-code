# Fear of falling: Scoping review and topic analysis using natural language

Kamila Kolpashnikova 
2023-10-16

## Overview

The code in this repo is replication of the paper "Fear of falling: Scoping review and
topic analysis using natural language" published in PLOS ONE 2023. Python files 
for replication are located in 'src' folder. Additional data, such as a curated
list of stopwords and cross-references across different databases, are in the 
'data' folder.

## Data Availability

The data should be extracted by researchers with the help of librarians from the
following databases:
- Medline
- Embase
- PsycInfo
- CINAHL
- Web of Science
- Scopus

## Software Requirements

- Python    3.9.7
- pandas    1.4.2
- pybliometrics 3.3.0
- numpy     1.22.3
- matplotlib    3.5.1
- scikit-learn  1.2.2
- seaborn   0.11.2
- networkx  2.6.3
- spacy     3.4.4
- gensim    4.1.2
- joblib    1.2.0
- scipy     1.7.1

## Instructions to Replicators

- get data from bibliographic databases
- install python and above-mentioned packages
- run lines (line by line) from 1-Data-Cleaning-2023-01-24-FoF-Paper.py to clean data
- run lines from 2-Models-And-Graphing-2023-01-24-FoF-Paper to run models and graph 

## Citations

Kolpashnikova, K., Harris, L.R., and Desai, S. (2023). Fear of falling: Scoping review and topic analysis using natural language processing. *PLOS ONE*. 