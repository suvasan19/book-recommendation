
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

# kaggle.api.authenticate()
# kaggle.api.dataset_download_file('dorianlazar/medium-articles-dataset', file_name='medium_data.csv',  path='data/')

# ! pip install -q kaggle
# ! kaggle datasets download -d sohelranaccselab/goodreads-book-reviews


df = pd.read_json('goodreads_reviews_spoiler.json',lines=True)

#print(df["review_sentences"].head())
#print(df.columns.tolist())
print(df.columns.tolist())
test_df  = df.head()

def flatten(lst):
    joinedlist = []

    for elem in lst:
        if type(elem) is list:
            for item in elem:
                if item != 0 and item !=1:
                    joinedlist.append(item)
        else:
            if item != 0 and item !=1:
                joinedlist.append(elem)
    return joinedlist   

def remove_stopwords_and_lemmatize(lst):
    filtered=[]
    tokens = lst[0].split()
    for word in tokens:
        print(word)
        if word.lower() not in stop_words:
            filtered.append(lemmatizer.lemmatize(word.lower()))
    return filtered        

test_df["review_sentences"] = test_df["review_sentences"].apply(flatten)
test_df["review_sentences"] = test_df["review_sentences"].apply(remove_stopwords_and_lemmatize)

print(test_df["review_sentences"].head())


def jaccard_similarity(doc1: str, doc2: str):
    
    union = len(set(doc1 + doc2))
    intersection = len(set(doc1) & set(doc2))

    return intersection/union

doc1 = "we love information retrieval course".split()
doc2 = "information retrieval is a course offered in sutd".split()

print(jaccard_similarity(doc1, doc2))


#export PATH="$HOME/Library/Python/3.8/bin:$PATH"
# bm25, Assignmt 3


def tf_(doc):
    # initialize frequencies dictionary
    frequencies = {}

    # Iterate through all words in document and add into dictionary or increment value of occurrences 
    for i in doc:
        frequencies[i] = frequencies[i] + 1 if i in frequencies else 1            

    # Return frequencies
    return frequencies

print("TF", tf_(['a','b','c','c','b','d']))

def df_(docs):
    # initialize df dictionary
    df = {}

    # Iterate through all docs
    for doc in range(0,len(docs)):
        # Iterate through all terms in doc
        for term in docs[doc]:
            # If the term exists in df already, add the doc id to its set
            if term in df:
                temp = df[term]
                temp.add(doc)
                df[term] = temp
            # if the term does not exist, we initialize the value in the dictionary to a set with the doc id
            else:
                df[term] = {doc}    
    
    # Map the values in the dictionary to be the length of the number of documents it is present in
    for i in df:
        df[i] = len(df[i])      

    # Return df
    return df

print("DF", df_([['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']]))

def idf_(df, corpus_size):
    import math
    idf = {}
    for term, freq in df.items():
        idf[term] = round(math.log((corpus_size) / (freq)), 2)
    return idf

def _score(query, doc, docs, k1=1.5, b=0.75):
    import math
    score = 0.0
    tf = tf_(doc)
    df = df_(docs)
    idf = idf_(df, len(docs))
    avg_doc_len = sum([len(x) for x in docs])/len(docs)
    
    for term in query:
        if term not in tf. keys(): 
            continue

        numerator = ((k1 + 1) * tf[term])
        denominator = ( (k1 * ((1 - b) + b * (len(doc)/avg_doc_len))) + tf[term] )
        score += math.log(len(docs)/df[term]) * ( numerator / denominator )

    return round(score,2)

query = ['b','c','e']
doc = ['b', 'c', 'd']
docs= [['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']]
print("Score", _score(query, doc, docs))




