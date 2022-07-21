
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

# kaggle.api.authenticate()
# kaggle.api.dataset_download_file('dorianlazar/medium-articles-dataset', file_name='medium_data.csv',  path='data/')

# ! pip install -q kaggle
# ! kaggle datasets download -d sohelranaccselab/goodreads-book-reviews

try:
    df = pd.read_pickle('goodreads_reviews_spoiler.pkl')
    print("Read from Pickle")
except (OSError, IOError) as e:
    print("Read from JSON")
    df = pd.read_json('goodreads_reviews_spoiler.json',lines=True)
    df.to_pickle('goodreads_reviews_spoiler.pkl')



#print(df["review_sentences"].head())
#print(df.columns.tolist())
print(df.columns.tolist())
test_df  = df.head(100)

print(test_df["review_sentences"].head())

def flatten(lst):
    return [x[1] for x in lst]


# bm25, Assignmt 3

def remove_stopwords_and_lemmatize(lst):
    filtered=[]
    
    for ls in lst:
       # tokens = ls.split()
     #   tokens = re.split('\W+', ls)

        tokens = word_tokenize(ls)
        # remove all tokens that are not alphabetic
        # words = [word for word in tokens if word.isalpha()]

        for word in tokens:
            if word.lower() not in stop_words and word.isalpha():
                filtered.append(lemmatizer.lemmatize(word.lower()))
    return filtered        

test_df["review_sentences"] = test_df["review_sentences"].apply(flatten)
print("FLATTEN")
print(test_df["review_sentences"].head())

print("0", test_df["book_id"][0])
print("12", test_df["book_id"][12])
print("2", test_df["book_id"][2])
print("15", test_df["book_id"][15])
print("16", test_df["book_id"][16])
print("85", test_df["book_id"][85])
#print("85",test_df["review_sentences"][85], test_df["book_id"][85])


test_df["review_sentences"] = test_df["review_sentences"].apply(remove_stopwords_and_lemmatize)
# REMOVE STOPWORDS AND LEMMATIZE TAKES TOO LONG
# SAVE PRE PROCESSED DF INSTEAD OF JUST LOADED DF
print(test_df["review_sentences"].head())

'''
    Reviews are combined and preprocessed for stopwords, punctuation and lemmatization
    Still need to group reviews by book id, (perhaps use group_by)

    Also need to scrape the book title off goodreads site and keep it in the dataframe or use the api
'''

def jaccard_similarity(doc1, doc2):
    # print("doc1", doc1, "doc2", doc2)
    union = len(set(doc1 + doc2))
    intersection = len(set(doc1) & set(doc2))

    return intersection/union

'''
Loop through all other rows
run jaccard on reviews
keep track of score, review
sort
'''

def ranked_jaccard_similarity(index):
    ranks = []
    reviews = test_df['review_sentences'][index]

    for i,row in test_df.iterrows():
        if i == index:
            continue
        else:
            rev = row['review_sentences']
            js = jaccard_similarity(reviews, rev)
           # print("jaccard", js)
            ranks.append((i, js))

    return sorted(ranks, key=lambda x: x[1], reverse = True)

print("RANKS")
print(ranked_jaccard_similarity(0)[0:15])



# doc1 = "we love information retrieval course".split()
# doc2 = "information retrieval is a course offered in sutd".split()

# print(jaccard_similarity(doc1, doc2))


# #export PATH="$HOME/Library/Python/3.8/bin:$PATH"


# def tf_(doc):
#     # initialize frequencies dictionary
#     frequencies = {}

#     # Iterate through all words in document and add into dictionary or increment value of occurrences 
#     for i in doc:
#         frequencies[i] = frequencies[i] + 1 if i in frequencies else 1            

#     # Return frequencies
#     return frequencies

# print("TF", tf_(['a','b','c','c','b','d']))

# def df_(docs):
#     # initialize df dictionary
#     df = {}

#     # Iterate through all docs
#     for doc in range(0,len(docs)):
#         # Iterate through all terms in doc
#         for term in docs[doc]:
#             # If the term exists in df already, add the doc id to its set
#             if term in df:
#                 temp = df[term]
#                 temp.add(doc)
#                 df[term] = temp
#             # if the term does not exist, we initialize the value in the dictionary to a set with the doc id
#             else:
#                 df[term] = {doc}    
    
#     # Map the values in the dictionary to be the length of the number of documents it is present in
#     for i in df:
#         df[i] = len(df[i])      

#     # Return df
#     return df

# print("DF", df_([['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']]))

# def idf_(df, corpus_size):
#     import math
#     idf = {}
#     for term, freq in df.items():
#         idf[term] = round(math.log((corpus_size) / (freq)), 2)
#     return idf


# def _score(query, doc, docs, k1=1.5, b=0.75):
#     import math
#     score = 0.0
#     tf = tf_(doc)
#     df = df_(docs)
#     idf = idf_(df, len(docs))
#     avg_doc_len = sum([len(x) for x in docs])/len(docs)
    
#     for term in query:
#         if term not in tf.keys(): 
#             continue

#         numerator = ((k1 + 1) * tf[term])
#         denominator = ( (k1 * ((1 - b) + b * (len(doc)/avg_doc_len))) + tf[term] )
#         score += math.log(len(docs)/df[term]) * ( numerator / denominator )

#     return round(score,2)

# query = ['b','c','e']
# doc = ['b', 'c', 'd']
# docs= [['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']]
# print("Score", _score(query, doc, docs))




