
import numpy as np # linear algebra
# import os
# print("hello",os.path)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import nltk
from nltk.corpus import stopwords

import nltk
import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
import re
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))


import json

# person = '{"name": "Bob", "languages": ["English", "French"]}'

# # # df.to_pickle('filename.pkl', protocol=4)
# data = []

# with open('goodreads_reviews_spoiler.json', "r") as f:
#     # df = pickle.load(fh)
#     # for line in f:
#     #     data = f.readlines()
#     #     data_json_str = "[" + ','.join(str(data)) + "]"
#     #     # data_df = pd.read_json(data_json_str)

#     data = f.readlines()
#     print(data[0])
#     # data_json_str = "[" + ','.join(data) +  "]"
#     # data_df = pd.read_json(data_json_str)
#     # print(data_json_str[0:300])
#     # print(data_json_str[-50:])

#     # data_json_str = "{" + data_json_str +  "}"
#     book_dict = json.load(data)

# import preprocess
# import jaccard
# import bm25

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
df = df.drop(['user_id', 'timestamp', 'rating', 'has_spoiler', 'review_id'], axis=1)
# test_df  = df.head(100)
test_df  = df

# print(test_df["review_sentences"].head())

def flatten(lst):
    return [x[1] for x in lst]


# # bm25, Assignmt 3

def remove_stopwords_and_lemmatize(lst):
    filtered=[]
    
    for ls in lst:
        # tokens = ls.split()
        # tokens = re.split('\W+', ls)

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

# print("0", test_df["book_id"][0])
# print("12", test_df["book_id"][12])
# print("2", test_df["book_id"][2])
# print("15", test_df["book_id"][15])
# print("16", test_df["book_id"][16])
# print("85", test_df["book_id"][85])
#print("85",test_df["review_sentences"][85], test_df["book_id"][85])


print("Remove Stopwords and Lemmatize")
test_df["review_sentences"] = test_df["review_sentences"].apply(remove_stopwords_and_lemmatize)
# REMOVE STOPWORDS AND LEMMATIZE TAKES TOO LONG
# SAVE PRE PROCESSED DF INSTEAD OF JUST LOADED DF
# print(test_df["review_sentences"].head())

# print(test_df['book_id'].nunique())

# print("GROUP BY")
# def group_by_books(df):
#     return df.groupby(['book_id'])['review_sentences'].agg(sum)

print("curr len", len(test_df))
# print(test_df.head())

print("Groupby Book_ID")
test_df = test_df.groupby(['book_id'])['review_sentences'].agg(sum)
# print(test_df.head())

print("after group len:",len(test_df))
print("PICKLE")
test_df.to_pickle('goodreads_reviews_spoiler_processed.pkl')
'''
    Reviews are combined and preprocessed for stopwords, punctuation and lemmatization
    Still need to group reviews by book id, (perhaps use group_by)

    Also need to scrape the book title off goodreads site and keep it in the dataframe or use the api
'''



