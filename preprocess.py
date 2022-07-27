
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
        # tokens = re.split('\W+', ls)

        tokens = word_tokenize(ls)
        # remove all tokens that are not alphabetic
        # words = [word for word in tokens if word.isalpha()]

        for word in tokens:
            if word.lower() not in stop_words and word.isalpha():
                filtered.append(lemmatizer.lemmatize(word.lower()))
    return filtered        


def group_by_books(df):
    return df.group_by(['book_id'])['review_sentences'].apply(list)




