import requests
from bs4 import BeautifulSoup
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
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
test_df  = df.head(10)


URL = "https://www.goodreads.com/book/show/22816087"
print(URL)
page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")
bookTitle = soup.find(id="bookTitle").text.strip()
print(bookTitle)

def scrape_df(df):
    idCol = df["book_id"]
    print(idCol)
    titles = []
    for bid in idCol:
        bookID = bid
        URL = "https://www.goodreads.com/book/show/"+str(bookID)
        print(URL)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")
        
        bookTitle = soup.find(id="bookTitle").text.strip()
        print(bookTitle)
        #titles.append(bookTitle)
        
    #df["bookTitle"] = titles    
    
    
        #print(booktitle)
    # for i,row in df.iterrows():
    #     bookID = row["book_id"]
    #     URL = "https://www.goodreads.com/book/show/"+str(bookID)
    #     page = requests.get(URL)
    #     soup = BeautifulSoup(page.content, "html.parser")
    #     bookTitle = str(soup.find(id="bookTitle").text).strip()
    #     row[bookTitle] = bookTitle
        #print(booktitle)
    return df   


test_df=scrape_df(test_df)     

#print(test_df["bookTitle"])
print("program done")
