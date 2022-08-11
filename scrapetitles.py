import requests
from bs4 import BeautifulSoup
# import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# # import nltk
# # nltk.download('stopwords')
# # nltk.download('wordnet')
# # nltk.download('omw-1.4')
# # from nltk.corpus import stopwords
# # from nltk.stem import WordNetLemmatizer

# # from nltk.tokenize import word_tokenize

# # lemmatizer = WordNetLemmatizer()

# # stop_words = set(stopwords.words('english'))

# # kaggle.api.authenticate()
# # kaggle.api.dataset_download_file('dorianlazar/medium-articles-dataset', file_name='medium_data.csv',  path='data/')

# # ! pip install -q kaggle
# # ! kaggle datasets download -d sohelranaccselab/goodreads-book-reviews

try:
    print("Read from Pickle")
    df = pd.read_pickle('goodreads_reviews_spoiler_processed.pkl')
except (OSError, IOError) as e:
    print("Read from JSON")
    df = pd.read_json('goodreads_reviews_spoiler.json',lines=True)
    df.to_pickle('goodreads_reviews_spoiler.pkl')

# df = df.reset_index()
# df["title"] = np.nan

# print("Read New Books File")


# # booksdf = df.drop([ a_i - b_i for a_i, b_i in zip(booksdf.columns.tolist(), cols)], axis=1)
# print(booksdf.columns.tolist())
# print(booksdf.head())
# booksdf.rename(columns = {'best_book_id':'book_id', 'original_title':'title'}, inplace = True)

# print(booksdf.head())

# col = 'book_id'
# cols_to_replace = ['title']
# df.loc[df[col].isin(booksdf[col]), cols_to_replace] = booksdf.loc[booksdf[col].isin(df[col]),cols_to_replace].values

# # df.loc[:, ['title']] = booksdf[['Latitude', 'Longitude']]

# # df['title'] = booksdf.loc[df['book_id'] == booksdf['bookID']]['title'] 

# #print(df["review_sentences"].head())
# #print(df.columns.tolist())

# print(df.reset_index().columns.tolist())
# print(df.reset_index().head(10))
# print(df.reset_index().tail(3))

# print("Books", len(booksdf))
# print("Spoilers Books", len(df))
# print("NAN", df['title'].isna().sum())
# print("BLANK", (df['title'].values == '').sum())
# print("Len", len(df))
# df = df[df['title'].notna()]
# df = df[df['title'].values != '']
# print("Len", len(df))

# # URL = "https://www.goodreads.com/book/show/22816087"
# # print(URL)
# # page = requests.get(URL)
# # soup = BeautifulSoup(page.content, "html.parser")
# # bookTitle = soup.find(id="bookTitle").text.strip()
# # print(bookTitle)

def scrape(bookID):
    URL = "https://www.goodreads.com/book/show/"+str(bookID)
    print(URL)
    page = requests.get(URL)
    # soup = BeautifulSoup(html, "lxml")
    soup = BeautifulSoup(page.content, "html.parser")
    try:    
        bookTitle = soup.find(id="bookTitle").text.strip()
        print(bookTitle)
        return bookTitle  
    except:
        print("an error occured")
        return None

# # test_df['book_title'] = scrape(test_df['book_id'])
# # print(test_df.head())
test_df['book_title'] = test_df['book_id'].apply(lambda x : scrape(x))
print(test_df.head())


def scrape_df(df):
    nullCounter = 0
    for i,row in df.iterrows():
        
        print(row["book_id"])
        print(i)
        bookID = row["book_id"]
        URL = "https://www.goodreads.com/book/show/"+str(bookID)
        print(URL)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")
        try:    
            bookTitle = soup.find(id="bookTitle").text.strip()
            print(bookTitle)
            row["bookTitle"] = bookTitle    
        except:
            print("an error occured")
            nullCounter = nullCounter + 1
    
    return df     
        
        
    
    
#         #print(booktitle)
#     # for i,row in df.iterrows():
#     #     bookID = row["book_id"]
#     #     URL = "https://www.goodreads.com/book/show/"+str(bookID)
#     #     page = requests.get(URL)
#     #     soup = BeautifulSoup(page.content, "html.parser")
#     #     bookTitle = str(soup.find(id="bookTitle").text).strip()
#     #     row[bookTitle] = bookTitle
#         #print(booktitle)
    

import concurrent.futures

MAX_THREADS = 30

def download_url(url):
    print(url)
    resp = requests.get(url)
    title = ''.join(x for x in url if x.isalpha()) + "html"
    
    with open(title, "wb") as fh:
        fh.write(resp.content)
        
    time.sleep(0.25)
    
def download_stories(story_urls):
    threads = min(MAX_THREADS, len(story_urls))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        executor.map(download_url, story_urls)

def main(story_urls):
    t0 = time.time()
    download_stories(story_urls)
    t1 = time.time()
    print(f"{t1-t0} seconds to download {len(story_urls)} stories.")

# # test_df=scrape_df(test_df)     

# # print(test_df["bookTitle"])

# # print("program done")

df = pd.read_pickle('goodreads_reviews_spoiler_processed_scraped.pkl')
# df.to_pickle('scraped_protocol_4.pkl')
df.to_csv('scraped_CSV.csv')