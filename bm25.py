import pandas as pd

def tf_(doc):
    # initialize frequencies dictionary
    frequencies = {}

    # Iterate through all words in document and add into dictionary or increment value of occurrences 
    for term in doc: 
        term_count = frequencies.get (term, 0) + 1 
        frequencies[term] = term_count 
    
    return frequencies        

    # Return frequencies
    return frequencies

print("TF", tf_(['a','b','c','c','b','d']))

def df_(docs):
    # initialize df dictionary
    df = {}

    for doc in docs: 
        for term in set(doc): 
            df_count = df.get(term, 0) + 1 
            df[term] = df_count 

    return df

    # # Iterate through all docs
    # for doc in range(0,len(docs)):
    #     # Iterate through all terms in doc
    #     for term in docs[doc]:
    #         # If the term exists in df already, add the doc id to its set
    #         if term in df:
    #             temp = df[term]
    #             temp.add(doc)
    #             df[term] = temp
    #         # if the term does not exist, we initialize the value in the dictionary to a set with the doc id
    #         else:
    #             df[term] = {doc}    
    
    # # Map the values in the dictionary to be the length of the number of documents it is present in
    # for i in df:
    #     df[i] = len(df[i])      

    # # Return df
    # return df

a_df = df_([['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']])
print("DF", a_df)

def idf_(df, corpus_size):
    import math
    idf = {}
    for term, freq in df.items():
        idf[term] = round(math.log((corpus_size) / (freq)), 2)
    return idf

print("IDF", idf_(a_df, 3))

df = pd.read_pickle('goodreads_reviews_spoiler_processed_scraped.pkl')

def _score(query, doc, docs, tf, df, idf, avg_doc_len, k1=1.5, b=0.75):
    import math
    score = 0.0
    # tf = tf_(doc)
    # df = df_(docs)
    # avg_doc_len = sum([len(x) for x in docs])/len(docs)
    
    for term in query:
        if term not in tf.keys(): 
            continue

        numerator = idf[term] * tf[term] * (k1 + 1) 
        denominator = tf[term] + k1 * (1 - b + b * len(doc) / avg_doc_len) 
        score += (numerator / denominator)

        # numerator = ((k1 + 1) * tf[term])
        # denominator = ( (k1 * ((1 - b) + b * (len(doc)/avg_doc_len))) + tf[term] )
        # score += math.log(len(docs)/df[term]) * ( numerator / denominator )

    return round(score,2)

# query = ['b','c','e']
# doc = ['b', 'c', 'd']
# docs= [['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']]
docs = df['review_sentences'].tolist()
# print("Score", _score(query, doc, docs))

bm25_d = {}
rankings = {}
print("CALC DF AND IDF")
df_b = df_(docs)
idf = idf_(df_b, len(docs))
avg_doc_len = sum([len(x) for x in docs])/len(docs)

def calculate_bm25_rankings(query,doc, docs, tf, df_b, avg_doc_len):
        
    for i,book in df.iterrows():
        print(book.title)
        bm25_d[book.title] = {}
        rankings[book.title] = []

        if book.title not in bm25_d:
            bm25_d[book.title] = {}
        if book.title not in rankings:
            rankings[book.title] = []

        for j,other in df.iterrows():
            if i == j or other.title in bm25_d[book.title]:
                continue
            tf = tf_(other['review_sentences'])
            s = _score(book['review_sentences'], other['review_sentences'], docs, tf, df_b, idf, avg_doc_len)
            bm25_d[book.title][other.title] = s
            rankings[book.title].append((other['title'], s))
            if other.title not in bm25_d:
                bm25_d[other.title] = {}
            if other.title not in rankings:
                rankings[other.title] = []
            bm25_d[other.title][book.title] = s
            rankings[other.title].append((book['title'], s))


    for i in rankings:
        sorted(i, key=lambda x: x[1], reverse = True)
    print(bm25_d)
    print(rankings)
    bm25_d.to_pickle('bm25_matrix.pkl')
    rankings.to_pickle('bm25_rankings.pkl')

def calculate_bm25(book):
    rankings = []

    for i,row in df.iterrows():
        if row.title == book.title:
            continue
        tf = tf_(row['review_sentences'])
        s = _score(book.review_sentences, row.review_sentences, docs, tf, df, idf, avg_doc_len)
        rankings.append((book.title, s))
    
    return sorted(rankings, key=lambda x: x[1], reverse = True)[0:10]
print("CALC")
book = df.loc[df['title']=="Harry Potter and the Philosopher's Stone"]
print(book)
print(book.title)
print(calculate_bm25(book))

# def cos_sim(query, doc, docs, tf, df, avg_doc_len, k1=1.5, b=0.75):
#     score = 0.0

#     for term in query:
#         if term not in tf.keys(): 
#             continue

#         # numerator = ((k1 + 1) * tf[term])
#         # denominator = ( (k1 * ((1 - b) + b * (len(doc)/avg_doc_len))) + tf[term] )
#         # score += math.log(len(docs)/df[term]) * ( numerator / denominator )
#         # score += tf

#     return round(score,2)

# def calculate_cos_sim(book):
#     for i,row in df.iterrows():
#         if row.title == book.title:
#             continue
#         tf = tf_(row['review_sentences'])
#         s = _score(book.review_sentences, row.review_sentences, docs, tf, df, avg_doc_len)
#         rankings.append((book.title, s))
    
#     return sorted(rankings, key=lambda x: x[1], reverse = True)

