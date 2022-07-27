# doc1 = "we love information retrieval course".split()
# doc2 = "information retrieval is a course offered in sutd".split()

# print(jaccard_similarity(doc1, doc2))

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

a_df = df_([['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']])
print("DF", a_df)

def idf_(df, corpus_size):
    import math
    idf = {}
    for term, freq in df.items():
        idf[term] = round(math.log((corpus_size) / (freq)), 2)
    return idf

print("IDF", idf_(a_df, 3))


def _score(query, doc, docs, k1=1.5, b=0.75):
    import math
    score = 0.0
    tf = tf_(doc)
    df = df_(docs)
    idf = idf_(df, len(docs))
    avg_doc_len = sum([len(x) for x in docs])/len(docs)
    
    for term in query:
        if term not in tf.keys(): 
            continue

        numerator = ((k1 + 1) * tf[term])
        denominator = ( (k1 * ((1 - b) + b * (len(doc)/avg_doc_len))) + tf[term] )
        score += math.log(len(docs)/df[term]) * ( numerator / denominator )

    return round(score,2)

query = ['b','c','e']
doc = ['b', 'c', 'd']
docs= [['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']]
# docs = df['review_sentences'].tolist()
print("Score", _score(query, doc, docs))

# For each book in df
    # For other book in df
        # s = score(book['review_sentences'], other['review_sentences'], docs)
        # bm25_d[book][other] = s
        # rankings[book].append((other['title'], s))

# for i in rankings:
#   sorted(i, key=lambda x: x[1], reverse = True)
# bm25_d.to_pickle('bm25_matrix.pkl')
# rankings.to_pickle('bm25_rankings.pkl')

