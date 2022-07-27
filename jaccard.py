import pandas as pd
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

def calculate_jaccard_matrix():
    # try: #Try Reading from Pickle
    #     print("Read from Pickle")
    # except (OSError, IOError) as e:
    #     # No pickle, Create Matrix
    #     print("Read from JSON")
    #     df = pd.read_json('goodreads_reviews_spoiler.json',lines=True)
    #     df.to_pickle('goodreads_reviews_spoiler.pkl')

    df = pd.read_pickle('goodreads_reviews_spoiler_processed_scraped.pkl')

    jaccard_matrix = {}

    # for i,row in df.iterrows():
        # jaccard_matrix = 

    # Store score for each book pair
    # OR
    # Store ranked list of recommendations
    # df.to_pickle('jaccard_matrix.pkl')



