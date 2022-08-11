import pandas as pd
def jaccard_similarity(doc1, doc2):
    # print("doc1", doc1, "doc2", doc2)\
    union = len(set(doc1 + doc2))
    intersection = len(set(doc1) & set(doc2))

    return intersection/union

# def ranked_jaccard_similarity(index):
#     ranks = []
#     reviews = test_df['review_sentences'][index]

#     for i,row in test_df.iterrows():
#         if i == index:
#             continue
#         else:
#             rev = row['review_sentences']
#             js = jaccard_similarity(reviews, rev)
#             # print("jaccard", js)
#             ranks.append((i, js))

#     return sorted(ranks, key=lambda x: x[1], reverse = True)

# print("RANKS")
# print(ranked_jaccard_similarity(0)[0:15])

# def calculate_jaccard_matrix():
    # try: #Try Reading from Pickle
    #     print("Read from Pickle")
    # except (OSError, IOError) as e:
    #     # No pickle, Create Matrix
    #     print("Read from JSON")
    #     df = pd.read_json('goodreads_reviews_spoiler.json',lines=True)
    #     df.to_pickle('goodreads_reviews_spoiler.pkl')
print("read pickle")
df = pd.read_pickle('goodreads_reviews_spoiler_processed_scraped.pkl')

import multiprocessing

# create as many processes as there are CPUs on your machine
num_processes = 1
print(num_processes)

# calculate the chunk size as an integer
chunk_size = int(df.shape[0]/num_processes)

# this solution was reworked from the above link.
# will work even if the length of the dataframe is not evenly divisible by num_processes
# chunks = [df.iloc[df.index[i:i + chunk_size]] for i in range(0, df.shape[0], chunk_size)]
chunks = [df.iloc[i:i + chunk_size,:] for i in range(0, df.shape[0], chunk_size)]

jaccard_matrix = {}
jaccard_rank = {}

print("Loop books")

# for i in jaccard_rank:
#     sorted(i, key=lambda x: x[1], reverse = True)

def func(df):
    for i,book in df.iterrows():
        print(book)
        reviews = book['review_sentences']
        if book.title not in jaccard_matrix:
            jaccard_matrix[book.title] = {}
        if book.title not in jaccard_rank:
            jaccard_rank[book.title] = []

        for j,other in df.iterrows():
            if i == j or other.title in jaccard_matrix[book.title]:
                continue

            rev = other['review_sentences']
            jaccard_score = jaccard_similarity(reviews, rev)
            jaccard_matrix[book.title][other.title] = jaccard_score
            jaccard_rank[book.title].append((other['title'], jaccard_score))
            if other.title not in jaccard_matrix:
                jaccard_matrix[other.title] = {}
            if other.title not in jaccard_rank:
                jaccard_rank[other.title] = []
            jaccard_matrix[other.title][book.title] = jaccard_score
            jaccard_rank[other.title].append((book['title'], jaccard_score))


# create our pool with `num_processes` processes
pool = multiprocessing.Pool(processes=num_processes)

# apply our function to each chunk in the list
result = pool.map(func, chunks)

for i in range(len(result)):
       # since result[i] is just a dataframe
   # we can reassign the original dataframe based on the index of each chunk
   df.iloc[result[i].index] = result[i]

jaccard_matrix.to_pickle('jaccard_matrix.pkl')
jaccard_rank.to_pickle('jaccard_rank.pkl')