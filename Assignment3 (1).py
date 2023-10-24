#!/usr/bin/env python
# coding: utf-8

# In[9]:


import nltk
nltk.download('inaugural')


# In[2]:


from nltk.corpus import inaugural

inaugural.fileids()


# In[8]:


address = inaugural.raw('1861-Lincoln.txt')  # Change the filename to the one you want to read


# In[5]:


import nltk
from nltk.corpus import inaugural
from nltk.tokenize import word_tokenize
import math

# Load the NLTK corpus of inaugural addresses
nltk.download('inaugural')
inaugural_addresses = inaugural.fileids()

# Define the queries from HW 2
queries = {
    1: "government budget",
    2: "foreign policy",
    3: "economic growth",
    4: "environmental protection",
    5: "civil rights"
}

# Define the Jelinek-Mercer (JM) smoothing function
def jm_smoothing(query, document, lambda_value):
    query_tokens = word_tokenize(query.lower())
    doc_tokens = word_tokenize(document.lower())
    
    query_term_count = {term: query_tokens.count(term) for term in set(query_tokens)}
    doc_term_count = {term: doc_tokens.count(term) for term in set(query_tokens)}
    
    doc_length = len(doc_tokens)
    jm_score = 0.0
    
    for term in query_term_count:
        p_term_given_doc = (lambda_value * (doc_term_count[term] / doc_length)) + ((1 - lambda_value) * (query_term_count[term] / len(query_tokens)))
        jm_score += math.log(p_term_given_doc)
    
    return jm_score

# Create a function to rank documents for a query with a given lambda
def rank_documents(query, lambda_value):
    ranked_docs = []
    for doc_id in inaugural_addresses:
        doc_text = inaugural.raw(doc_id)
        score = jm_smoothing(query, doc_text, lambda_value)
        ranked_docs.append((doc_id, score))
    
    # Sort the documents by score in descending order
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_docs[:5]

# Compute precision at 5 for a given query and lambda
def compute_precision_at_5(query_id, lambda_value):
    query_text = queries[query_id]
    ranked_docs = rank_documents(query_text, lambda_value)
    
    # Assuming the first 5 inaugural addresses are relevant
    relevant_docs = inaugural_addresses[:5]
    
    # Calculate precision at 5
    precision = len(set(relevant_docs).intersection(set([doc_id for doc_id, _ in ranked_docs]))) / 5.0
    
    return precision

# Compute and print the results for each query and lambda
for query_id in queries:
    print(f"Query {query_id}: '{queries[query_id]}'")
    for lambda_value in [0.2, 0.5]:
        ranked_docs = rank_documents(queries[query_id], lambda_value)
        print(f"Lambda = {lambda_value}:")
        for doc_id, score in ranked_docs:
            print(f"Document Title: {doc_id}")
            print(f"Similarity Score: {score:.4f}")
        precision = compute_precision_at_5(query_id, lambda_value)
        print(f"Precision at 5: {precision:.4f}\n")


# In[7]:


import nltk
from nltk.corpus import inaugural
from nltk.util import ngrams
import math

# Access the list of inaugural addresses
inaugural_files = inaugural.fileids()

# Define the two values of mu
mu_values = [100, 1000]  # Adjust these values as needed

# Define a function to calculate Dirichlet Prior score for a term
def dirichlet_score(term_freq, doc_length, collection_length, mu):
    return math.log((term_freq + mu * collection_length / doc_length) / (collection_length + mu))

# Initialize a dictionary to store document lengths
doc_lengths = {}

# Calculate document lengths
for file_id in inaugural_files:
    words = inaugural.words(file_id)
    doc_lengths[file_id] = len(words)

# Initialize a dictionary to store the term frequencies for each term in each document
term_freqs = {}

# Calculate term frequencies
for file_id in inaugural_files:
    words = inaugural.words(file_id)
    term_freqs[file_id] = nltk.FreqDist(words)

# Define the queries
queries = [
    "government",
    "freedom",
    "America",
    "citizens",
    "constitution"
]

# Retrieve the ranked list of the first 5 most similar inaugural addresses for each query and each value of mu
for mu in mu_values:
    print(f"Results for mu = {mu}:")
    for query in queries:
        query_results = []
        for file_id in inaugural_files:
            score = 0
            if query in term_freqs[file_id]:
                term_freq = term_freqs[file_id][query]
                doc_length = doc_lengths[file_id]
                collection_length = sum(doc_lengths.values())
                score = dirichlet_score(term_freq, doc_length, collection_length, mu)
            query_results.append((file_id, score))
        
        # Sort the results by score in descending order
        query_results.sort(key=lambda x: x[1], reverse=True)
        
        # Display the top 5 results for each query
        top_5_results = query_results[:5]
        print(f"Query: '{query}'")
        for result in top_5_results:
            print(f"Document: {result[0]}, Score: {result[1]:.4f}")
    
        # Compute precision in the top 5
        relevant_docs = [result[0] for result in top_5_results]
        precision = len(set(relevant_docs).intersection(set(inaugural_files[:5]))) / 5
        print(f"Precision in top 5: {precision:.2f}")
    print("\n")


# In[ ]:




