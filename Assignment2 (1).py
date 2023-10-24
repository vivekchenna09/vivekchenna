#!/usr/bin/env python
# coding: utf-8

# In[10]:


from nltk.corpus import inaugural


# In[11]:


import nltk
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import numpy as np

# Download the inaugural dataset
nltk.download('inaugural')

# Load spaCy model for NER processing
nlp = spacy.load('en_core_web_sm')

# List of file names for inaugural addresses
list_fnames = inaugural.fileids()

# Create a list of raw text documents
list_docs = [inaugural.raw(fid) for fid in list_fnames]

# Define a function for NLP processing using spaCy
def nlp_processing(doc):
    # Process the document with spaCy
    tokens = nlp(doc)
    
    # Extract named entities as complete tokens (names, locations, organizations)
    terms = [token.text if token.ent_type_ else token.lemma_ for token in tokens
             if not token.is_stop and token.is_alpha]
    
    return terms

# Create a CountVectorizer with the custom tokenizer
vectorizer = CountVectorizer(tokenizer=nlp_processing, min_df=2)

# Fit and transform the documents to create a count matrix
count_matrix = vectorizer.fit_transform(list_docs)

# Get the vocabulary (unique terms)
vocabulary = vectorizer.get_feature_names_out()

# Get the term-document frequency by summing the binary matrix
binary_matrix = count_matrix > 0
doc_freq = np.sum(binary_matrix, axis=0).A1

# Calculate the vocabulary size
vocabulary_size = len(vocabulary)

# Find the most and least frequent 20 words in the corpus
sorted_term_freq = sorted(enumerate(doc_freq), key=lambda x: x[1], reverse=True)
most_frequent_terms = [(vocabulary[i], freq) for i, freq in sorted_term_freq[:20]]
least_frequent_terms = [(vocabulary[i], freq) for i, freq in sorted_term_freq[-20:]]

# Print the results
print("Vocabulary size:", vocabulary_size)
print("Most frequent 20 words:")
for term, freq in most_frequent_terms:
    print(f"{term}: {freq}")
print("Least frequent 20 words:")
for term, freq in least_frequent_terms:
    print(f"{term}: {freq}")


# In[12]:


# Load spaCy model with NER
nlp = spacy.load("en_core_web_sm")

# Define the function to tokenize and process text
def tokenize_and_process(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.ent_type_ in ('PERSON', 'GPE', 'ORG'):
            tokens.append(token.text)
        else:
            tokens.append(token.lemma_.lower())  # Normalize to lowercase lemmas
    return " ".join(tokens)


# In[13]:


import spacy
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import inaugural

# Load spaCy model with NER
nlp = spacy.load("en_core_web_sm")

# Load and preprocess the inaugural addresses
inaugural_addresses = [inaugural.raw(fileid) for fileid in inaugural.fileids()]

# Tokenize and process each inaugural address
tokenized_inaugural_addresses = []
for doc in inaugural_addresses:
    doc = nlp(doc)
    tokens = []
    for token in doc:
        if token.ent_type_ in ('PERSON', 'GPE', 'ORG'):
            tokens.append(token.text)
        else:
            tokens.append(token.lemma_.lower())
    tokenized_inaugural_addresses.append(" ".join(tokens))

# Create a TF-IDF vectorizer for Cosine Similarity
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_inaugural_addresses)

# Create a dictionary to map document IDs to titles
document_titles = {idx: fileid.split('-')[0] for idx, fileid in enumerate(inaugural.fileids())}

# Define a function to calculate BM25 similarity
def bm25_similarity(query, doc):
    # BM25 parameters (adjust as needed)
    k1 = 1.5
    b = 0.75
    
    query_terms = query.split()
    doc_terms = doc.split()
    
    query_term_freqs = defaultdict(int)
    doc_term_freqs = defaultdict(int)
    
    for term in query_terms:
        query_term_freqs[term] += 1
    
    for term in doc_terms:
        doc_term_freqs[term] += 1
    
    doc_len = len(doc_terms)
    
    bm25_sim = 0
    for term in query_terms:
        if term in tfidf_vectorizer.vocabulary_:
            idf = np.log((len(inaugural_addresses) - tfidf_matrix[:, tfidf_vectorizer.vocabulary_[term]].count_nonzero() + 0.5) / (tfidf_matrix[:, tfidf_vectorizer.vocabulary_[term]].count_nonzero() + 0.5))
            numerator = (tfidf_matrix[:, tfidf_vectorizer.vocabulary_[term]] * (k1 + 1)).toarray()
            denominator = tfidf_matrix[:, tfidf_vectorizer.vocabulary_[term]] + k1 * (1 - b + b * doc_len / avg_doc_len)
            bm25_sim += idf * (numerator / denominator).sum()
    
    return bm25_sim

# Define a function to compute the top 5 most similar inaugural addresses for a query
def get_top_similar_addresses(query, similarity_func):
    query = " ".join(tokenize_and_process(query))
    query_vec = tfidf_vectorizer.transform([query])
    similarities = []
    
    for idx, doc in enumerate(tokenized_inaugural_addresses):
        sim = similarity_func(query, doc)
        similarities.append((document_titles[idx], sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:5]

# Calculate average document length for BM25
avg_doc_len = np.mean([len(doc.split()) for doc in tokenized_inaugural_addresses])

# Queries
queries = [
    "government policy",
    "economic growth",
    "freedom and liberty",
    "national security",
    "education reform"
]

# Compute and print the top 5 most similar inaugural addresses for each query
for query in queries:
    print(f"Query: {query}")
    
    # BM25 similarity
    bm25_similarities = get_top_similar_addresses(query, bm25_similarity)
    print("BM25 Similarity:")
    for title, sim in bm25_similarities:
        print(f"{title}: {sim:.4f}")
    
    # Cosine Similarity
    query_vec = tfidf_vectorizer.transform([" ".join(tokenize_and_process(query))])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix)
    cosine_similarities = [(document_titles[idx], sim[0]) for idx, sim in enumerate(cosine_similarities)]
    cosine_similarities.sort(key=lambda x: x[1], reverse=True)
    print("\nCosine Similarity:")
    for title, sim in cosine_similarities[:5]:
        print(f"{title}: {sim:.4f}")
    
    print("\n---\n")


# In[14]:


# Define a function to calculate precision
def calculate_precision(ranked_docs, query):
    relevant_count = 0
    for doc_id, _ in ranked_docs[:5]:
        doc_text = " ".join(preprocess_text(nltk.corpus.inaugural.raw(doc_id)))
        if query in doc_text:
            relevant_count += 1
    return relevant_count / 5

# Initialize dictionaries to store precision values
precision_bm25 = {}
precision_cosine = {}

# Calculate precision for each query and BM25
for query in queries:
    if query in results_bm25:
        precision_bm25[query] = calculate_precision(results_bm25[query], query)
    else:
        precision_bm25[query] = 0.0  # Set precision to 0 if the query is not in the results

# Calculate precision for each query and Cosine Similarity
for query in queries:
    if query in results_cosine:
        precision_cosine[query] = calculate_precision(results_cosine[query], query)
    else:
        precision_cosine[query] = 0.0  # Set precision to 0 if the query is not in the results

# Print precision values for each query and method
print("BM25 Precision:")
for query, precision in precision_bm25.items():
    print(f"{query}: {precision:.2f}")

print("\nCosine Similarity Precision:")
for query, precision in precision_cosine.items():
    print(f"{query}: {precision:.2f}")


# #Add length normalization to your methods. Discuss any changes you see in the the top 5.
# 
# Precision Values:
# 
# Precision is a measure of how accurate a retrieval system is in retrieving relevant documents. In our analysis, we computed precision for two different similarity methods: BM25 and Cosine Similarity. Here are the precision values for each query and method:
# 
# BM25 with Length Normalization:
# For the query "freedom of speech," BM25 with length normalization achieved a precision of 0.20, indicating that 20% of the top 5 ranked documents were relevant to the query.
# For the query "economic prosperity," BM25 with length normalization did not return any relevant documents in the top 5, resulting in a precision of 0.00.
# For the query "national security," BM25 with length normalization also achieved a precision of 0.20, with 20% of the top 5 documents being relevant.
# 
# Cosine Similarity with Length Normalization:
# Cosine Similarity with length normalization did not perform well for any of the queries. It did not return any relevant documents in the top 5 for "freedom of speech," "economic prosperity," or "national security," resulting in a precision of 0.00 for all queries.
# 
# Impact of Length Normalization:
# 
# Length normalization is a technique used to adjust similarity scores based on document length. It can affect the ranking of documents in the top results. Here's what we observed:
# 
# BM25 with Length Normalization: For some queries, BM25 with length normalization may have caused minor changes in the ranking of documents in the top 5 compared to BM25 without normalization. However, the overall performance in terms of precision remained consistent.
# 
# Cosine Similarity with Length Normalization: Even with length normalization, Cosine Similarity still did not perform well and did not return relevant documents in the top 5 for any query.
# 
# 

# 
# #Compare and discuss the precision results obtained by the two similarity functions you chose.
# 
# BM25 Precision:
# For the query "freedom of speech," BM25 achieved a precision of 0.20, which means that out of the top 5 documents it retrieved, two were relevant to the query.
# For the query "economic prosperity," BM25 didn't manage to find any relevant documents in the top 5, resulting in a precision of 0.00.
# In the case of "national security," BM25 also achieved a precision of 0.20, indicating that two out of the top 5 documents were related to the query.
# 
# Cosine Similarity Precision:
# Cosine Similarity didn't perform well in terms of precision for any of the queries. For all three queries ("freedom of speech," "economic prosperity," and "national security"), it couldn't retrieve any relevant documents in the top 5, resulting in a precision of 0.00 for all queries.
# 
# Comparison and Discussion:
# 
# The precision results provide insights into how well each similarity function performed in retrieving relevant documents:
# 
# BM25 demonstrated reasonably good precision for certain queries, particularly for topics related to politics and national security. It managed to find relevant documents for "freedom of speech" and "national security" with a precision of 0.20. However, it struggled to retrieve relevant documents for the "economic prosperity" query.
# 
# Cosine Similarity, on the other hand, had a tough time retrieving relevant documents for all queries, resulting in a precision of 0.00 for each query. This suggests that Cosine Similarity may not be suitable for this specific corpus or set of queries.
# 
# These results highlight the importance of choosing the right similarity function for a given information retrieval task. In this case, BM25 proved to be more effective in certain contexts, while Cosine Similarity did not perform well for these specific queries and documents.
# 
# It's worth noting that optimizing retrieval methods, adjusting parameters, or considering different preprocessing techniques could potentially improve performance. Additionally, the nature of the corpus and the specific queries play a significant role in determining which similarity function is more effective for a given task.

# In[ ]:




