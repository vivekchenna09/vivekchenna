{\rtf1\ansi\ansicpg1252\cocoartf2757
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;\red224\green232\blue240;\red12\green14\blue18;}
{\*\expandedcolortbl;;\cssrgb\c90196\c92941\c95294;\cssrgb\c5098\c6667\c9020;}
\paperw11900\paperh16840\margl1440\margr1440\vieww25100\viewh15700\viewkind0
\deftab720
\pard\pardeftab720\sa320\partightenfactor0

\f0\fs32 \cf2 \cb3 \expnd0\expndtw0\kerning0
  5) #Add length normalization to your methods. Discuss any changes you see in the the top 5.\
Precision Values:\
Precision is a measure of how accurate a retrieval system is in retrieving relevant documents. In our analysis, we computed precision for two different similarity methods: BM25 and Cosine Similarity. Here are the precision values for each query and method:\
BM25 with Length Normalization:\
For the query "freedom of speech," BM25 with length normalization achieved a precision of 0.20, indicating that 20% of the top 5 ranked documents were relevant to the query.\
For the query "economic prosperity," BM25 with length normalization did not return any relevant documents in the top 5, resulting in a precision of 0.00.\
For the query "national security," BM25 with length normalization also achieved a precision of 0.20, with 20% of the top 5 documents being relevant.\
Cosine Similarity with Length Normalization:\
Cosine Similarity with length normalization did not perform well for any of the queries. It did not return any relevant documents in the top 5 for "freedom of speech," "economic prosperity," or "national security," resulting in a precision of 0.00 for all queries.\
Impact of Length Normalization:\
Length normalization is a technique used to adjust similarity scores based on document length. It can affect the ranking of documents in the top results. Here's what we observed:\
BM25 with Length Normalization: For some queries, BM25 with length normalization may have caused minor changes in the ranking of documents in the top 5 compared to BM25 without normalization. However, the overall performance in terms of precision remained consistent.\
Cosine Similarity with Length Normalization: Even with length normalization, Cosine Similarity still did not perform well and did not return relevant documents in the top 5 for any query.\
\
\
6). #Compare and discuss the precision results obtained by the two similarity functions you chose.\
BM25 Precision:\
For the query "freedom of speech," BM25 achieved a precision of 0.20, which means that out of the top 5 documents it retrieved, two were relevant to the query.\
For the query "economic prosperity," BM25 didn't manage to find any relevant documents in the top 5, resulting in a precision of 0.00.\
In the case of "national security," BM25 also achieved a precision of 0.20, indicating that two out of the top 5 documents were related to the query.\
Cosine Similarity Precision:\
Cosine Similarity didn't perform well in terms of precision for any of the queries. For all three queries ("freedom of speech," "economic prosperity," and "national security"), it couldn't retrieve any relevant documents in the top 5, resulting in a precision of 0.00 for all queries.\
Comparison and Discussion:\
The precision results provide insights into how well each similarity function performed in retrieving relevant documents:\
BM25 demonstrated reasonably good precision for certain queries, particularly for topics related to politics and national security. It managed to find relevant documents for "freedom of speech" and "national security" with a precision of 0.20. However, it struggled to retrieve relevant documents for the "economic prosperity" query.\
Cosine Similarity, on the other hand, had a tough time retrieving relevant documents for all queries, resulting in a precision of 0.00 for each query. This suggests that Cosine Similarity may not be suitable for this specific corpus or set of queries.\
These results highlight the importance of choosing the right similarity function for a given information retrieval task. In this case, BM25 proved to be more effective in certain contexts, while Cosine Similarity did not perform well for these specific queries and documents.\
It's worth noting that optimizing retrieval methods, adjusting parameters, or considering different preprocessing techniques could potentially improve performance. Additionally, the nature of the corpus and the specific queries play a significant role in determining which similarity function is more effective for a given task.\
}